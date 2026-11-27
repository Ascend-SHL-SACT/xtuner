"""GLM-5.3 kpool aclnn_v2 indexer ``_SCORE_ROW_CHUNK=0``（whole-shard 布局）回归测试。

回归背景（job dyb-q35-xtuner-llm-23 事故）：512 卡 llm 任务全 rank 在 step-1 前向即崩。
``_parse_row_chunk`` 的解析契约是 ``XTUNER_KPOOL_SCORE_ROW_CHUNK <= 0`` 语义为 whole-shard
布局（返回 0，见 ``kpool_npu._parse_row_chunk`` docstring），而修复前的
``_score_and_topk_aclnn_v2`` 无条件 ``range(c_start, row_hi, _SCORE_ROW_CHUNK)``：
chunk=0 时 Python 直接抛 ``ValueError: range() arg 3 must not be zero``——解析层承诺的
"<=0 = whole-shard" 在执行层不成立，512 个 rank 全部在第一个 indexer 调用上炸掉。
修复（本仓库工作区未提交改动）：``chunk <= 0`` 时每段取
``chunk = max(row_hi - c_start, 1)``，即每段单次迭代覆盖整段 ``[c_start, row_hi)``；
``max(.., 1)`` 兜住 fully-dense 段（``c_start == row_hi``，空 range，与 chunked 布局一致）。

修复语义（kpool_npu.py 内 XTUNER_KPOOL_CAUSAL_CROP / whole-shard 相关注释）：per-row 分数
与 topk 不依赖其它行，row-chunk 只改变 (document, row chunk) item 的行数切分，不改任何
一行可见池集合内的打分与选择——selected pool-id 集合与 whole-shard 路径逐位一致
（causal crop 契约：causal 不可见格在两种布局下都保持 ``finfo.min`` 预填；可见格分数只由
该行自己与其可见键决定）。唯一的实现定义空间是"行可见池数 < select_k 时的 pad 列选取"，
而 aclnn_v2 路径的 dense-prefix exclusion 把所有 ``prefix <= select_k`` 的行走纯算术
dense fill（闭式解），op 行 ``prefix > select_k`` 恒满宽有效——两个布局在该路径上都没有
实现定义的 pad 空间，因此逐位等价是强契约而非松约定。

TestAclnnV2ScoreRowChunkZero（NPU，直接调 ``_score_and_topk_aclnn_v2`` 生产契约输入）
    test_chunk_zero_returns_valid_selection_without_value_error: chunk=0 不抛 ValueError，
        返回 ``[S, select_k]`` int32；逐行锁定前缀语义——``prefix <= select_k`` 的 dense 行
        等于闭式解 ``[0, prefix) + _SEL_PAD``，``prefix > select_k`` 的 op 行全 ``select_k``
        个 id 落在 ``[0, prefix)`` 且互异（修复前代码在同一输入上必抛 range step=0）。
    test_chunk_zero_bitwise_equal_chunked_layout: 同一输入下 chunk=0（whole-shard）与
        chunk in {1024, 512}（chunked）输出 ``torch.equal`` 逐位一致；512 额外制造文档中段
        的 item 缝（2048/1024/512 行文档均被切多刀）。
TestAclnnV2DispatchChunkZero（NPU，公开入口 ``npu_kpool_topk_indices``）
    test_public_entry_chunk_zero_smoke_and_chunked_parity: 门开（``_INDEXER_ACLNN_V2=True``）
        时 chunk=0 经公开入口可用，spy 确证恰走 aclnn_v2 分支；输出 ``[S, 1, width]`` int32，
        逐行锁定展开契约（有效槽 ``min(prefix, select_k)*kpool`` 个、token id 全落在本文档
        ``[doc_start, doc_start + prefix*kpool)`` 且行内互异）；chunk=0 与 chunk=1024 的
        公开输出逐位一致。
TestBatchedScoreRowChunkZeroBaseline（NPU，对照基线：triton 批量路径 ``row_chunk`` 参数）
    test_long_documents_row_chunk_zero_bitwise_equal_chunked: row_chunk=0 与 {1024, 512}
        逐位一致（全文档 >= 300 token，每行 ``prefix >= select_k``，无实现定义 pad 行）。
    test_mixed_documents_valid_slot_sets_match_prefix_closed_form: 混合长短文档下
        row_chunk=0 与 1024 的每行有效槽集合（``sel < row_prefix``）：``prefix <= select_k``
        的行等于闭式解 ``[0, prefix)``（可见列分数高于 ``finfo.min`` 预填，必进 topk；pad 列
        全在 prefix 之外被拒），``prefix > select_k`` 的行恰 select_k 个互异 id 且全在
        ``[0, prefix)``（topk 只取可见列中分数最高者）——两种布局集合级契约一致。
TestAclnnV2MidDocShardChunkZero（NPU，【mid-doc SP shard 补充】global cu + ``shard_start``
        构造——生产 SP32 下 ``_segment_ranges`` 给出的 ranges 全部是文档中段 shard 的真实
        形状：row 侧 shard-local、``doc_start``/``pool_lo`` global，``d0 = shard_start +
        row_lo - doc_start`` 锚定；以上 6 用例全部 shard_start=0，mid-doc 几何此前只靠
        端到端训练间接覆盖）
    test_mid_doc_shard_dense_prefix_truncated_bitwise_equal_chunked: shard 起点落在文档
        doc-local 50（< dense_row_cap=67）且 dense 前缀在 shard 内部截断于 cap——chunk=0 的
        single item 从 ``c_start = cap - d0`` 锚定，dense fill 行（闭式解 ``[0, prefix)`` +
        pad，前缀从文档头累计：shard 行 0 已见 12 池）与 op 行的全部槽位同 chunk={1024,512}
        （512 在 op 区切出多道 mid-doc item 缝）逐位相等。
    test_mid_doc_shard_fully_dense_block_empty_range_bitwise_equal_chunked: mid-doc 首块
        整块全 dense（doc-local [40,60) 全部 ``prefix <= select_k``，``dense_end_doc ==
        row_hi``）——chunk=0 走 ``max(row_hi - c_start, 1)`` 空 range 分支（修复前构造
        ``range(step=0)`` 在空 range 上同样抛 ValueError：Python 在 range 构造期校验步长），
        该块零 op item，输出（纯 dense）与 chunked 逐位相等；同 shard 后续整段可见文档
        覆盖 dense+op 混合与 item 缝。
    test_shard_both_ends_mid_doc_mixed_docs_bitwise_equal_chunked: shard 两端都落在文档
        中间（首块 d0=700 > cap：零 dense 行、``c_start == row_lo``、shard 行 0 已见 175 池；
        尾块只盖文档前 500 行，``floor_pools`` 只算 shard 可见的 125 池而非全文档 325 池）
        且 pack 内多文档混合（整段可见长文档、3 token 全 dense 微文档、被切文档）；
        chunk=0 与 {1024, 256}（256 在首/尾被切块内都制造 item 缝）逐位相等。
    test_mid_doc_shard_large_select_k_bitwise_equal_chunked: select_k in {64, 128} 的
        mid-doc shard（dense 前缀在 shard 内截断于 ``4*(select_k+1)-1`` = 259 / 515），
        chunk=0 与 {1024, 512} 逐位相等——回应 bitwise 等价验证仅 select_k=16 的规模缺口。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import pytest
import torch
from torch import Tensor

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.sparse_mla import kpool_npu
from xtuner.v1.utils.device import get_device


try:
    from xtuner.v1.ops.sparse_mla.kpool_npu import lightning_indexer_score_into

    _HAS_FUSED_INDEXER = lightning_indexer_score_into is not None
except ImportError:  # triton-ascend 缺失（纯 CPU CI）: 批量 triton 路径不可用
    lightning_indexer_score_into = None  # type: ignore[assignment]
    _HAS_FUSED_INDEXER = False


DEVICE = get_device()  # 本盒为 "npu"；纯 CPU 环境退化为 "cpu"
_requires_accelerator = pytest.mark.skipif(DEVICE == "cpu", reason="requires an NPU/CUDA accelerator")
_requires_fused_indexer = pytest.mark.skipif(
    not _HAS_FUSED_INDEXER, reason="requires the fused lightning indexer kernel (triton-ascend)"
)

if DEVICE == "npu":
    torch.npu.set_device("npu:0")

# 生产契约规模：T=3969 行、长短文档混合。短文档（3/17/65 token）全走 dense fill 计划步，
# 长文档（2048/1024/512/300 token）的 doc-local p >= kpool*(select_k+1) 行走 op item 计划步，
# 两种 plan step 都覆盖；chunk=512 时长文档均被切成多个 (document, row chunk) item。
DOC_LENS = [2048, 3, 17, 1024, 65, 512, 300]
SELECT_K = 16  # index_topk=64 / index_kpool=4
INDEX_KPOOL = 4
INDEX_HEAD_DIM = 128
NUM_HEADS = 8
# 全文档 >= 300 token（每行 prefix >= 75 > select_k，无实现定义 pad 行）的对照布局。
LONG_DOC_LENS = [2048, 1024, 512, 300]
# 混合长短文档：段多、前缀参差（含 prefix=0 与 prefix<select_k 的 pad 行）。
MIXED_DOC_LENS = [200, 3, 17, 64, 65, 1, 129, 300, 7, 77]


@contextmanager
def _patched_attr(module: Any, name: str, value: Any) -> Iterator[None]:
    """临时改写模块级属性，try/finally 保证恢复（模块级 ``_SCORE_ROW_CHUNK`` 等门状态）。"""
    saved = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, saved)


def _make_ctx(doc_lens: list[int]) -> SequenceContext:
    """构造单 rank（无 SP mesh、``shard_start == 0``）的 packed ``SequenceContext``。"""
    cu = [0]
    for ln in doc_lens:
        cu.append(cu[-1] + ln)
    cu_q = torch.tensor(cu, dtype=torch.int32)
    return SequenceContext(
        input_ids=None,
        cu_seq_lens_q=cu_q,
        cu_seq_lens_k=cu_q,
        max_length_q=max(doc_lens),
        max_length_k=max(doc_lens),
        device=DEVICE,
    )


def _build_aclnn_v2_inputs(
    doc_lens: list[int], seed: int
) -> dict[str, Any]:
    """按 ``_score_and_topk_aclnn_v2`` 的入参契约一次性构造全部输入。

    Args:
        doc_lens (list[int]): 各文档 token 数（packing 顺序即文档顺序）。
        seed (int): 随机种子（randn bf16，几乎不产生精确 fp32 平分，平分序是实现定义的）。

    Returns:
        dict[str, Any]: ``q_local [S, NH, D]`` bf16 / ``pool_key [P_total, D]`` bf16 /
            ``w_all_f [S, NH]`` fp32（已 NH**-0.5 缩放）/ ``ctx`` / ``ranges``
            （``_segment_ranges`` 输出的 ``(row_lo, row_hi, pool_lo, pool_hi, doc_start)``）、
            ``row_prefix [S]`` int32。
    """
    torch.manual_seed(seed)
    total = sum(doc_lens)
    q_local = torch.randn(total, NUM_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    w_all_f = torch.randn(total, NUM_HEADS, dtype=torch.float32, device=DEVICE) * (NUM_HEADS**-0.5)
    num_pools = sum(-(-ln // INDEX_KPOOL) for ln in doc_lens)
    pool_key = torch.randn(num_pools, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    ctx = _make_ctx(doc_lens)
    ranges = kpool_npu._segment_ranges(ctx, total, INDEX_KPOOL)
    row_prefix, _row_doc_start, _slot_offsets, _prefix_host = kpool_npu._expand_meta(
        ctx, total, INDEX_KPOOL, DEVICE
    )
    return {
        "q_local": q_local,
        "pool_key": pool_key,
        "w_all_f": w_all_f,
        "ctx": ctx,
        "ranges": ranges,
        "row_prefix": row_prefix,
    }


def _run_aclnn_v2(inputs: dict[str, Any], chunk: int, select_k: int = SELECT_K) -> Tensor:
    """以指定 ``_SCORE_ROW_CHUNK`` 直接调 ``_score_and_topk_aclnn_v2`` 并同步。"""
    with _patched_attr(kpool_npu, "_SCORE_ROW_CHUNK", chunk):
        sel = kpool_npu._score_and_topk_aclnn_v2(
            inputs["q_local"],
            inputs["pool_key"],
            inputs["w_all_f"],
            inputs["ranges"],
            select_k,
            INDEX_HEAD_DIM,
            INDEX_KPOOL,
            inputs["ctx"],
            None,
        )
    if DEVICE != "cpu":
        getattr(torch, DEVICE).synchronize()
    return sel


def _assert_sel_prefix_contract(sel: Tensor, row_prefix: Tensor, *, label: str, select_k: int = SELECT_K) -> None:
    """锁定 aclnn_v2 输出的行级前缀语义（形状 / dtype / dense 闭式解 / op 行有效性）。

    Args:
        sel (Tensor): ``[S, select_k]`` int32 候选输出。
        row_prefix (Tensor): ``[S]`` 每行因果可见池数。
        label (str): 失败信息前缀（区分 chunk 布局）。
        select_k (int): 槽宽（默认模块级 ``SELECT_K``；大 select_k 用例显式传入）。
    """
    seq_len = row_prefix.shape[0]
    assert sel.shape == (seq_len, select_k), (
        f"{label}: sel shape {tuple(sel.shape)} != {(seq_len, select_k)}"
    )
    assert sel.dtype == torch.int32, f"{label}: sel dtype {sel.dtype} != torch.int32"

    dense_rows = 0
    op_rows = 0
    for r in range(seq_len):
        prefix = int(row_prefix[r])
        row = sel[r]
        valid = row[row < row_prefix[r]]
        expect = min(prefix, select_k)
        assert valid.numel() == expect, (
            f"{label}: row {r} prefix={prefix} valid slots {valid.numel()} != {expect}; "
            f"row={row.tolist()}"
        )
        if prefix <= select_k:
            # dense-prefix exclusion：该行走纯算术 dense fill，闭式解 [0, prefix) + _SEL_PAD。
            dense_rows += 1
            expect_row = [c if c < prefix else kpool_npu._SEL_PAD for c in range(select_k)]
            assert row.tolist() == expect_row, (
                f"{label}: dense row {r} prefix={prefix} closed-form mismatch; "
                f"got={row.tolist()} expect={expect_row}"
            )
        else:
            # op 行：prefix > select_k，select_k 个 id 必须全部可见且互异。
            op_rows += 1
            v64 = valid.to(torch.int64)
            assert v64.min() >= 0 and v64.max() < prefix, (
                f"{label}: op row {r} prefix={prefix} ids out of [0, {prefix}): {row.tolist()}"
            )
            assert torch.unique(v64).numel() == v64.numel(), (
                f"{label}: op row {r} prefix={prefix} duplicate ids: {row.tolist()}"
            )
    assert dense_rows > 0 and op_rows > 0, (
        f"{label}: layout must cover both plan steps, dense_rows={dense_rows} op_rows={op_rows}"
    )


@_requires_accelerator
@pytest.mark.gpu
class TestAclnnV2ScoreRowChunkZero:
    """【核心回归 + 等价性】``_score_and_topk_aclnn_v2`` 在 ``_SCORE_ROW_CHUNK <= 0`` 下
    可用、语义正确，且与 chunked 布局逐位一致（修复前必抛 range step=0）。"""

    def test_chunk_zero_returns_valid_selection_without_value_error(self):
        """chunk=0（whole-shard）直接调用不抛 ValueError，返回合法 sel 张量。

        回归断言：修复前代码无条件 ``range(c_start, row_hi, 0)`` 在同一输入上必抛
        ``ValueError: range() arg 3 must not be zero``（llm-23 事故根因）。
        """
        inputs = _build_aclnn_v2_inputs(DOC_LENS, seed=7)
        try:
            sel = _run_aclnn_v2(inputs, chunk=0)
        except ValueError as exc:
            pytest.fail(f"chunk=0 must not raise ValueError (regression), got: {exc!r}")
        _assert_sel_prefix_contract(sel, inputs["row_prefix"], label="chunk=0")

    @pytest.mark.parametrize("chunk", [1024, 512], ids=["chunked-1024", "chunked-512"])
    def test_chunk_zero_bitwise_equal_chunked_layout(self, chunk: int):
        """同一输入下 chunk=0（whole-shard）与 chunked 布局输出逐位一致。

        依据 causal crop 契约：per-row 分数与 topk 不依赖其它行，selected pool-id 集合与
        full-width path bit-identical；aclnn_v2 路径无实现定义 pad 空间（dense 行闭式解、
        op 行恒满宽），因此强逐位相等。512 额外在长文档中段制造 item 缝。
        """
        inputs = _build_aclnn_v2_inputs(DOC_LENS, seed=7)
        sel_whole = _run_aclnn_v2(inputs, chunk=0)
        sel_chunked = _run_aclnn_v2(inputs, chunk=chunk)
        assert torch.equal(sel_whole, sel_chunked), (
            f"chunk=0 vs chunk={chunk} outputs differ on identical inputs: "
            f"diff elems={(sel_whole != sel_chunked).sum().item()} / {sel_whole.numel()}; "
            f"first diff row="
            f"{(sel_whole != sel_chunked).any(dim=-1).nonzero().flatten()[:5].tolist()}"
        )


@_requires_accelerator
@pytest.mark.gpu
class TestAclnnV2DispatchChunkZero:
    """【分发 smoke】公开入口 ``npu_kpool_topk_indices`` 在 ``_INDEXER_ACLNN_V2=True`` 时
    chunk=0 同样可用（模块在 import 时读 env，进程内改模块属性即可）。"""

    def test_public_entry_chunk_zero_smoke_and_chunked_parity(self, monkeypatch: pytest.MonkeyPatch):
        """门开 + chunk=0 经公开入口可跑，spy 确证走 aclnn_v2 分支，输出契约成立，
        且与 chunk=1024 的公开输出逐位一致。"""
        torch.manual_seed(11)
        total = sum(DOC_LENS)
        q = torch.randn(1, total, NUM_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        k = torch.randn(1, total, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        gate_scores = torch.randn(1, total, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        weights = torch.randn(1, total, NUM_HEADS, dtype=torch.bfloat16, device=DEVICE)
        kpool_ape = torch.randn(INDEX_KPOOL, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        ctx = _make_ctx(DOC_LENS)
        topk_width = SELECT_K * INDEX_KPOOL  # packed 行宽（唯一布局，见 kpool_npu）

        cu = [int(v) for v in ctx.cu_seq_lens_q_list]
        doc_start_of = {}
        for r, tok in enumerate(range(total)):
            doc = max(i for i, c in enumerate(cu) if c <= tok)
            doc_start_of[r] = cu[doc]

        outputs: dict[int, Tensor] = {}
        # spy 闭包在打补丁前捕获原函数（不可在循环内重复捕获模块属性——monkeypatch 链会把
        # spy 自己包进 real，形成无限递归）。chunk=0 带 spy 确证派发；parity 跑前先还原。
        real = kpool_npu._score_and_topk_aclnn_v2
        calls: list[int] = []

        def _spy(*args: Any, **kwargs: Any) -> Tensor:
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(kpool_npu, "_score_and_topk_aclnn_v2", _spy)
        with _patched_attr(kpool_npu, "_INDEXER_ACLNN_V2", True), _patched_attr(kpool_npu, "_SCORE_ROW_CHUNK", 0):
            out0 = kpool_npu.npu_kpool_topk_indices(
                q, k, gate_scores, weights, kpool_ape, ctx,
                index_head_dim=INDEX_HEAD_DIM, index_topk=SELECT_K * INDEX_KPOOL, index_kpool=INDEX_KPOOL,
            )
        if DEVICE != "cpu":
            getattr(torch, DEVICE).synchronize()
        assert sum(calls) >= 1, "chunk=0: aclnn_v2 scorer not dispatched via the public entry"
        outputs[0] = out0

        monkeypatch.setattr(kpool_npu, "_score_and_topk_aclnn_v2", real)  # parity 跑不再 spy
        with _patched_attr(kpool_npu, "_INDEXER_ACLNN_V2", True), _patched_attr(kpool_npu, "_SCORE_ROW_CHUNK", 1024):
            outputs[1024] = kpool_npu.npu_kpool_topk_indices(
                q, k, gate_scores, weights, kpool_ape, ctx,
                index_head_dim=INDEX_HEAD_DIM, index_topk=SELECT_K * INDEX_KPOOL, index_kpool=INDEX_KPOOL,
            )
        if DEVICE != "cpu":
            getattr(torch, DEVICE).synchronize()

        for chunk, out in outputs.items():
            label = f"chunk={chunk}"
            assert out.shape == (total, 1, topk_width), (
                f"{label}: out shape {tuple(out.shape)} != {(total, 1, topk_width)}"
            )
            assert out.dtype == torch.int32, f"{label}: out dtype {out.dtype} != torch.int32"
            flat = out[:, 0, :]
            for r in range(total):
                # packed 行契约：读窗 [0, min(local_pos + 1, topk)) 稠密因果去重、窗外全 -1。
                lo = doc_start_of[r]
                gpos = r  # shard_start = 0，global token 即行号
                span = min(gpos - lo + 1, topk_width)
                valid = flat[r][flat[r] >= 0]
                assert valid.numel() == span, (
                    f"{label}: row {r} valid token slots {valid.numel()} != read span {span}"
                )
                assert bool(((valid >= lo) & (valid <= gpos)).all()), (
                    f"{label}: row {r} token ids outside causal doc span [{lo}, {gpos}]: "
                    f"{valid.tolist()}"
                )
                assert torch.unique(valid).numel() == valid.numel(), (
                    f"{label}: row {r} duplicate token ids"
                )

        assert torch.equal(outputs[0], outputs[1024]), (
            "public-entry chunk=0 vs chunk=1024 outputs differ: "
            f"diff elems={(outputs[0] != outputs[1024]).sum().item()} / {outputs[0].numel()}"
        )


@_requires_accelerator
@_requires_fused_indexer
@pytest.mark.gpu
class TestBatchedScoreRowChunkZeroBaseline:
    """【可选对照基线】triton 批量路径 ``_score_and_topk_batched`` 的 ``row_chunk`` 参数
    一直支持 0（whole-shard）：chunk=0 与 chunked 输出一致，作为 aclnn_v2 契约的参照。"""

    def test_long_documents_row_chunk_zero_bitwise_equal_chunked(self):
        """全长文档（每行 prefix >= select_k，无实现定义 pad 行）下 row_chunk=0 与
        {1024, 512} 输出逐位一致。"""
        torch.manual_seed(13)
        total = sum(LONG_DOC_LENS)
        q_local = torch.randn(total, NUM_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        w_all_f = torch.randn(total, NUM_HEADS, dtype=torch.float32, device=DEVICE) * (NUM_HEADS**-0.5)
        num_pools = sum(-(-ln // INDEX_KPOOL) for ln in LONG_DOC_LENS)
        pool_key = torch.randn(num_pools, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        ctx = _make_ctx(LONG_DOC_LENS)
        ranges = kpool_npu._segment_ranges(ctx, total, INDEX_KPOOL)
        row_prefix, _row_doc_start, _slot_offsets, prefix_host = kpool_npu._expand_meta(
            ctx, total, INDEX_KPOOL, DEVICE
        )

        outs: dict[int, Tensor] = {}
        with _patched_attr(kpool_npu, "_CAUSAL_CROP", True):
            for chunk in (0, 1024, 512):
                outs[chunk] = kpool_npu._score_and_topk_batched(
                    q_local, pool_key, w_all_f, ranges, row_prefix, prefix_host, SELECT_K, INDEX_HEAD_DIM,
                    row_chunk=chunk,
                )
        if DEVICE != "cpu":
            getattr(torch, DEVICE).synchronize()

        for chunk in (1024, 512):
            assert outs[chunk].shape == outs[0].shape, (
                f"row_chunk={chunk}: shape {tuple(outs[chunk].shape)} != "
                f"{tuple(outs[0].shape)}"
            )
            assert torch.equal(outs[0], outs[chunk]), (
                f"row_chunk=0 vs {chunk} differ: "
                f"diff elems={(outs[0] != outs[chunk]).sum().item()} / {outs[0].numel()}"
            )

    def test_mixed_documents_valid_slot_sets_match_prefix_closed_form(self):
        """混合长短文档下 row_chunk=0 与 1024 的每行有效槽集合一致且满足前缀契约：
        ``prefix <= select_k`` 等于闭式解 ``[0, prefix)``（pad 行的 pad 列选取是实现定义的，
        故只锁集合）；``prefix > select_k`` 为 select_k 个互异可见 id。"""
        torch.manual_seed(17)
        total = sum(MIXED_DOC_LENS)
        q_local = torch.randn(total, NUM_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        w_all_f = torch.randn(total, NUM_HEADS, dtype=torch.float32, device=DEVICE) * (NUM_HEADS**-0.5)
        num_pools = sum(-(-ln // INDEX_KPOOL) for ln in MIXED_DOC_LENS)
        pool_key = torch.randn(num_pools, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        ctx = _make_ctx(MIXED_DOC_LENS)
        ranges = kpool_npu._segment_ranges(ctx, total, INDEX_KPOOL)
        row_prefix, _row_doc_start, _slot_offsets, prefix_host = kpool_npu._expand_meta(
            ctx, total, INDEX_KPOOL, DEVICE
        )

        outs: dict[int, Tensor] = {}
        with _patched_attr(kpool_npu, "_CAUSAL_CROP", True):
            for chunk in (0, 1024):
                outs[chunk] = kpool_npu._score_and_topk_batched(
                    q_local, pool_key, w_all_f, ranges, row_prefix, prefix_host, SELECT_K, INDEX_HEAD_DIM,
                    row_chunk=chunk,
                )
        if DEVICE != "cpu":
            getattr(torch, DEVICE).synchronize()

        for chunk, sel in outs.items():
            for r in range(total):
                prefix = int(row_prefix[r])
                valid = sel[r][sel[r] < row_prefix[r]].to(torch.int64)
                if prefix <= SELECT_K:
                    # prefix <= select_k：全部可见列分数高于 finfo.min 预填，必进 topk，
                    # pad 列（finfo.min 平分）全部落在 prefix 之外被有效性测试拒掉——
                    # 有效集合闭式解 [0, prefix)，与行序/pad 选取无关。
                    expect = torch.arange(prefix, device=sel.device, dtype=torch.int64)
                    assert torch.equal(torch.sort(valid).values, expect), (
                        f"row_chunk={chunk}: row {r} prefix={prefix} valid-id set mismatch: "
                        f"got {valid.tolist()} expect {expect.tolist()}"
                    )
                else:
                    # prefix > select_k：topk 只取可见列里分数最高的 select_k 个——
                    # 具体取哪几个由分数决定，但必互异且全在 [0, prefix)。
                    assert valid.numel() == SELECT_K, (
                        f"row_chunk={chunk}: row {r} prefix={prefix} valid count "
                        f"{valid.numel()} != {SELECT_K}"
                    )
                    assert bool((valid >= 0).all()) and bool((valid < prefix).all()), (
                        f"row_chunk={chunk}: row {r} ids outside [0, {prefix}): "
                        f"{valid.tolist()}"
                    )
                    assert torch.unique(valid).numel() == valid.numel(), (
                        f"row_chunk={chunk}: row {r} duplicate ids: {valid.tolist()}"
                    )


# ---------------------------------------------------------------------------
# mid-doc SP shard 几何（以上 6 用例全部 shard_start=0）。生产 SP32 下 ``cu_seq_lens_q``
# 是 global pack 的累计长度、``shard_start`` 定位本 rank 的 global 起点，``_segment_ranges``
# 给出的 ``(row_lo, row_hi, pool_lo, pool_hi, doc_start)`` 中 row 侧 shard-local、
# ``doc_start``/``pool_lo`` global，whole-shard item 的 ``chunk_end_doc = d0 +
# (c_hi - row_lo)``（``d0 = shard_start + row_lo - doc_start``）锚定正是这一几何下
# 最要害的语义：mid-doc shard 的 dense 前缀可能被 ``dense_row_cap`` 在 shard 内截断、
# 整块全 dense（空 range）、或两端都落在文档中间。
# ---------------------------------------------------------------------------
TRUNC_DOC_LENS = [2048, 500, 300]            # shard 从 doc-local 50 进入：dense 前缀在 cap=67 截断
FULLY_DENSE_MID_DOC_LENS = [60, 500, 800]    # shard 从 doc-local 40 进入 20 行短文档：首块整块全 dense
BOTH_ENDS_DOC_LENS = [1024, 2048, 3, 1300]   # 前端 doc-local 700 进入（> cap），尾部只盖末文档前 500 行
SHARD_START_TRUNC = 50
SHARD_START_FULLY_DENSE = 40
SHARD_START_BOTH_ENDS = 700


def _make_shard_ctx(global_doc_lens: list[int], shard_start: int, shard_len: int) -> SequenceContext:
    """构造 mid-doc SP shard 的 packed ``SequenceContext``。

    ``cu_seq_lens_q`` 是 global pack 的累计长度，``shard_start`` 定位本 rank shard 的
    global 起点（shard-local 行 ``r`` 的 global token 是 ``shard_start + r``）——生产
    SP 分片下 ``_segment_ranges``/``_expand_meta`` 据此把每行锚回其文档局部位置。
    """
    cu = [0]
    for ln in global_doc_lens:
        cu.append(cu[-1] + ln)
    cu_q = torch.tensor(cu, dtype=torch.int32)
    return SequenceContext(
        input_ids=None,
        cu_seq_lens_q=cu_q,
        cu_seq_lens_k=cu_q,
        max_length_q=max(global_doc_lens),
        max_length_k=max(global_doc_lens),
        device=DEVICE,
        shard_start=shard_start,
    )


def _build_aclnn_v2_shard_inputs(
    global_doc_lens: list[int], shard_start: int, shard_len: int, seed: int
) -> dict[str, Any]:
    """按 mid-doc shard 几何构造 ``_score_and_topk_aclnn_v2`` 的全部入参。

    Args:
        global_doc_lens (list[int]): global pack 的各文档 token 数。
        shard_start (int): 本 rank shard 的 global 起点（落在某文档内部）。
        shard_len (int): 本 rank shard 行数（shard-local 张量的第 0 维）。
        seed (int): 随机种子（randn bf16，几乎不产生精确 fp32 平分）。

    Returns:
        dict[str, Any]: ``q_local``/``w_all_f`` 为 shard-local（只含本 rank 的
            ``shard_len`` 行）；``pool_key`` 覆盖 global 全 pack 池（生产语义：key 侧先
            SP gather 再整 pack 池化，``_segment_ranges`` 累计的 ``pool_lo`` 是 global
            池号）；``ranges``/``row_prefix`` 由生产函数从 (global cu, shard_start)
            推导，不经测试侧任何手工改写。
    """
    torch.manual_seed(seed)
    q_local = torch.randn(shard_len, NUM_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    w_all_f = torch.randn(shard_len, NUM_HEADS, dtype=torch.float32, device=DEVICE) * (NUM_HEADS**-0.5)
    num_pools = sum(-(-ln // INDEX_KPOOL) for ln in global_doc_lens)
    pool_key = torch.randn(num_pools, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    ctx = _make_shard_ctx(global_doc_lens, shard_start, shard_len)
    ranges = kpool_npu._segment_ranges(ctx, shard_len, INDEX_KPOOL)
    row_prefix, _row_doc_start, _slot_offsets, _prefix_host = kpool_npu._expand_meta(
        ctx, shard_len, INDEX_KPOOL, DEVICE
    )
    assert 0 < shard_start < sum(global_doc_lens) and ranges, (
        "shard geometry must sit strictly inside the pack and intersect at least one document"
    )
    return {
        "q_local": q_local,
        "pool_key": pool_key,
        "w_all_f": w_all_f,
        "ctx": ctx,
        "ranges": ranges,
        "row_prefix": row_prefix,
    }


def _first_op_row(row_prefix: Tensor, select_k: int) -> int | None:
    """第一个可见池数超过 ``select_k``（计划中走 op item）的 shard 行号；全 dense 为 None。

    生产 plan 里 ``c_start``（首个 op 行）由 ``dense_end_doc = min(d0 + 块长,
    dense_row_cap)`` 锚定；mid-doc shard（``d0 > 0``）时该边界落在 shard 行
    ``dense_row_cap - d0``——用 row_prefix 的直接扫描交叉验证 plan 几何。
    """
    for r, p in enumerate(row_prefix.tolist()):
        if p > select_k:
            return r
    return None


@_requires_accelerator
@pytest.mark.gpu
class TestAclnnV2MidDocShardChunkZero:
    """【mid-doc SP shard 补充】同一 shard 输入上 chunk=0（whole-shard）与 chunked 布局
    逐位一致，并用 row_prefix 标量锚锁定 dense 截断 / 全 dense / 两端切文档的 plan 几何
    （修复前任意 chunk=0 调用——包括空 range 的全 dense 块——都抛 ``range() arg 3 must
    not be zero``）。"""

    def test_mid_doc_shard_dense_prefix_truncated_bitwise_equal_chunked(self):
        """shard 从文档 doc-local 50 进入（< dense_row_cap=67）：dense 前缀在 shard 内部
        截断于 cap，chunk=0 的 single item 从 ``c_start = cap - d0`` 锚定，输出与
        chunk={1024, 512} 全槽位逐位相等（含 dense fill 行与 op 行）。"""
        shard_len = sum(TRUNC_DOC_LENS) - SHARD_START_TRUNC  # doc0 余 1998 行 + doc1/doc2 整段
        inputs = _build_aclnn_v2_shard_inputs(TRUNC_DOC_LENS, SHARD_START_TRUNC, shard_len, seed=7)
        row_prefix = inputs["row_prefix"]
        # mid-doc 前缀延伸可见性：shard 行 0 处于 doc-local 50，已见 (50+1)//4 = 12 个池
        # （可见性锚定文档头，而非 shard 头从头计数的 1）。
        assert int(row_prefix[0]) == (SHARD_START_TRUNC + 1) // INDEX_KPOOL, (
            f"mid-doc row 0 must see pools counted from the document head, got prefix "
            f"{int(row_prefix[0])}"
        )
        # dense 前缀在 shard 内部截断于 dense_row_cap：首个 op 行是 shard 行 cap - d0
        # （若不截断将是 67，shard 已过 cap 则是 0——两种错误锚定都被此断言排除）。
        dense_row_cap = INDEX_KPOOL * (SELECT_K + 1) - 1
        assert _first_op_row(row_prefix, SELECT_K) == dense_row_cap - SHARD_START_TRUNC, (
            f"first op row must anchor at dense_row_cap - d0 = {dense_row_cap - SHARD_START_TRUNC}"
        )
        sel_whole = _run_aclnn_v2(inputs, chunk=0)
        _assert_sel_prefix_contract(sel_whole, row_prefix, label="chunk=0 mid-doc truncated dense")
        for chunk in (1024, 512):
            sel_chunked = _run_aclnn_v2(inputs, chunk=chunk)
            assert torch.equal(sel_whole, sel_chunked), (
                f"mid-doc truncated dense prefix: chunk=0 vs chunk={chunk} outputs differ: "
                f"diff elems={(sel_whole != sel_chunked).sum().item()} / {sel_whole.numel()}; "
                f"first diff row="
                f"{(sel_whole != sel_chunked).any(dim=-1).nonzero().flatten()[:5].tolist()}"
            )

    def test_mid_doc_shard_fully_dense_block_empty_range_bitwise_equal_chunked(self):
        """mid-doc 首块整块全 dense：shard 从 doc-local 40 进入 60 token 短文档，块
        [40, 60) 全部 ``prefix <= select_k`` → ``dense_end_doc == row_hi`` →
        ``c_start == row_hi``，chunk=0 走 ``max(row_hi - c_start, 1)`` 空 range 分支
        （该块零 op item，输出纯 dense closed-form）；同 shard 后续整段可见文档覆盖
        dense+op 混合与 512 的 item 缝。chunk=0 与 chunked 逐位相等。"""
        shard_len = (60 - SHARD_START_FULLY_DENSE) + 500 + 800  # 首块 20 行 + doc1/doc2 整段
        inputs = _build_aclnn_v2_shard_inputs(FULLY_DENSE_MID_DOC_LENS, SHARD_START_FULLY_DENSE, shard_len, seed=11)
        row_prefix = inputs["row_prefix"]
        assert int(row_prefix[0]) == (SHARD_START_FULLY_DENSE + 1) // INDEX_KPOOL
        # 首块整块全 dense：首个 op 行落在后续文档 doc-local dense_row_cap 处，即 shard
        # 行 (60 - 40) + 67 = 87——首块贡献 0 个 op item（空 range 分支实际走到）。
        dense_row_cap = INDEX_KPOOL * (SELECT_K + 1) - 1
        assert _first_op_row(row_prefix, SELECT_K) == (60 - SHARD_START_FULLY_DENSE) + dense_row_cap, (
            "fully-dense mid-doc block must contribute no op rows"
        )
        sel_whole = _run_aclnn_v2(inputs, chunk=0)
        _assert_sel_prefix_contract(sel_whole, row_prefix, label="chunk=0 mid-doc fully-dense block")
        for chunk in (1024, 512):
            sel_chunked = _run_aclnn_v2(inputs, chunk=chunk)
            assert torch.equal(sel_whole, sel_chunked), (
                f"mid-doc fully-dense block: chunk=0 vs chunk={chunk} outputs differ: "
                f"diff elems={(sel_whole != sel_chunked).sum().item()} / {sel_whole.numel()}; "
                f"first diff row="
                f"{(sel_whole != sel_chunked).any(dim=-1).nonzero().flatten()[:5].tolist()}"
            )

    @pytest.mark.parametrize("chunk", [1024, 256], ids=["chunked-1024", "chunked-256"])
    def test_shard_both_ends_mid_doc_mixed_docs_bitwise_equal_chunked(self, chunk: int):
        """shard 两端都落在文档中间 + pack 内多文档混合：首块 doc0 从 doc-local 700 进入
        （> dense_row_cap → 零 dense 行、``c_start == row_lo``，shard 行 0 已见 175 池）；
        doc1（2048 token）整段可见；doc2（3 token）全 dense 微文档；尾块 doc3 只覆盖前
        500 行（后 800 行在其它 rank，op item 的 ``floor_pools`` 只算 shard 可见的
        125 池而非全文档 325 池）。chunk=0 与 chunked（256 在首/尾被切块内都制造
        item 缝）输出逐位相等。"""
        shard_len = (1024 - SHARD_START_BOTH_ENDS) + 2048 + 3 + 500  # doc0 余 324 + doc1 + doc2 + doc3 前 500
        inputs = _build_aclnn_v2_shard_inputs(BOTH_ENDS_DOC_LENS, SHARD_START_BOTH_ENDS, shard_len, seed=13)
        row_prefix = inputs["row_prefix"]
        assert int(row_prefix[0]) == (SHARD_START_BOTH_ENDS + 1) // INDEX_KPOOL
        # 首块全部行已过 dense cap：首个 op 行就是 shard 行 0（c_start == row_lo 锚定，
        # 该块没有任何 dense fill 行）。
        assert _first_op_row(row_prefix, SELECT_K) == 0, (
            "front mid-doc block beyond dense_row_cap must start op rows at shard row 0"
        )
        # 尾部切文档：最后一个 shard 行只看到 doc3 前 500 行的 125 个池（文档共 325 池，
        # 余行在其它 rank 上，永不可选）。
        assert int(row_prefix[-1]) == 500 // INDEX_KPOOL, (
            f"trailing mid-doc shard row must see only the shard-visible pools, got "
            f"prefix {int(row_prefix[-1])}"
        )
        sel_whole = _run_aclnn_v2(inputs, chunk=0)
        _assert_sel_prefix_contract(sel_whole, row_prefix, label="chunk=0 both-ends mid-doc")
        sel_chunked = _run_aclnn_v2(inputs, chunk=chunk)
        assert torch.equal(sel_whole, sel_chunked), (
            f"both-ends mid-doc shard: chunk=0 vs chunk={chunk} outputs differ: "
            f"diff elems={(sel_whole != sel_chunked).sum().item()} / {sel_whole.numel()}; "
            f"first diff row="
            f"{(sel_whole != sel_chunked).any(dim=-1).nonzero().flatten()[:5].tolist()}"
        )

    @pytest.mark.parametrize(
        ("select_k", "doc_lens", "shard_start"),
        [(64, [2000, 400], 100), (128, [1600, 300], 300)],
        ids=["select-k-64", "select-k-128"],
    )
    def test_mid_doc_shard_large_select_k_bitwise_equal_chunked(self, select_k: int, doc_lens: list[int], shard_start: int):
        """更大 select_k（64 / 128）下的 mid-doc shard：dense 前缀在 shard 内部截断于
        ``dense_row_cap = 4*(select_k+1)-1``（= 259 / 515），首块 op 区被 chunk={1024,
        512} 切出多道 item 缝。chunk=0 与 chunked 全槽位逐位相等——回应 bitwise 等价
        验证仅 select_k=16 的规模缺口。"""
        shard_len = sum(doc_lens) - shard_start
        inputs = _build_aclnn_v2_shard_inputs(doc_lens, shard_start, shard_len, seed=17 if select_k == 64 else 19)
        row_prefix = inputs["row_prefix"]
        dense_row_cap = INDEX_KPOOL * (select_k + 1) - 1
        assert int(row_prefix[0]) == (shard_start + 1) // INDEX_KPOOL
        assert _first_op_row(row_prefix, select_k) == dense_row_cap - shard_start, (
            f"select_k={select_k}: first op row must anchor at dense_row_cap - d0 "
            f"= {dense_row_cap - shard_start}"
        )
        sel_whole = _run_aclnn_v2(inputs, chunk=0, select_k=select_k)
        _assert_sel_prefix_contract(sel_whole, row_prefix, label=f"chunk=0 select_k={select_k}", select_k=select_k)
        for chunk in (1024, 512):
            sel_chunked = _run_aclnn_v2(inputs, chunk=chunk, select_k=select_k)
            assert torch.equal(sel_whole, sel_chunked), (
                f"select_k={select_k}: chunk=0 vs chunk={chunk} outputs differ: "
                f"diff elems={(sel_whole != sel_chunked).sum().item()} / {sel_whole.numel()}; "
                f"first diff row="
                f"{(sel_whole != sel_chunked).any(dim=-1).nonzero().flatten()[:5].tolist()}"
            )
