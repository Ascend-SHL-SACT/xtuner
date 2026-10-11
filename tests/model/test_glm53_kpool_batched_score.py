"""GLM-5.3 kpool indexer 批量打分路径（``_BATCHED_SCORE``）与派生张量缓存回归测试。

``kpool_npu`` 新增批量打分路径：整 call 预填 ``finfo.min`` 的 ``[S, W]`` fp32 分数缓冲 + 整 call
可见性掩码（``col < row_prefix``）+ 每文档段一次 ``lightning_indexer_score_into``（跨步行 strided
写入）+ 整 shard 一次批量 topk，替代 legacy 的逐段 pad→score→topk→scatter 链；两条路径共享同一
``sel < row_prefix`` 有效性判定与纯算术 token 展开。相关派生张量缓存一并回归：
``build_pool_index``（按 (ctx, kpool, seq_len, device) 共享只读 int64 结果，``build_pools`` 返回
独立 int32 副本）、``_cached_param_derived``（按参数 id+版本缓存 fp32/rearrange 派生，弱引用校验
活对象）、``_cached_segment_ids``（按 lens+device 缓存 ``repeat_interleave`` 段 id）。

TestKpoolBatchedScoreEquivalence（NPU）
    test_mixed_packed_documents_bitwise_equal: 混合文档 [200,3,17,64,65,1,129,300,7,77]，门开/门关
        输出逐位一致，且 spy 确证批量路径恰在门开时派发。
    test_single_document_bitwise_equal: 单文档 [512] 逐位一致。
    test_many_small_documents_bitwise_equal: 大量小文档 [5,9,1,2,33,16,8,4,64,12,3,27] 逐位一致
        （段多、前缀参差，锁定批量 topk 尾槽经 ``sel < row_prefix`` 掩回 -1 的语义）。
    test_trinary_tie_stress_bitwise_equal: q/k 取三值 randint(-1,0,1) bf16 制造大量精确 fp32 平分，
        两条路径仍逐位一致。
TestKpoolBatchedScoreFallback（NPU）
    test_select_k_above_max_pools_gate_stays_off_bitwise_equal: 全部文档短于 select_k*kpool 时批量
        门不派发（spy 未触发），门开输出与门关（legacy）逐位一致。
    test_gate_off_minus_one_padding_semantics: 门关输出形状/dtype 契约成立，且每行
        ``min(prefix, select_k)`` 个可见池之外的 index_topk 槽全为 -1、界内槽全非负。
TestKpoolDegenerateSlot（NPU）
    test_degenerate_slot_set_equivalence_mixed_documents_production_topk: 生产 2048/512 混合短文档下退化
        分支（门开）与 legacy（门关）集合等价：每行有效 slot 数相等、排序后有效 token id 集合相等且等于
        闭式解 [doc_start, doc_start + prefix*kpool)。刻意不做逐位比较——有效 id 的行内顺序是设计内差异
        （退化分支列序 vs legacy 分数序），下游 gather 消费者以 fp32 累加、对顺序不敏感；同时锁定退化分支
        零打分 spy、dtype/shape 契约与"两条路径顺序确有差异"。
    test_degenerate_slot_set_equivalence_tie_stress: q/k 取三值 randint(-1,0,1) 制造大量精确 fp32 平分、
        分数序任意时集合级契约仍成立。
    test_degenerate_slot_layout_edge_cases: 单 token 文档（prefix=0 全无效）、len<kpool、len%kpool!=0、
        单文档等退化布局逐一集合等价。
    test_degenerate_boundary_max_pools_select_k_minus_one: 2044-token 文档使 max_pools == select_k - 1，
        恰走退化分支：spy 全零、dtype/shape 与集合等价成立。
    test_boundary_max_pools_equal_select_k_dispatches_batched: 2045-token 文档使 max_pools == select_k，
        恰好回落批量打分分支：spy 派发一次、与 legacy 集合等价（生产规模下两条路径输入宽度不同，平分列的
        topk 顺序是实现定义的，刻意不锁逐位；小规模逐位等价由 TestKpoolBatchedScoreEquivalence 覆盖）。
    test_gate_off_keeps_score_order_bitwise_golden: 池分数随文档内列 id 严格单调递增的构造输入下，门关
        输出逐位等于"分数序（降列序）"闭式金标准——门关路径若被退化分支污染即刻失配；门开输出逐位等于
        "列序"金标准，锁定退化分支语义与尾槽。
    test_degenerate_skips_all_scoring: 退化条件下 _score_and_topk_batched、_score_pools 与
        lightning_indexer_score 均 0 次调用；门关对照下 legacy 打分入口确有派发。
TestKpoolBuildPoolIndexCache（CPU 或 NPU）
    test_same_context_and_args_return_identical_object: 同一 SequenceContext + 同参重复调用返回同一
        对象（``is``），且内容与逐 token 参考一致。
    test_equal_distinct_context_gets_equal_distinct_tensor: 等长 cu_seq_lens 的第二个 context 得到
        等值但不同对象的张量（不跨 context 串缓存）。
    test_build_pools_copy_is_mutable_without_poisoning_cache: ``build_pools`` 返回的 int32
        ``pool_index`` 副本可被调用方原地改写，且不污染缓存——重复调用与内部 int64 缓存均不受影响。
TestLightningIndexerScoreInto（NPU）
    test_strided_into_matches_contiguous_reference_and_keeps_prefill: 行 stride 大于发射宽度的
        ``out``/``valid`` 下，``[0, P)`` 列与连续参考 ``lightning_indexer_score`` 逐位一致、pad 列
        写 ``finfo.min``、发射宽度之外的列保留 7.0 预填。
    test_early_return_leaves_prefill_untouched: rows==0 或 num_pools<=0 早退，不触碰预填。
"""

from typing import Any

import pytest
import torch
from torch import Tensor

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.sparse_mla import kpool, kpool_npu
from xtuner.v1.utils.device import get_device


try:
    from xtuner.v1.ops.sparse_mla.kpool_npu import (  # type: ignore[no-redef]
        _MAX_BLOCK_P,
        _pick_tiles,
        lightning_indexer_score,
        lightning_indexer_score_into,
    )

    _HAS_FUSED_INDEXER = lightning_indexer_score_into is not None
except ImportError:  # triton-ascend 缺失（纯 CPU CI）: 融合 indexer 不可用
    _MAX_BLOCK_P = 256
    _pick_tiles = None  # type: ignore[assignment]
    lightning_indexer_score = None  # type: ignore[assignment]
    lightning_indexer_score_into = None  # type: ignore[assignment]
    _HAS_FUSED_INDEXER = False


DEVICE = get_device()  # 本盒为 "npu"；纯 CPU 环境退化为 "cpu"
_requires_accelerator = pytest.mark.skipif(DEVICE == "cpu", reason="requires an NPU/CUDA accelerator")
_requires_fused_indexer = pytest.mark.skipif(
    not _HAS_FUSED_INDEXER, reason="requires the fused lightning indexer kernel (triton-ascend)"
)

if DEVICE == "npu":
    torch.npu.set_device("npu:0")


def _make_ctx(doc_lens: list[int], device: str | torch.device = DEVICE) -> SequenceContext:
    """构造单 rank（无 SP mesh）的 packed ``SequenceContext``。

    Args:
        doc_lens (list[int]): 各文档 token 数（ packing 顺序即文档顺序）。
        device (str | torch.device): 上下文设备标签。

    Returns:
        SequenceContext: ``cu_seq_lens_q == cu_seq_lens_k``、``shard_start == 0`` 的上下文。
    """
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
        device=device,
    )


def _make_inputs(
    doc_lens: list[int],
    seed: int,
    *,
    tie_stress: bool = False,
    num_heads: int = 8,
    head_dim: int = 128,
    index_kpool: int = 4,
    device: str | torch.device = DEVICE,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """按给定布局一次性生成 ``npu_kpool_topk_indices`` 的全部输入（两条路径共用同一批张量）。

    Args:
        doc_lens (list[int]): 各文档 token 数。
        seed (int): 随机种子。
        tie_stress (bool): True 时 q/k 取三值 randint(-1,0,1)，制造大量精确 fp32 平分。
        num_heads (int): indexer 头数 NH。
        head_dim (int): indexer 头维 D（也是 index_head_dim）。
        index_kpool (int): 池大小。
        device (str | torch.device): 张量设备。

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: ``(q, k, gate_scores, weights, kpool_ape)``，形状依次为
            ``[1, S, NH, D]`` / ``[1, S, D]`` / ``[1, S, D]`` / ``[1, S, NH]`` / ``[index_kpool, D]``（与
            ``npu_kpool_topk_indices`` 的入参契约一致，``k``/``gate_scores`` 必须带批量维——单 token 文档的
            ``[S, D]`` 输入会被 ``squeeze(0)`` 误降维）。
    """
    torch.manual_seed(seed)
    total = sum(doc_lens)
    if tie_stress:
        q = torch.randint(-1, 2, (1, total, num_heads, head_dim), device=device).to(torch.bfloat16)
        k = torch.randint(-1, 2, (1, total, head_dim), device=device).to(torch.bfloat16)
    else:
        q = torch.randn(1, total, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(1, total, head_dim, dtype=torch.bfloat16, device=device)
    gate = torch.randn(1, total, head_dim, dtype=torch.bfloat16, device=device)
    weights = torch.randn(1, total, num_heads, dtype=torch.bfloat16, device=device)
    ape = torch.randn(index_kpool, head_dim, dtype=torch.bfloat16, device=device)
    return q, k, gate, weights, ape


def _run_topk(
    monkeypatch: pytest.MonkeyPatch,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ctx: SequenceContext,
    *,
    batched: bool,
    index_topk: int = 64,
    index_kpool: int = 4,
) -> Tensor:
    """以指定门状态跑一次 ``npu_kpool_topk_indices``（批量路径是否派发由 ``batched_spy`` 观察）。

    Args:
        monkeypatch (pytest.MonkeyPatch): 门状态的补丁载体（测试结束自动还原）。
        inputs (tuple[Tensor, Tensor, Tensor, Tensor, Tensor]): ``_make_inputs`` 的返回。
        ctx (SequenceContext): packed 上下文。
        batched (bool): ``kpool_npu._BATCHED_SCORE`` 的注入值。
        index_topk (int): 语义 token 预算。
        index_kpool (int): 池大小。

    Returns:
        Tensor: ``[S, 1, kpool_output_width(...)]`` int32 输出。
    """
    monkeypatch.setattr(kpool_npu, "_BATCHED_SCORE", batched)
    q, k, gate, weights, ape = inputs
    return kpool_npu.npu_kpool_topk_indices(
        q,
        k,
        gate,
        weights,
        ape,
        ctx,
        index_head_dim=q.shape[-1],
        index_topk=index_topk,
        index_kpool=index_kpool,
    )


@pytest.fixture
def batched_spy(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """包装 ``kpool_npu._score_and_topk_batched`` 的派发 spy（测试结束自动还原）。

    Returns:
        list[bool]: 批量路径每派发一次追加一个 True 的观测列表。
    """
    engaged: list[bool] = []
    real = kpool_npu._score_and_topk_batched

    def _spy(*args: Any, **kwargs: Any) -> Tensor:
        engaged.append(True)
        return real(*args, **kwargs)

    monkeypatch.setattr(kpool_npu, "_score_and_topk_batched", _spy)
    return engaged


@pytest.fixture
def score_spy(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """包装 legacy 打分入口（``_score_pools`` 与融合核 ``lightning_indexer_score``）的派发 spy。

    Returns:
        dict[str, int]: 两个入口各自被调用次数的计数表。
    """
    counts = {"_score_pools": 0, "lightning_indexer_score": 0}
    real_pools = kpool_npu._score_pools
    real_score = kpool_npu.lightning_indexer_score
    assert real_score is not None, "score_spy requires the fused lightning indexer kernel"

    def _spy_pools(*args: Any, **kwargs: Any) -> Tensor:
        counts["_score_pools"] += 1
        return real_pools(*args, **kwargs)

    def _spy_score(*args: Any, **kwargs: Any) -> Tensor:
        counts["lightning_indexer_score"] += 1
        return real_score(*args, **kwargs)

    monkeypatch.setattr(kpool_npu, "_score_pools", _spy_pools)
    monkeypatch.setattr(kpool_npu, "lightning_indexer_score", _spy_score)
    return counts


def _row_prefixes(doc_lens: list[int], index_kpool: int) -> list[int]:
    """逐行的因果可见池前缀 ``(local_pos + 1) // index_kpool``（展平后的 packed 行序）。

    Args:
        doc_lens (list[int]): 各文档 token 数。
        index_kpool (int): 池大小。

    Returns:
        list[int]: 长度 ``sum(doc_lens)`` 的每行可见池数。
    """
    prefixes: list[int] = []
    for ln in doc_lens:
        for pos in range(ln):
            prefixes.append((pos + 1) // index_kpool)
    return prefixes


def _row_tail_counts(doc_lens: list[int], index_kpool: int) -> list[int]:
    """逐行的可见尾 token 数 ``(local_pos + 1) % index_kpool``（展平后的 packed 行序）。

    Args:
        doc_lens (list[int]): 各文档 token 数。
        index_kpool (int): 池大小。

    Returns:
        list[int]: 长度 ``sum(doc_lens)`` 的每行尾 token 数。
    """
    tails: list[int] = []
    for ln in doc_lens:
        for pos in range(ln):
            tails.append((pos + 1) % index_kpool)
    return tails


def _monotone_inputs(
    doc_lens: list[int],
    *,
    num_heads: int = 8,
    head_dim: int = 128,
    index_kpool: int = 4,
    device: str | torch.device = DEVICE,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """构造"池分数随文档内列 id 严格单调递增"的输入（两条路径的选中列顺序因此可预测）。

    手法：k 只在 dim 0 取 ``(文档内列 id + 1)``、其余维全 0；q 只在 dim 0 取 1；gate 与 ape 全 0 使池内
    softmax 均匀、池键恰等于池内公共 token 值；weights 全 1。于是 ``score(row, c)`` 严格随列 id ``c``
    递增：legacy 的降序 topk 给出降列序，退化分支给出升列序，顺序差异可被闭式金标准逐位锁定。

    Args:
        doc_lens (list[int]): 各文档 token 数（须全部短于 ``select_k * index_kpool``，即退化负载）。
        num_heads (int): indexer 头数 NH。
        head_dim (int): indexer 头维 D（也是 index_head_dim）。
        index_kpool (int): 池大小。
        device (str | torch.device): 张量设备。

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: ``(q, k, gate_scores, weights, kpool_ape)``，形状依次为
            ``[1, S, NH, D]`` / ``[1, S, D]`` / ``[1, S, D]`` / ``[1, S, NH]`` / ``[index_kpool, D]``（与
            ``npu_kpool_topk_indices`` 的入参契约一致，``k``/``gate_scores`` 必须带批量维——单 token 文档的
            ``[S, D]`` 输入会被 ``squeeze(0)`` 误降维）。
    """
    total = sum(doc_lens)
    col_of_token = torch.zeros(total, dtype=torch.int64, device=device)
    start = 0
    for ln in doc_lens:
        col_of_token[start : start + ln] = torch.arange(ln, device=device) // index_kpool
        start += ln
    k = torch.zeros(1, total, head_dim, dtype=torch.bfloat16, device=device)
    k[:, :, 0] = (col_of_token + 1).to(torch.bfloat16)
    q = torch.zeros(1, total, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    q[..., 0] = 1.0
    gate = torch.zeros(1, total, head_dim, dtype=torch.bfloat16, device=device)
    weights = torch.ones(1, total, num_heads, dtype=torch.bfloat16, device=device)
    ape = torch.zeros(index_kpool, head_dim, dtype=torch.bfloat16, device=device)
    return q, k, gate, weights, ape


def _degenerate_golden(
    doc_lens: list[int],
    *,
    index_topk: int,
    index_kpool: int,
    column_order: str,
) -> Tensor:
    """退化负载下 ``npu_kpool_topk_indices`` 语义区 + 尾槽的闭式金标准（纯 CPU 算术，不依赖实现）。

    每行可见前缀 ``prefix = (local_pos + 1) // index_kpool`` 个真实池列全部有效：``column_order="column_asc"``
    对应退化分支的列序（第 j 个选中列 = j），``"score_desc"`` 对应 legacy 在严格递增分数下的降序 topk
    （第 j 个选中列 = prefix - 1 - j）。池列 c 覆盖 token ``doc_start + c * index_kpool + slot``；尾槽与
    ``_visible_tail_tokens`` 同式（``token_id - tail_count + 1 + i``，i < tail_count），其余槽 -1。

    Args:
        doc_lens (list[int]): 各文档 token 数（须全部短于 ``select_k * index_kpool``）。
        index_topk (int): 语义 token 预算。
        index_kpool (int): 池大小。
        column_order (str): ``"column_asc"``（退化分支列序）或 ``"score_desc"``（legacy 分数序）。

    Returns:
        Tensor: ``[S, kpool_output_width(index_topk, index_kpool, 512)]`` int32 CPU 金标准。
    """
    width = kpool.kpool_output_width(index_topk, index_kpool, 512)
    exp = torch.full((sum(doc_lens), width), -1, dtype=torch.int32)
    select_k = index_topk // index_kpool
    doc_start = 0
    for ln in doc_lens:
        for local in range(ln):
            row = doc_start + local
            prefix = min((local + 1) // index_kpool, select_k)
            for j in range(prefix):
                col = j if column_order == "column_asc" else prefix - 1 - j
                for slot in range(index_kpool):
                    exp[row, j * index_kpool + slot] = doc_start + col * index_kpool + slot
            tail_count = (local + 1) % index_kpool
            for i in range(tail_count):
                exp[row, index_topk + i] = row - tail_count + 1 + i
        doc_start += ln
    return exp


def _assert_prefix_complete_set_equivalence(
    actual: Tensor,
    expected: Tensor,
    doc_lens: list[int],
    *,
    index_topk: int,
    index_kpool: int,
) -> None:
    """断言两条路径输出满足"可见前缀内的池列全部有效"的集合级契约（刻意不做逐位比较）。

    契约来源：select_k 不小于任何行的可见前缀时（退化负载必然如此；max_pools == select_k 的批量分支
    边界亦然），两条路径每行选中的都是该行可见前缀内的全部真实池列，经算术展开后有效 token id 恰为连续
    区间 ``[doc_start, doc_start + prefix * index_kpool)``。因此断言：每行有效 slot 数相等且等于
    ``prefix * index_kpool``、排序后的有效 token id 等于该区间。有效 id 的行内顺序刻意不比较——列序与
    分数序是设计内差异（下游 gather 消费者以 fp32 累加、对顺序不敏感）。

    Args:
        actual (Tensor): 待测路径输出 ``[S, 1, width]`` int32。
        expected (Tensor): 参照路径输出 ``[S, 1, width]`` int32。
        doc_lens (list[int]): 各文档 token 数。
        index_topk (int): 语义 token 预算。
        index_kpool (int): 池大小。
    """
    assert actual.dtype == expected.dtype == torch.int32
    assert actual.shape == expected.shape
    sem_actual = actual[:, 0, :index_topk]
    sem_expected = expected[:, 0, :index_topk]
    counts_actual = (sem_actual >= 0).sum(-1).cpu()
    counts_expected = (sem_expected >= 0).sum(-1).cpu()
    assert torch.equal(counts_actual, counts_expected), "per-row valid slot counts must match between the two paths"

    prefix = torch.tensor(_row_prefixes(doc_lens, index_kpool), dtype=torch.int64)
    assert torch.equal(counts_actual.to(torch.int64), prefix * index_kpool), (
        "per-row valid slot count must equal prefix * index_kpool"
    )

    doc_starts: list[int] = []
    doc_start = 0
    for ln in doc_lens:
        doc_starts.extend([doc_start] * ln)
        doc_start += ln
    doc_start_t = torch.tensor(doc_starts, dtype=torch.int64).unsqueeze(-1)  # [S, 1]
    pos = torch.arange(index_topk).unsqueeze(0)  # [1, index_topk]
    sentinel = 1 << 30  # 高于任何合法 token id：-1 槽替换后排序沉底
    sorted_expected = torch.where(
        pos < counts_actual.to(torch.int64).unsqueeze(-1),
        doc_start_t + pos,
        torch.full((len(doc_starts), index_topk), sentinel, dtype=torch.int64),
    )
    for name, sem in (("actual", sem_actual), ("expected", sem_expected)):
        vals = torch.where(sem >= 0, sem, torch.full_like(sem, sentinel))
        sorted_vals = vals.sort(dim=-1).values.cpu().to(torch.int64)
        assert torch.equal(sorted_vals, sorted_expected), f"{name}: sorted valid token ids must equal the closed form"


class TestKpoolBatchedScoreEquivalence:
    @_requires_accelerator
    @_requires_fused_indexer
    @pytest.mark.gpu
    def test_mixed_packed_documents_bitwise_equal(
        self, monkeypatch: pytest.MonkeyPatch, batched_spy: list[bool]
    ) -> None:
        """混合长短文档布局：门开（批量）与门关（legacy）输出逐位一致，批量门恰派发一次。"""
        doc_lens = [200, 3, 17, 64, 65, 1, 129, 300, 7, 77]
        ctx = _make_ctx(doc_lens)
        inputs = _make_inputs(doc_lens, seed=1)
        legacy = _run_topk(monkeypatch, inputs, ctx, batched=False)
        assert batched_spy == [], "legacy run must not dispatch the batched scorer"
        batched = _run_topk(monkeypatch, inputs, ctx, batched=True)
        assert batched_spy == [True], batched_spy
        assert batched.dtype == torch.int32 and batched.shape == legacy.shape
        assert torch.equal(batched, legacy)

    @_requires_accelerator
    @_requires_fused_indexer
    @pytest.mark.gpu
    def test_single_document_bitwise_equal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """单文档 [512]：整段一次发射 vs 逐段发射，输出逐位一致。"""
        doc_lens = [512]
        ctx = _make_ctx(doc_lens)
        inputs = _make_inputs(doc_lens, seed=2)
        legacy = _run_topk(monkeypatch, inputs, ctx, batched=False)
        batched = _run_topk(monkeypatch, inputs, ctx, batched=True)
        assert torch.equal(batched, legacy)

    @_requires_accelerator
    @_requires_fused_indexer
    @pytest.mark.gpu
    def test_many_small_documents_bitwise_equal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """大量小文档：段数多、前缀参差，可见池少于 select_k 的行批量 topk 尾槽同样掩回 -1。"""
        doc_lens = [5, 9, 1, 2, 33, 16, 8, 4, 64, 12, 3, 27]
        ctx = _make_ctx(doc_lens)
        inputs = _make_inputs(doc_lens, seed=3)
        legacy = _run_topk(monkeypatch, inputs, ctx, batched=False)
        batched = _run_topk(monkeypatch, inputs, ctx, batched=True)
        assert torch.equal(batched, legacy)

    @_requires_accelerator
    @_requires_fused_indexer
    @pytest.mark.gpu
    def test_trinary_tie_stress_bitwise_equal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """三值 q/k 制造大量精确 fp32 平分：两条路径的 topk 结果仍逐位一致。"""
        doc_lens = [130, 70, 258, 64]
        ctx = _make_ctx(doc_lens)
        inputs = _make_inputs(doc_lens, seed=5, tie_stress=True)
        legacy = _run_topk(monkeypatch, inputs, ctx, batched=False)
        batched = _run_topk(monkeypatch, inputs, ctx, batched=True)
        assert torch.equal(batched, legacy)


class TestKpoolBatchedScoreFallback:
    @_requires_accelerator
    @_requires_fused_indexer
    @pytest.mark.gpu
    def test_select_k_above_max_pools_gate_stays_off_bitwise_equal(
        self, monkeypatch: pytest.MonkeyPatch, batched_spy: list[bool]
    ) -> None:
        """select_k > max_pools 时批量门不得派发：门开输出与门关（legacy）逐位一致且 spy 未触发。"""
        doc_lens = [1, 2, 3, 4]
        ctx = _make_ctx(doc_lens)
        inputs = _make_inputs(doc_lens, seed=4)
        # 构造自检：最大文档 4 token -> 每文档至多 1 个池，select_k=16 必然越过批量门。
        max_pools = max(-(-ln // 4) for ln in doc_lens)
        assert max_pools < 64 // 4
        gate_off = _run_topk(monkeypatch, inputs, ctx, batched=False)
        assert batched_spy == [], "legacy run must not dispatch the batched scorer"
        gate_on = _run_topk(monkeypatch, inputs, ctx, batched=True)
        assert batched_spy == [], batched_spy
        assert torch.equal(gate_on, gate_off)

    @_requires_accelerator
    @pytest.mark.gpu
    def test_gate_off_minus_one_padding_semantics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """门关路径 -1 填充语义：可见池之外的槽全 -1、界内槽全非负、尾槽为查询自身全局位置。"""
        doc_lens = [200, 3, 17]
        index_kpool, index_topk = 4, 64
        ctx = _make_ctx(doc_lens)
        inputs = _make_inputs(doc_lens, seed=8, index_kpool=index_kpool)
        out = _run_topk(monkeypatch, inputs, ctx, batched=False, index_topk=index_topk, index_kpool=index_kpool)
        assert out.dtype == torch.int32
        assert out.shape == (sum(doc_lens), 1, 512)
        visible = out[:, 0, :index_topk]
        for row, prefix in enumerate(_row_prefixes(doc_lens, index_kpool)):
            n_valid = min(prefix, index_topk // index_kpool)
            if n_valid > 0:
                assert (visible[row, : n_valid * index_kpool] >= 0).all(), f"row {row} valid slots must be >= 0"
            assert (visible[row, n_valid * index_kpool :] == -1).all(), f"row {row} pad slots must be -1"
        # 尾槽锚：尾数非零的行，其最后一个有效尾槽 == 查询自身全局位置（尾数按文档内 local_pos 计）。
        for row, tail_count in enumerate(_row_tail_counts(doc_lens, index_kpool)):
            if tail_count > 0:
                assert out[row, 0, index_topk + tail_count - 1].item() == row


class TestKpoolDegenerateSlot:
    @_requires_accelerator
    @_requires_fused_indexer
    @pytest.mark.gpu
    def test_degenerate_slot_set_equivalence_mixed_documents_production_topk(
        self, monkeypatch: pytest.MonkeyPatch, batched_spy: list[bool], score_spy: dict[str, int]
    ) -> None:
        """生产 2048/512 混合短文档：退化分支与 legacy 集合等价、打分全跳过、dtype/shape 一致。"""
        doc_lens = [200, 3, 17, 64, 65, 1, 129, 300, 7, 77, 33, 512]
        index_topk, index_kpool = 2048, 4
        select_k = index_topk // index_kpool
        assert max(-(-ln // index_kpool) for ln in doc_lens) < select_k, "layout must stay in the degenerate workload"
        ctx = _make_ctx(doc_lens)
        inputs = _make_inputs(doc_lens, seed=21)
        legacy = _run_topk(monkeypatch, inputs, ctx, batched=False, index_topk=index_topk, index_kpool=index_kpool)
        legacy_counts = dict(score_spy)
        assert batched_spy == [], "legacy run must not dispatch the batched scorer"
        assert legacy_counts["_score_pools"] >= 1, "legacy run must dispatch per-segment scoring"
        degenerate = _run_topk(monkeypatch, inputs, ctx, batched=True, index_topk=index_topk, index_kpool=index_kpool)
        assert batched_spy == [], batched_spy
        assert score_spy == legacy_counts, "degenerate branch must not score at all"
        assert degenerate.dtype == legacy.dtype == torch.int32
        width = kpool.kpool_output_width(index_topk, index_kpool, 512)
        assert degenerate.shape == legacy.shape == (sum(doc_lens), 1, width)
        # 有效 token id 的行内顺序是设计内差异（退化分支列序 vs legacy 分数序）；随机分数 + 数百行下
        # 两条路径逐行同序的概率可忽略，锁定"确有顺序差异"以防集合契约退化为平凡恒等。
        assert not torch.equal(degenerate[:, 0, :index_topk], legacy[:, 0, :index_topk])
        _assert_prefix_complete_set_equivalence(
            degenerate, legacy, doc_lens, index_topk=index_topk, index_kpool=index_kpool
        )

    @_requires_accelerator
    @_requires_fused_indexer
    @pytest.mark.gpu
    def test_degenerate_slot_set_equivalence_tie_stress(
        self, monkeypatch: pytest.MonkeyPatch, batched_spy: list[bool], score_spy: dict[str, int]
    ) -> None:
        """三值 randint(-1,0,1) q/k 制造大量精确平分：分数序任意时集合级契约仍必须成立。"""
        doc_lens = [40, 33, 17, 9, 5, 1, 124, 3]
        index_topk = 128
        select_k = index_topk // 4
        assert max(-(-ln // 4) for ln in doc_lens) < select_k, "layout must stay in the degenerate workload"
        ctx = _make_ctx(doc_lens)
        inputs = _make_inputs(doc_lens, seed=22, tie_stress=True)
        legacy = _run_topk(monkeypatch, inputs, ctx, batched=False, index_topk=index_topk)
        degenerate = _run_topk(monkeypatch, inputs, ctx, batched=True, index_topk=index_topk)
        assert batched_spy == [], batched_spy
        assert score_spy["_score_pools"] >= 1, "legacy run must dispatch per-segment scoring"
        _assert_prefix_complete_set_equivalence(degenerate, legacy, doc_lens, index_topk=index_topk, index_kpool=4)

    @_requires_accelerator
    @_requires_fused_indexer
    @pytest.mark.gpu
    @pytest.mark.parametrize("doc_lens", [[1], [4], [7], [16], [1, 9, 1], [3, 1, 2]])
    def test_degenerate_slot_layout_edge_cases(self, monkeypatch: pytest.MonkeyPatch, doc_lens: list[int]) -> None:
        """退化布局边缘：单 token 文档（prefix=0 全无效）、len<kpool、len%kpool!=0、单/多小文档。"""
        index_topk = 128
        assert max(-(-ln // 4) for ln in doc_lens) < index_topk // 4, "layout must stay in the degenerate workload"
        ctx = _make_ctx(doc_lens)
        inputs = _make_inputs(doc_lens, seed=23)
        legacy = _run_topk(monkeypatch, inputs, ctx, batched=False, index_topk=index_topk)
        degenerate = _run_topk(monkeypatch, inputs, ctx, batched=True, index_topk=index_topk)
        if max(doc_lens) == 1:
            assert (degenerate[:, 0, :index_topk] == -1).all(), "prefix=0 rows must have no valid slot at all"
        _assert_prefix_complete_set_equivalence(degenerate, legacy, doc_lens, index_topk=index_topk, index_kpool=4)

    @_requires_accelerator
    @_requires_fused_indexer
    @pytest.mark.gpu
    def test_degenerate_boundary_max_pools_select_k_minus_one(
        self, monkeypatch: pytest.MonkeyPatch, batched_spy: list[bool], score_spy: dict[str, int]
    ) -> None:
        """2044-token 文档使 max_pools == select_k - 1：恰走退化分支，spy 全零且与 legacy 集合等价。"""
        doc_lens = [2044, 3]
        index_topk, index_kpool = 2048, 4
        select_k = index_topk // index_kpool
        assert max(-(-ln // index_kpool) for ln in doc_lens) == select_k - 1, "layout must sit on the degenerate side"
        ctx = _make_ctx(doc_lens)
        inputs = _make_inputs(doc_lens, seed=24, index_kpool=index_kpool)
        legacy = _run_topk(monkeypatch, inputs, ctx, batched=False, index_topk=index_topk, index_kpool=index_kpool)
        assert batched_spy == [], "legacy run must not dispatch the batched scorer"
        legacy_counts = dict(score_spy)
        assert legacy_counts["_score_pools"] >= 1, "legacy run must dispatch per-segment scoring"
        degenerate = _run_topk(monkeypatch, inputs, ctx, batched=True, index_topk=index_topk, index_kpool=index_kpool)
        assert batched_spy == [], batched_spy
        assert score_spy == legacy_counts, "degenerate branch must not score at all"
        width = kpool.kpool_output_width(index_topk, index_kpool, 512)
        assert degenerate.shape == legacy.shape == (sum(doc_lens), 1, width)
        _assert_prefix_complete_set_equivalence(
            degenerate, legacy, doc_lens, index_topk=index_topk, index_kpool=index_kpool
        )

    @_requires_accelerator
    @_requires_fused_indexer
    @pytest.mark.gpu
    def test_boundary_max_pools_equal_select_k_dispatches_batched(
        self, monkeypatch: pytest.MonkeyPatch, batched_spy: list[bool]
    ) -> None:
        """2045-token 文档使 max_pools == select_k：恰好回落批量打分分支（派发一次、与 legacy 集合等价）。"""
        doc_lens = [2045, 3]
        index_topk, index_kpool = 2048, 4
        select_k = index_topk // index_kpool
        assert max(-(-ln // index_kpool) for ln in doc_lens) == select_k, "layout must sit exactly on the batched side"
        ctx = _make_ctx(doc_lens)
        inputs = _make_inputs(doc_lens, seed=25, index_kpool=index_kpool)
        legacy = _run_topk(monkeypatch, inputs, ctx, batched=False, index_topk=index_topk, index_kpool=index_kpool)
        assert batched_spy == [], "legacy run must not dispatch the batched scorer"
        batched = _run_topk(monkeypatch, inputs, ctx, batched=True, index_topk=index_topk, index_kpool=index_kpool)
        assert batched_spy == [True], batched_spy
        # 生产规模下两条路径的 topk 输入宽度不同（[S, P] vs [S, P + slack]），平分列的 topk 顺序是实现
        # 定义的，故此处锁定集合级契约；小规模逐位等价由 TestKpoolBatchedScoreEquivalence 覆盖。
        _assert_prefix_complete_set_equivalence(
            batched, legacy, doc_lens, index_topk=index_topk, index_kpool=index_kpool
        )

    @_requires_accelerator
    @_requires_fused_indexer
    @pytest.mark.gpu
    def test_gate_off_keeps_score_order_bitwise_golden(
        self, monkeypatch: pytest.MonkeyPatch, batched_spy: list[bool], score_spy: dict[str, int]
    ) -> None:
        """严格递增分数下：门关逐位等于分数序（降列序）金标准、门开逐位等于列序金标准。

        门关路径若被退化分支污染（顺序变成列序），降列序金标准即刻失配——这是"门关不参与退化分支"的
        逐位证明；门开金标准同时锁定退化分支的列序语义与尾槽。
        """
        doc_lens = [9, 1, 17, 4]
        index_topk, index_kpool = 128, 4
        select_k = index_topk // index_kpool
        assert max(-(-ln // index_kpool) for ln in doc_lens) < select_k, "layout must stay in the degenerate workload"
        ctx = _make_ctx(doc_lens)
        inputs = _monotone_inputs(doc_lens, index_kpool=index_kpool)
        legacy = _run_topk(monkeypatch, inputs, ctx, batched=False, index_topk=index_topk, index_kpool=index_kpool)
        legacy_counts = dict(score_spy)
        assert legacy_counts["_score_pools"] >= 1, "gate-off run must dispatch legacy scoring"
        degenerate = _run_topk(monkeypatch, inputs, ctx, batched=True, index_topk=index_topk, index_kpool=index_kpool)
        assert batched_spy == [] and score_spy == legacy_counts, "degenerate branch must not score at all"
        golden_desc = _degenerate_golden(
            doc_lens, index_topk=index_topk, index_kpool=index_kpool, column_order="score_desc"
        )
        golden_asc = _degenerate_golden(
            doc_lens, index_topk=index_topk, index_kpool=index_kpool, column_order="column_asc"
        )
        assert not torch.equal(golden_asc, golden_desc), "the two orders must genuinely differ in this layout"
        assert torch.equal(legacy[:, 0].cpu(), golden_desc), "gate-off output must keep the legacy score order"
        assert torch.equal(degenerate[:, 0].cpu(), golden_asc), "gate-on output must be the degenerate column order"

    @_requires_accelerator
    @_requires_fused_indexer
    @pytest.mark.gpu
    def test_degenerate_skips_all_scoring(
        self, monkeypatch: pytest.MonkeyPatch, batched_spy: list[bool], score_spy: dict[str, int]
    ) -> None:
        """退化条件下批量打分与 legacy 打分入口（_score_pools / lightning_indexer_score）均 0 次调用。"""
        doc_lens = [40, 33, 17, 9, 5, 1, 124, 3]
        index_topk = 128
        ctx = _make_ctx(doc_lens)
        inputs = _make_inputs(doc_lens, seed=26)
        degenerate = _run_topk(monkeypatch, inputs, ctx, batched=True, index_topk=index_topk)
        assert batched_spy == [], "degenerate workload must not dispatch the batched scorer"
        assert score_spy == {"_score_pools": 0, "lightning_indexer_score": 0}, score_spy
        legacy = _run_topk(monkeypatch, inputs, ctx, batched=False, index_topk=index_topk)
        assert score_spy["_score_pools"] >= 1 and score_spy["lightning_indexer_score"] >= 1, (
            "gate-off contrast: the legacy scoring entry points must actually dispatch"
        )
        assert degenerate.shape == legacy.shape


class TestKpoolBuildPoolIndexCache:
    def test_same_context_and_args_return_identical_object(self) -> None:
        """同一 SequenceContext + 同参重复调用返回同一对象（``is``），内容与逐 token 参考一致。"""
        doc_lens = [7, 3, 16]
        index_kpool = 4
        seq_len = sum(doc_lens)
        ctx = _make_ctx(doc_lens)
        first = kpool.build_pool_index(ctx, seq_len, index_kpool, DEVICE)
        second = kpool.build_pool_index(ctx, seq_len, index_kpool, DEVICE)
        assert second is first
        # 语义锚：文档 0 长度 7 -> 池 0 = token 0..3、池 1 = token 4..6 + 一个 -1 尾槽；池 2 起为文档 1；
        # 文档 2 长度 16 -> 4 个完整池（共 2+1+4 = 7 个池）。
        assert first.shape == (7, index_kpool)
        assert first[0].tolist() == [0, 1, 2, 3]
        assert first[1].tolist() == [4, 5, 6, -1]
        assert first[2].tolist() == [7, 8, 9, -1]
        assert first[3].tolist() == [10, 11, 12, 13]
        assert first[6].tolist() == [22, 23, 24, 25]

    def test_equal_distinct_context_gets_equal_distinct_tensor(self) -> None:
        """等长 cu_seq_lens 的第二个 context 得到等值但不同对象的张量（不跨 context 串缓存）。"""
        doc_lens = [7, 3, 16]
        seq_len = sum(doc_lens)
        ctx_a = _make_ctx(doc_lens)
        ctx_b = _make_ctx(doc_lens)
        first = kpool.build_pool_index(ctx_a, seq_len, 4, DEVICE)
        second = kpool.build_pool_index(ctx_b, seq_len, 4, DEVICE)
        assert first is not second
        assert torch.equal(first, second)

    def test_build_pools_copy_is_mutable_without_poisoning_cache(self) -> None:
        """``build_pools`` 返回的 int32 副本可原地改写：不污染缓存，重复调用与内部 int64 均不变。"""
        doc_lens = [9, 5]
        index_kpool = 4
        seq_len = sum(doc_lens)
        ctx = _make_ctx(doc_lens)
        torch.manual_seed(7)
        k = torch.randn(seq_len, 32, dtype=torch.bfloat16, device=DEVICE)
        gate = torch.randn(seq_len, 32, dtype=torch.bfloat16, device=DEVICE)
        ape = torch.randn(index_kpool, 32, dtype=torch.bfloat16, device=DEVICE)
        pool_key1, pool_index1, _ = kpool.build_pools(k, gate, ape, ctx, index_kpool=index_kpool)
        assert pool_index1.dtype == torch.int32
        cached_int64 = kpool.build_pool_index(ctx, seq_len, index_kpool, DEVICE)
        snapshot = cached_int64.clone()

        pool_index1.fill_(-123)  # 调用方原地改写返回副本

        pool_key2, pool_index2, _ = kpool.build_pools(k, gate, ape, ctx, index_kpool=index_kpool)
        assert torch.equal(pool_index2, snapshot.to(torch.int32))
        assert torch.equal(pool_key2, pool_key1)
        assert torch.equal(kpool.build_pool_index(ctx, seq_len, index_kpool, DEVICE), snapshot)


@_requires_accelerator
@_requires_fused_indexer
@pytest.mark.gpu
class TestLightningIndexerScoreInto:
    def test_strided_into_matches_contiguous_reference_and_keeps_prefill(self) -> None:
        """跨步行 strided 缓冲：``[0, P)`` 列与连续参考逐位一致、pad 列写 finfo.min、其余保留预填。"""
        rows, num_pools, num_heads, head_dim = 37, 100, 4, 64
        _block_s, block_p = _pick_tiles(rows, num_pools)  # type: ignore[misc]
        pad = (-num_pools) % block_p
        pools_padded = num_pools + pad
        assert pad > 0, "test requires P not a multiple of the tile BP"
        # 发射宽度之外再留一段缓冲（批量路径以 _MAX_BLOCK_P 为尾部余量），行 stride 大于发射宽度，
        # strided 契约与"不得越界写"才被真正行使。
        width = pools_padded + _MAX_BLOCK_P // 8  # type: ignore[misc]
        torch.manual_seed(11)
        q = torch.randn(rows, num_heads, head_dim, dtype=torch.bfloat16, device=DEVICE)
        pool_keys = torch.randn(num_pools, head_dim, dtype=torch.bfloat16, device=DEVICE)
        weights = torch.randn(rows, num_heads, dtype=torch.bfloat16, device=DEVICE)
        # 参差可见前缀：每行可见池数 0..num_pools，pad 列与越前缀列都必须落 finfo.min。
        prefix = (torch.arange(rows, device=DEVICE) * 7) % (num_pools + 1)
        valid_ref = torch.arange(num_pools, device=DEVICE).unsqueeze(0) < prefix.unsqueeze(-1)
        valid_w = torch.zeros(rows, width, dtype=torch.bool, device=DEVICE)
        valid_w[:, :num_pools] = valid_ref
        out = torch.full((rows, width), 7.0, dtype=torch.float32, device=DEVICE)
        pool_keys_pad = torch.nn.functional.pad(pool_keys, (0, 0, 0, width - num_pools))  # 尾部补零键行

        lightning_indexer_score_into(q, pool_keys_pad, weights, valid_w, out, num_pools, head_dim**-0.5)

        reference = lightning_indexer_score(  # type: ignore[misc]
            q.unsqueeze(0),
            pool_keys.unsqueeze(0),
            weights.unsqueeze(0),
            valid_ref.unsqueeze(0),
            scale=head_dim**-0.5,
        ).squeeze(0)
        assert reference.shape == (rows, num_pools)
        # (1) 参考定义域内的列逐位一致；pad 列 [P, align_up(P, BP)) 写 finfo.min。
        assert torch.equal(out[:, :num_pools], reference)
        assert torch.equal(
            out[:, num_pools:pools_padded],
            torch.full((rows, pad), float(torch.finfo(torch.float32).min), device=DEVICE),
        )
        # (2) 发射宽度之外的列保留 7.0 预填（不得越界写）。
        assert torch.equal(out[:, pools_padded:], torch.full((rows, width - pools_padded), 7.0, device=DEVICE))

    def test_early_return_leaves_prefill_untouched(self) -> None:
        """rows==0 或 num_pools<=0 早退：不发射、不触碰预填。"""
        num_heads, head_dim, width, pools = 4, 64, 128, 100
        out = torch.full((4, width), 7.0, dtype=torch.float32, device=DEVICE)
        valid = torch.zeros(4, width, dtype=torch.bool, device=DEVICE)
        q = torch.randn(4, num_heads, head_dim, dtype=torch.bfloat16, device=DEVICE)
        keys = torch.randn(width, head_dim, dtype=torch.bfloat16, device=DEVICE)
        weights = torch.randn(4, num_heads, dtype=torch.bfloat16, device=DEVICE)
        lightning_indexer_score_into(q[:0], keys, weights[:0], valid[:0], out[:0], pools, head_dim**-0.5)
        lightning_indexer_score_into(q, keys, weights, valid, out, 0, head_dim**-0.5)
        assert torch.equal(out, torch.full((4, width), 7.0, device=DEVICE))
