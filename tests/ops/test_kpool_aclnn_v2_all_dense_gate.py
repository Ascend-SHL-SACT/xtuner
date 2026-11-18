"""GLM-5.3 kpool aclnn_v2 indexer ``select_k <= max_pools`` gate（全 dense/below-width
负载跳过 V2 分支）回归测试。

回归背景（job llm-24 事故）：512 卡 320B 纯文本 SFT run 误带了 VL 启动模板的
``XTUNER_KPOOL_INDEXER_BACKEND=aclnn_v2`` 后端。整包所有文档的可见池数都低于
``select_k``（"全 dense / below-width" 负载，典型是短样本文本 SFT）时，
``_score_and_topk_aclnn_v2`` 的 plan 中没有任何 op item——每文档的全部行都满足
``(p + 1) // kpool <= select_k``（即 ``p <= kpool*(select_k+1)-1 = dense_row_cap``，
``dense_end_doc`` 未被 cap 截断、``c_start == row_hi``，``range(c_start, row_hi,
chunk)`` 为空）——函数只会逐文档发射 dense 准备算子，重建一份与调用方退化分支
（``npu_kpool_topk_indices`` 内 ``select_k > max_pools`` 路径的
``col_ids = torch.arange(select_k, ...)`` +
``torch.where(col_ids.unsqueeze(0) < row_prefix.unsqueeze(-1), col_ids.unsqueeze(0),
_SEL_PAD)``）完全相同的 buffer。512 卡实测每步 ~29 万次发射、~46 秒发射排队等待，而
退化分支只需 2 次发射（job llm-22/25）。

修复语义（本仓库工作区未提交改动，两阶段）：初版 gate 在原有 ``_INDEXER_ACLNN_V2 and
npu and select_k <= 8192 and (select_k <= 2048 or select_k % 1024 == 0)`` 之后追加
``select_k <= max_pools``（ceil 池数）；gate 语义升级后 V2 与 batched 两个分支共用精确
的 ``has_op_rows`` 判据——纯 host 遍历 ranges：
``any((shard_start + r[1] - r[4]) >= index_kpool * (select_k + 1) for r in ranges)``
（``shard_start + row_hi - doc_start`` 是 range 行段的 doc-local 终点，与 plan 的
``chunk_end_doc`` 同量；该判据与 "plan 含 >= 1 个 op item" 精确等价——item 存在
⟺ 某行 doc-local 位置 >= ``dense_row_cap + 1 = kpool*(select_k+1)``）。零 item 调用
在两个分支都直接走退化闭式构造。

ceil/floor 漏窗（升级闭合的窗口）：文档长 L 满足 ``ceil(L/kpool) >= select_k`` 但
``L < kpool*(select_k+1)`` 时——测试规模 select_k=16/kpool=4 即 L ∈ [61, 67]（生产
select_k=512/kpool=4 即 [2045, 2051]），ceil 池数 >= select_k（旧 gate 放行
V2/batched 的死工作）而 floor 前缀 <= select_k（行全 dense、plan 零 item）。65/66/67
是窗内 floor 前缀恰达 select_k 的最紧段（dense 判据是 ``prefix <= select_k``，prefix
== select_k 的行仍是 dense 行）；68 token 起 row 67 前缀 17 > select_k，首个真 op 行
出现。判据还是 shard 感知的：长文档只盖 dense 头部（shard 行 doc-local 全 < cap）的
shard 同样拦截，尽管该文档 ceil 池数远超 select_k。

数值等价性依据：全 dense 负载下 V2 路径的输出与退化闭式解逐位一致——两条构造是同一
算术 ``col_ids < prefix``（V2 的 per-document dense fill
``dense_prefix = (d0 + arange(n_dense) + 1) // kpool`` 在 d0=0 时与 ``_expand_meta``
的 ``row_prefix = (doc_local + 1) // kpool`` 逐行相同，``col_ids``/``_SEL_PAD`` 同值
同 dtype）。注意 floor/ceil 语义差：``_expand_meta`` 的 ``row_prefix`` 是 floor（3
token 文档全 0），``_segment_ranges`` 的 pool 数是 ceil（同文档为 1）。另注：该几何下
"neg" 计划步（op item 的 ``floor_pools == 0`` 分支）不可达——op item 只在 dense 前缀
被 cap 截断时存在，其 ``chunk_end_doc >= dense_row_cap + 1 > kpool``，故全 dense
fixture 里 prefix=0 行（3 token 文档）一律走 dense fill（全 ``_SEL_PAD``），与退化
闭式解的全 pad 行逐位一致。

TestAclnnV2AllDenseClosedFormEquivalence（NPU，直接调 ``_score_and_topk_aclnn_v2``）
    test_all_dense_v2_output_bitwise_equal_degenerate_closed_form: 全 dense 负载下直接
        调 V2 的输出与退化闭式解 buffer（调用方 ``select_k > max_pools`` 分支表达式的
        逐字镜像）``torch.equal`` 逐位一致，chunk=0 / chunk=1024 两种 plan 布局都成立；
        输出 ``[S, select_k]`` int32。fixture 显式断言前置条件：
        ``max_pools == select_k - 1``（紧边界）、其余 gate 子句对该常量恒开（被测的
        唯一开关是新增的 ``select_k <= max_pools``）与 floor/ceil 锚（3 token 文档
        ceil 池数为 1 而 row_prefix 全 0）。
    test_all_dense_v2_plan_emits_zero_op_item_calls: spy 底层
        ``cann_ops_transformer.ops.lightning_indexer`` 算子——全 dense 负载下 V2 函数
        零次 op 调用（~29 万次发射/步事故的根因），输出仍等于闭式解。
TestAclnnV2AllDenseDispatchSkip（NPU，公开入口 ``npu_kpool_topk_indices``）
    test_public_entry_all_dense_skips_v2_bitwise_equal_backend_off: 门开
        （``_INDEXER_ACLNN_V2=True``）+ 全 dense 负载下 spy 证实 V2 函数零调用；输出
        与 ``_INDEXER_ACLNN_V2=False``（走退化路径）的公开入口输出逐位一致，且每行
        展开契约成立（有效槽 ``prefix*kpool`` 个、token 全落在本文档
        ``[doc_start, doc_start + prefix*kpool)`` 且互异；below-width 行 token 恰为
        闭式升序 ``arange(doc_start, doc_start + prefix*kpool)``、其余槽 -1）。
TestAclnnV2MixedPackGateOpen（NPU，混合负载不得误伤）
    test_mixed_pack_full_width_doc_still_dispatches_v2: 含池数 >= select_k 的文档
        （2048/1024/512 token）与 below-width 文档（3/17/60/59 token）的混合 pack 下
        spy 证实 V2 仍被调用，且恰 4 次 op item（2048→2、1024→1、512→1，chunk=1024；
        below-width 文档贡献 0 个 op item、仍走 V2 内部 dense fill 计划步）；每行展开
        契约成立（prefix <= select_k 的行 token 为闭式升序；prefix > select_k 的行恰
        ``select_k*kpool`` 个互异 in-span token）。fixture 显式断言
        ``min_pools < select_k <= max_pools``。
TestAclnnV2CeilFloorLeakWindow（NPU，【gate 语义升级】ceil/floor 漏窗：旧
        ``select_k <= max_pools``（ceil 池数）放行、新 ``has_op_rows`` 拦截的文档长度
        段——测试规模 [61, 67] token，65/66/67 是 floor 前缀恰达 select_k 的最紧段）
    test_leak_window_docs_intercept_v2_dispatch[doc-{65,66,67}]: 单文档 pack 经公开入口
        （``_INDEXER_ACLNN_V2=True``）——fixture 断言 ``ceil(L/4) >= SELECT_K`` 且
        ``L < INDEX_KPOOL*(SELECT_K+1)``（旧 gate 必放行、新 gate 必拦截的前置）、全部
        行 floor 前缀 <= select_k（dense 判据是 ``prefix <= select_k``）；spy 证实 V2
        零调用，输出与 ``_INDEXER_ACLNN_V2=False`` 逐位一致，行级契约成立。
    test_first_op_row_doc_keeps_v2_dispatch: 68 token（= ``kpool*(select_k+1)``，row 67
        前缀 17 > select_k 的首个真 op 行）——spy >= 1 且恰 1 次 op item（前 67 行
        dense fill + 1 行 op）；行级契约成立。边界精确性对照：漏窗内 67 token 拦截、
        68 token 放行，一字不差。
    test_leak_window_v2_output_bitwise_equal_degenerate_closed_form: 全 67-token 文档
        负载直接调 ``_score_and_topk_aclnn_v2``——输出与退化闭式解 ``torch.equal``
        （chunk=0 / chunk=1024 双布局），数值等价钉在漏窗上沿（行前缀最大恰
        select_k=16 的 dense 边界）。
TestAclnnV2ShardAwareGate（NPU，【gate 语义升级】shard 感知：长文档 dense 头部 shard）
    test_dense_head_shard_of_long_doc_intercepts_v2: 300 token 文档（ceil 池数 75 >=
        select_k，旧 gate 必放行）的 shard 只盖 doc-local [0, 60)（行前缀 <= 15 <
        select_k，doc-local 终点 60 < 68）——spy 证实 V2 零调用（旧 ceil gate 不会拦，
        新 has_op_rows 判据按 shard 实际盖到的行段判定）；输出与 backend-off 逐位一致，
        行级契约成立（doc 锚回 global 文档头）。
TestAclnnV2BatchedGateDegenerateShortCircuit（NPU，【gate 语义升级】batched 分支 gate
        同步升级：零 item 调用直通退化构造而非 batched scorer 的死工作）
    test_batched_gate_skips_batched_scorer_on_leak_window: ``_INDEXER_ACLNN_V2=False`` +
        全 67-token 文档负载（旧 ``select_k <= max_pools`` 会派发 batched scorer）——
        spy 证实 ``_score_and_topk_batched`` 零调用、直接走退化闭式分支；输出行级契约
        （prefix 闭式解升序）成立。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import pytest
import torch
from torch import Tensor

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.sparse_mla import kpool_npu
from xtuner.v1.ops.sparse_mla.kpool import kpool_output_width
from xtuner.v1.utils.device import get_device


try:
    from cann_ops_transformer import ops as _cot_ops

    _HAS_V2_OP = _cot_ops is not None and hasattr(_cot_ops, "lightning_indexer")
except ImportError:  # cot_patched 不在 PYTHONPATH（纯 CPU CI）: aclnn_v2 路径不可用
    _cot_ops = None  # type: ignore[assignment]
    _HAS_V2_OP = False


try:
    from xtuner.v1.ops.sparse_mla.kpool_npu import lightning_indexer_score_into

    _HAS_FUSED_INDEXER = lightning_indexer_score_into is not None
except ImportError:  # triton-ascend 缺失（纯 CPU CI）: 批量 triton 路径不可用
    lightning_indexer_score_into = None  # type: ignore[assignment]
    _HAS_FUSED_INDEXER = False


DEVICE = get_device()  # 本盒为 "npu"；纯 CPU 环境退化为 "cpu"
_requires_accelerator = pytest.mark.skipif(DEVICE == "cpu", reason="requires an NPU/CUDA accelerator")
_requires_v2_op = pytest.mark.skipif(
    not _HAS_V2_OP, reason="requires the cann_ops_transformer LightningIndexerV2 op (cot_patched on PYTHONPATH)"
)
_requires_fused_indexer = pytest.mark.skipif(
    not _HAS_FUSED_INDEXER, reason="requires the fused lightning indexer kernel (triton-ascend)"
)

if DEVICE == "npu":
    torch.npu.set_device("npu:0")

# 生产契约规模：SELECT_K=16 对应 index_topk=64 / index_kpool=4（GLM-5.3 文本配置）。
SELECT_K = 16
INDEX_KPOOL = 4
INDEX_HEAD_DIM = 128
NUM_HEADS = 8
# 全 dense / below-width 负载：所有文档 ceil(len/kpool) <= 15 < SELECT_K，gate 必须拦截。
# 60 token 文档是紧边界（15 池 = SELECT_K - 1；64 token 即 16 池、gate 放开）；3 token
# 文档 ceil 池数为 1 而 row_prefix 全 0（floor/ceil 差的锚点，prefix=0 全 pad 行）。
ALL_DENSE_DOC_LENS = [60, 3, 17, 59, 40, 3, 58, 12]
# 混合 pack：2048/1024/512 token 文档的池数（512/256/128）>= SELECT_K，gate 必须放行；
# 3/17/60/59 token 文档仍 below-width（V2 内部走 dense fill 计划步、零 op item）。
GATE_MIXED_DOC_LENS = [2048, 3, 17, 1024, 60, 512, 59]
# ceil/floor 漏窗（gate 语义升级的标的）：L ∈ [61, 67] token 时 ceil(L/4) >= 16（旧
# ``select_k <= max_pools`` gate 放行）而全部行 floor 前缀 <= 16（dense 判据
# ``prefix <= select_k``，plan 零 op item）——65/66/67 是窗内 floor 前缀恰达 select_k
# 的最紧段（生产 select_k=512/kpool=4 的对应窗口是 [2045, 2051] token）。
LEAK_WINDOW_DOC_LENS = [65, 66, 67]
# 首个真 op 行文档长度：``kpool*(select_k+1) = 68``，row 67 前缀 17 > select_k——
# has_op_rows 判据的下界一字不差（67 拦截、68 放行）。
FIRST_OP_DOC_LEN = 68
# shard 感知 fixture：300 token 文档（ceil 池数 75 >= select_k，旧 gate 必放行）的
# shard 只盖其 doc-local [0, 60)（行前缀 <= 15 < select_k，doc-local 终点 60 < 68，
# 新 gate 拦截）。前置 50 token 文档使该文档不在 pack 头部（pool_lo 累计非零）。
DENSE_HEAD_SHARD_GLOBAL_LENS = [50, 300, 40]
DENSE_HEAD_SHARD_START = 50
DENSE_HEAD_SHARD_LEN = 60


@contextmanager
def _patched_attr(module: Any, name: str, value: Any) -> Iterator[None]:
    """临时改写模块级属性，try/finally 保证恢复（模块级 ``_INDEXER_ACLNN_V2`` 等门状态）。"""
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


def _sync() -> None:
    if DEVICE != "cpu":
        getattr(torch, DEVICE).synchronize()


def _build_aclnn_v2_inputs(doc_lens: list[int], seed: int) -> dict[str, Any]:
    """按 ``_score_and_topk_aclnn_v2`` 的入参契约一次性构造全部输入。

    Args:
        doc_lens (list[int]): 各文档 token 数（packing 顺序即文档顺序）。
        seed (int): 随机种子（randn bf16；全 dense 负载下输出与分数无关，纯 int 算术）。

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
    _sync()
    return sel


def _pack_pool_counts(ranges: list[tuple[int, int, int, int, int]]) -> list[int]:
    """各文档的 ceil 池数（``pool_hi - pool_lo``），即 gate 读到的 ``max_pools`` 来源。"""
    return [pool_hi - pool_lo for _row_lo, _row_hi, pool_lo, pool_hi, _doc_start in ranges]


def _degenerate_closed_form(row_prefix: Tensor, select_k: int, device: Any) -> Tensor:
    """调用方退化分支（``npu_kpool_topk_indices`` 内 ``select_k > max_pools`` 路径）
    buffer 构造的逐字镜像：``col_ids = arange(select_k)`` 后
    ``where(col_ids < row_prefix, col_ids, _SEL_PAD)``。"""
    col_ids = torch.arange(select_k, device=device, dtype=torch.int32)
    return torch.where(
        col_ids.unsqueeze(0) < row_prefix.unsqueeze(-1), col_ids.unsqueeze(0), kpool_npu._SEL_PAD
    )


def _build_public_inputs(
    doc_lens: list[int], seed: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, SequenceContext]:
    """构造公开入口 ``npu_kpool_topk_indices`` 的全部张量入参（单 rank、无 SP）。"""
    torch.manual_seed(seed)
    total = sum(doc_lens)
    q = torch.randn(1, total, NUM_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    k = torch.randn(1, total, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    gate_scores = torch.randn(1, total, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    weights = torch.randn(1, total, NUM_HEADS, dtype=torch.bfloat16, device=DEVICE)
    kpool_ape = torch.randn(INDEX_KPOOL, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    ctx = _make_ctx(doc_lens)
    return q, k, gate_scores, weights, kpool_ape, ctx


def _run_public_entry(
    q: Tensor, k: Tensor, gate_scores: Tensor, weights: Tensor, kpool_ape: Tensor, ctx: SequenceContext
) -> Tensor:
    """以生产文本配置（index_topk=64 / kpool=4）调公开入口并同步。"""
    out = kpool_npu.npu_kpool_topk_indices(
        q, k, gate_scores, weights, kpool_ape, ctx,
        index_head_dim=INDEX_HEAD_DIM, index_topk=SELECT_K * INDEX_KPOOL, index_kpool=INDEX_KPOOL,
    )
    _sync()
    return out


def _make_shard_ctx(global_doc_lens: list[int], shard_start: int, shard_len: int) -> SequenceContext:
    """构造 shard 几何的 packed ``SequenceContext``（chunk-zero 文件同一模式）。

    ``cu_seq_lens_q`` 是 global pack 的累计长度，``shard_start`` 定位本 rank shard 的
    global 起点——shard-local 行 ``r`` 的 global token 是 ``shard_start + r``。生产
    ``score_shard_start=None`` 时 gate/``_segment_ranges``/``_expand_meta`` 都用
    ``int(seq_ctx.shard_start)`` 解析，同一几何。
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


def _build_shard_public_inputs(
    global_doc_lens: list[int], shard_start: int, shard_len: int, seed: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, SequenceContext]:
    """构造 shard 几何下公开入口的全部张量入参。

    q/weights 是 shard-local（``shard_len`` 行）；k/gate_scores 必须覆盖 global 全
    pack（``build_pools`` 的契约：池化是整序列操作，生产语义是 key 侧先 SP gather 再
    整 pack 池化，``_segment_ranges`` 累计的 ``pool_lo`` 是 global 池号）。
    """
    torch.manual_seed(seed)
    total = sum(global_doc_lens)
    q = torch.randn(1, shard_len, NUM_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    k = torch.randn(1, total, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    gate_scores = torch.randn(1, total, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    weights = torch.randn(1, shard_len, NUM_HEADS, dtype=torch.bfloat16, device=DEVICE)
    kpool_ape = torch.randn(INDEX_KPOOL, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    ctx = _make_shard_ctx(global_doc_lens, shard_start, shard_len)
    return q, k, gate_scores, weights, kpool_ape, ctx


def _assert_public_row_contract(
    out: Tensor,
    row_prefix: Tensor,
    ctx: SequenceContext,
    *,
    label: str,
    shard_start: int = 0,
    select_k: int = SELECT_K,
) -> None:
    """锁定公开入口输出的行级展开契约（形状 / dtype / 有效槽计数 / 文档跨度 / 互异 /
    below-width 行的闭式升序布局）。

    below-width 行（``prefix <= select_k``）的 slot buffer 是纯算术闭式解，token 恰为
    ``arange(doc_start, doc_start + prefix*kpool)`` 升序、其余 ``index_topk`` 内槽位
    -1；op 行（``prefix > select_k``）的 id 集合由分数决定，只锁互异与 in-span。
    shard 行（``shard_start > 0``，mid-pack/mid-doc 几何）的行号是 shard-local，token
    锚回 global 文档头。
    """
    total = row_prefix.shape[0]
    width = kpool_output_width(select_k * INDEX_KPOOL, INDEX_KPOOL, alignment=512)
    topk_width = select_k * INDEX_KPOOL
    assert out.shape == (total, 1, width), f"{label}: out shape {tuple(out.shape)} != {(total, 1, width)}"
    assert out.dtype == torch.int32, f"{label}: out dtype {out.dtype} != torch.int32"

    flat = out[:, 0, :]
    cu = [int(v) for v in ctx.cu_seq_lens_q_list]
    doc_start_of = [0] * total
    for d in range(len(cu) - 1):
        # shard-local 行 r 的 global token 是 shard_start + r（_segment_ranges 同一几何）。
        for r in range(max(0, cu[d] - shard_start), min(total, cu[d + 1] - shard_start)):
            doc_start_of[r] = cu[d]

    for r in range(total):
        prefix = int(row_prefix[r])
        lo = doc_start_of[r]
        slots = flat[r][:topk_width]
        valid = slots[slots >= 0]
        expect_valid = min(prefix, select_k) * INDEX_KPOOL
        assert valid.numel() == expect_valid, (
            f"{label}: row {r} prefix={prefix} valid token slots {valid.numel()} != {expect_valid}"
        )
        hi = lo + prefix * INDEX_KPOOL
        assert bool(((valid >= lo) & (valid < hi)).all()), (
            f"{label}: row {r} token ids outside doc span [{lo}, {hi}): {valid.tolist()}"
        )
        assert torch.unique(valid).numel() == valid.numel(), f"{label}: row {r} duplicate token ids"
        if prefix <= select_k:
            # 闭式升序布局：有效段恰为 arange，pad 段全 -1（V2 dense fill / 退化闭式解
            # 的列序都保证这一点；legacy per-segment 分数的列序不保证——依赖该断言的
            # 用例须先确保退化分支可达）。
            expect = torch.arange(lo, hi, device=flat.device, dtype=torch.int32)
            assert torch.equal(slots[: prefix * INDEX_KPOOL], expect), (
                f"{label}: row {r} prefix={prefix} closed-form token layout mismatch: "
                f"got={slots.tolist()} expect head={expect.tolist()}"
            )
            assert bool((slots[prefix * INDEX_KPOOL :] == -1).all()), (
                f"{label}: row {r} prefix={prefix} pad slots must stay -1: {slots.tolist()}"
            )


@_requires_accelerator
@_requires_v2_op
@pytest.mark.gpu
class TestAclnnV2AllDenseClosedFormEquivalence:
    """【数值等价】全 dense（``select_k > max_pools``）负载下直接调
    ``_score_and_topk_aclnn_v2``，输出与调用方退化闭式解逐位一致，且 plan 零 op item
    （gate 拦截前 V2 在该负载上只做重复构造，~29 万次发射/步的根因）。"""

    def test_all_dense_v2_output_bitwise_equal_degenerate_closed_form(self):
        """全 dense 负载下 V2 输出 == 退化闭式解 buffer（torch.equal 逐位）。

        fixture 前置条件显式断言：``max_pools == SELECT_K - 1``（紧边界—— ceil 语义下
        15 池文档已入 pack，16 池即放行）；gate 其余子句对该常量恒开（被测唯一开关是
        新增 ``select_k <= max_pools``）；floor/ceil 锚（3 token 文档 ceil 池数 1、
        row_prefix 全 0）。chunk=0 / chunk=1024 两种 plan 布局都比对。
        """
        inputs = _build_aclnn_v2_inputs(ALL_DENSE_DOC_LENS, seed=7)
        ranges = inputs["ranges"]
        row_prefix = inputs["row_prefix"]

        pool_counts = _pack_pool_counts(ranges)
        assert max(pool_counts) == SELECT_K - 1, (
            "fixture precondition: all-dense pack must sit at the tight gate boundary "
            f"(max ceil pools {max(pool_counts)} != SELECT_K - 1 = {SELECT_K - 1})"
        )
        assert SELECT_K <= 8192 and SELECT_K <= 2048, (
            "fixture precondition: every other aclnn_v2 gate clause must stay open for "
            "SELECT_K so only the new `select_k <= max_pools` clause is under test"
        )
        # floor/ceil 锚：ceil 池数为 1 的 3 token 文档，row_prefix 是 floor 语义、全 0。
        cu = [int(v) for v in inputs["ctx"].cu_seq_lens_q_list]
        for d, ln in enumerate(ALL_DENSE_DOC_LENS):
            if ln == 3:
                assert pool_counts[d] == 1, f"3-token doc must have ceil pool count 1, got {pool_counts[d]}"
                assert bool((row_prefix[cu[d] : cu[d + 1]] == 0).all()), (
                    "3-token doc rows must carry floor row_prefix 0 (all-pad rows)"
                )
        assert int(row_prefix.max()) == SELECT_K - 1, (
            "fixture precondition: widest row prefix must equal SELECT_K - 1 (dense rows only)"
        )

        expected = _degenerate_closed_form(row_prefix, SELECT_K, DEVICE)
        sel_whole = _run_aclnn_v2(inputs, chunk=0)
        sel_chunked = _run_aclnn_v2(inputs, chunk=1024)
        for label, sel in (("chunk=0", sel_whole), ("chunk=1024", sel_chunked)):
            assert sel.shape == (sum(ALL_DENSE_DOC_LENS), SELECT_K), (
                f"{label}: sel shape {tuple(sel.shape)} != {(sum(ALL_DENSE_DOC_LENS), SELECT_K)}"
            )
            assert sel.dtype == torch.int32, f"{label}: sel dtype {sel.dtype} != torch.int32"
            assert torch.equal(sel, expected), (
                f"{label}: aclnn_v2 all-dense output differs from the degenerate closed form: "
                f"diff elems={(sel != expected).sum().item()} / {sel.numel()}"
            )
        assert torch.equal(sel_whole, sel_chunked), "chunk=0 and chunk=1024 plans must agree on all-dense packs"

    def test_all_dense_v2_plan_emits_zero_op_item_calls(self, monkeypatch: pytest.MonkeyPatch):
        """全 dense 负载下 V2 的 plan 零 op item：spy 底层 LightningIndexerV2 算子，
        直接调用 ``_score_and_topk_aclnn_v2`` 全程零次 op 调用（回归场景里这些调用
        纯属重建退化闭式解——512 卡 ~29 万次发射/步的根因），输出仍等于闭式解。"""
        inputs = _build_aclnn_v2_inputs(ALL_DENSE_DOC_LENS, seed=7)
        expected = _degenerate_closed_form(inputs["row_prefix"], SELECT_K, DEVICE)

        real_op = _cot_ops.lightning_indexer
        op_calls: list[int] = []

        def _spy_op(*args: Any, **kwargs: Any) -> Any:
            op_calls.append(1)
            return real_op(*args, **kwargs)

        monkeypatch.setattr(_cot_ops, "lightning_indexer", _spy_op)
        sel = _run_aclnn_v2(inputs, chunk=1024)
        assert sum(op_calls) == 0, (
            f"all-dense pack must produce zero LightningIndexerV2 op items, got {sum(op_calls)}"
        )
        assert torch.equal(sel, expected), "all-dense V2 output must still equal the degenerate closed form"


@_requires_accelerator
@_requires_fused_indexer
@pytest.mark.gpu
class TestAclnnV2AllDenseDispatchSkip:
    """【分发跳过】公开入口 ``npu_kpool_topk_indices`` 在门开 + 全 dense 负载下必须
    完全跳过 V2 函数（spy 零调用），输出与 ``_INDEXER_ACLNN_V2=False`` 的退化路径逐位
    一致。退化分支依赖批量打分分支的 ``select_k > max_pools`` 侧门（须 fused indexer
    在场，故整类带 ``_requires_fused_indexer``）。"""

    def test_public_entry_all_dense_skips_v2_bitwise_equal_backend_off(self, monkeypatch: pytest.MonkeyPatch):
        """门开 + 全 dense：spy 证实 ``_score_and_topk_aclnn_v2`` 零调用；输出与门关
        （退化路径）逐位一致，且行级展开契约（含闭式升序布局）成立。"""
        doc_lens = ALL_DENSE_DOC_LENS
        q, k, gate_scores, weights, kpool_ape, ctx = _build_public_inputs(doc_lens, seed=23)
        total = sum(doc_lens)
        pool_counts = _pack_pool_counts(kpool_npu._segment_ranges(ctx, total, INDEX_KPOOL))
        assert max(pool_counts) < SELECT_K, (
            f"fixture precondition: all-dense pack must keep every doc below select_k, got max {max(pool_counts)}"
        )
        row_prefix, _row_doc_start, _slot_offsets, _prefix_host = kpool_npu._expand_meta(
            ctx, total, INDEX_KPOOL, DEVICE
        )

        # spy 闭包在打补丁前捕获原函数（不可重复捕获模块属性——monkeypatch 链会把 spy
        # 自己包进 real，形成无限递归）。``_BATCHED_SCORE`` 显式钉 True：门开/门关两次
        # 调用在 gate 判定后走同一条退化闭式分支，parity 只隔离 gate 本身。
        real = kpool_npu._score_and_topk_aclnn_v2
        calls: list[int] = []

        def _spy(*args: Any, **kwargs: Any) -> Tensor:
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(kpool_npu, "_score_and_topk_aclnn_v2", _spy)
        with _patched_attr(kpool_npu, "_INDEXER_ACLNN_V2", True), _patched_attr(kpool_npu, "_BATCHED_SCORE", True):
            out_on = _run_public_entry(q, k, gate_scores, weights, kpool_ape, ctx)
        assert sum(calls) == 0, (
            f"gate must skip the aclnn_v2 scorer on all-dense packs, got {sum(calls)} calls"
        )

        monkeypatch.setattr(kpool_npu, "_score_and_topk_aclnn_v2", real)  # 门关跑不再 spy
        with _patched_attr(kpool_npu, "_INDEXER_ACLNN_V2", False), _patched_attr(kpool_npu, "_BATCHED_SCORE", True):
            out_off = _run_public_entry(q, k, gate_scores, weights, kpool_ape, ctx)

        assert torch.equal(out_on, out_off), (
            "gate-on (skip) and gate-off (degenerate) public outputs differ: "
            f"diff elems={(out_on != out_off).sum().item()} / {out_on.numel()}"
        )
        _assert_public_row_contract(out_on, row_prefix, ctx, label="gate-on all-dense")


@_requires_accelerator
@_requires_v2_op
@pytest.mark.gpu
class TestAclnnV2MixedPackGateOpen:
    """【不误伤】混合 pack（存在池数 >= select_k 的文档）下 gate 必须放行：V2 仍被
    派发、below-width 文档在 V2 内部照旧走 dense fill 计划步，行级契约不变。"""

    def test_mixed_pack_full_width_doc_still_dispatches_v2(self, monkeypatch: pytest.MonkeyPatch):
        """混合 pack spy 证实 V2 被调用（>= 1 次），且 V2 plan 内恰 4 次 op item
        （chunk=1024 下 2048→2、1024→1、512→1；below-width 文档 0 个）——gate 的放行
        判据是整包 ``max_pools``，不是逐文档；输出行级契约成立。"""
        doc_lens = GATE_MIXED_DOC_LENS
        q, k, gate_scores, weights, kpool_ape, ctx = _build_public_inputs(doc_lens, seed=29)
        total = sum(doc_lens)
        pool_counts = _pack_pool_counts(kpool_npu._segment_ranges(ctx, total, INDEX_KPOOL))
        assert min(pool_counts) < SELECT_K <= max(pool_counts), (
            f"fixture precondition: mixed pack must contain below-width docs (min pools "
            f"{min(pool_counts)}) and a doc with pools >= select_k (max pools {max(pool_counts)})"
        )
        row_prefix, _row_doc_start, _slot_offsets, _prefix_host = kpool_npu._expand_meta(
            ctx, total, INDEX_KPOOL, DEVICE
        )

        real = kpool_npu._score_and_topk_aclnn_v2
        calls: list[int] = []

        def _spy(*args: Any, **kwargs: Any) -> Tensor:
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(kpool_npu, "_score_and_topk_aclnn_v2", _spy)
        # op 层 spy（V2 每个 ``(document, row chunk)`` item 调一次底层算子），锁 plan
        # 几何；从-import 在调用时解析，patch 包属性即可拦截。
        real_op = _cot_ops.lightning_indexer
        op_calls: list[int] = []

        def _spy_op(*args: Any, **kwargs: Any) -> Any:
            op_calls.append(1)
            return real_op(*args, **kwargs)

        monkeypatch.setattr(_cot_ops, "lightning_indexer", _spy_op)
        with _patched_attr(kpool_npu, "_INDEXER_ACLNN_V2", True), _patched_attr(kpool_npu, "_SCORE_ROW_CHUNK", 1024):
            out = _run_public_entry(q, k, gate_scores, weights, kpool_ape, ctx)

        assert sum(calls) >= 1, (
            "mixed pack (some doc with pools >= select_k) must still dispatch the aclnn_v2 scorer"
        )
        # plan 几何锁：chunk=1024 下 op item 只来自全宽文档——2048 token 2 个、1024/512
        # token 各 1 个；below-width 文档（3/17/60/59 token）贡献 0 个 op item。
        assert sum(op_calls) == 4, (
            f"mixed pack op-item count {sum(op_calls)} != 4 (2048->2, 1024->1, 512->1 at "
            "chunk=1024; below-width docs contribute dense fills only)"
        )
        _assert_public_row_contract(out, row_prefix, ctx, label="mixed pack gate-on")


@_requires_accelerator
@_requires_v2_op
@_requires_fused_indexer
@pytest.mark.gpu
class TestAclnnV2CeilFloorLeakWindow:
    """【gate 语义升级：ceil/floor 漏窗】文档长 L 满足 ``ceil(L/4) >= select_k``（旧
    ``select_k <= max_pools`` gate 放行）而 ``L < kpool*(select_k+1)``（全部行 floor
    前缀 <= select_k、plan 零 op item）时，新 ``has_op_rows`` gate 必须拦截——V2 与
    batched 分支都不再为纯 dense 负载做死工作。判据与 "plan 含 op item" 精确等价：
    67 token 拦截、68 token（首个真 op 行）放行，一字不差。退化分支依赖批量打分分支
    可达，整类带 ``_requires_fused_indexer``。"""

    @pytest.mark.parametrize("doc_len", LEAK_WINDOW_DOC_LENS, ids=[f"doc-{n}" for n in LEAK_WINDOW_DOC_LENS])
    def test_leak_window_docs_intercept_v2_dispatch(self, doc_len: int, monkeypatch: pytest.MonkeyPatch):
        """漏窗单文档 pack（65/66/67 token）经公开入口：spy 证实 V2 零调用，输出与
        backend-off 逐位一致，行级契约成立。

        fixture 前置条件显式断言漏窗两侧：``ceil(L/4) >= SELECT_K``（旧 ceil gate 必
        放行）与 ``L < INDEX_KPOOL*(SELECT_K+1)``（新 gate 必拦截）；且全部行 floor
        前缀 <= select_k——dense 判据是 ``prefix <= select_k``，窗内行前缀最大恰
        select_k 仍是 dense 行。
        """
        ceil_pools = -(-doc_len // INDEX_KPOOL)
        assert ceil_pools >= SELECT_K, (
            f"fixture precondition: leak-window doc must have ceil pools >= select_k for the "
            f"old gate to pass it through, got ceil({doc_len}/4)={ceil_pools}"
        )
        assert doc_len < INDEX_KPOOL * (SELECT_K + 1), (
            f"fixture precondition: leak-window doc must keep every row dense (L < "
            f"{INDEX_KPOOL * (SELECT_K + 1)}), got {doc_len}"
        )
        doc_lens = [doc_len]
        q, k, gate_scores, weights, kpool_ape, ctx = _build_public_inputs(doc_lens, seed=37)
        row_prefix, _row_doc_start, _slot_offsets, _prefix_host = kpool_npu._expand_meta(
            ctx, doc_len, INDEX_KPOOL, DEVICE
        )
        assert int(row_prefix.max()) <= SELECT_K, (
            "fixture precondition: every leak-window row must stay a dense row "
            f"(max floor prefix {int(row_prefix.max())} > select_k)"
        )

        real = kpool_npu._score_and_topk_aclnn_v2
        calls: list[int] = []

        def _spy(*args: Any, **kwargs: Any) -> Tensor:
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(kpool_npu, "_score_and_topk_aclnn_v2", _spy)
        with _patched_attr(kpool_npu, "_INDEXER_ACLNN_V2", True), _patched_attr(kpool_npu, "_BATCHED_SCORE", True):
            out_on = _run_public_entry(q, k, gate_scores, weights, kpool_ape, ctx)
        assert sum(calls) == 0, (
            f"{doc_len}-token doc (ceil pools {ceil_pools} >= select_k, all rows dense) must "
            f"be intercepted by the has_op_rows gate, got {sum(calls)} V2 calls"
        )

        monkeypatch.setattr(kpool_npu, "_score_and_topk_aclnn_v2", real)  # 门关跑不再 spy
        with _patched_attr(kpool_npu, "_INDEXER_ACLNN_V2", False), _patched_attr(kpool_npu, "_BATCHED_SCORE", True):
            out_off = _run_public_entry(q, k, gate_scores, weights, kpool_ape, ctx)

        assert torch.equal(out_on, out_off), (
            f"{doc_len}-token doc: gate-on (skip) and gate-off (degenerate) outputs differ: "
            f"diff elems={(out_on != out_off).sum().item()} / {out_on.numel()}"
        )
        _assert_public_row_contract(out_on, row_prefix, ctx, label=f"leak window doc-{doc_len} gate-on")

    def test_first_op_row_doc_keeps_v2_dispatch(self, monkeypatch: pytest.MonkeyPatch):
        """68 token（= ``kpool*(select_k+1)``）文档：row 67 前缀 17 > select_k 的首个
        真 op 行出现，spy >= 1 且恰 1 次 op item（前 67 行 dense fill + 尾行 op）——
        拦截边界一字不差（67 拦、68 放）。"""
        assert FIRST_OP_DOC_LEN == INDEX_KPOOL * (SELECT_K + 1), (
            "fixture precondition: first-op doc length must sit exactly at dense_row_cap + 1"
        )
        doc_lens = [FIRST_OP_DOC_LEN]
        q, k, gate_scores, weights, kpool_ape, ctx = _build_public_inputs(doc_lens, seed=41)
        row_prefix, _row_doc_start, _slot_offsets, _prefix_host = kpool_npu._expand_meta(
            ctx, FIRST_OP_DOC_LEN, INDEX_KPOOL, DEVICE
        )
        # 边界锚：末行是首个 prefix > select_k 的真 op 行；次末行 prefix 恰 == select_k
        # （dense 判据 ``prefix <= select_k`` 的闭边界）。
        assert int(row_prefix[-1]) == SELECT_K + 1, (
            f"fixture precondition: row 67 must be the first op row (prefix "
            f"{int(row_prefix[-1])} != select_k + 1)"
        )
        assert int(row_prefix[-2]) == SELECT_K, (
            f"fixture precondition: row 66 must stay dense at prefix == select_k, got {int(row_prefix[-2])}"
        )

        real = kpool_npu._score_and_topk_aclnn_v2
        calls: list[int] = []

        def _spy(*args: Any, **kwargs: Any) -> Tensor:
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(kpool_npu, "_score_and_topk_aclnn_v2", _spy)
        real_op = _cot_ops.lightning_indexer
        op_calls: list[int] = []

        def _spy_op(*args: Any, **kwargs: Any) -> Any:
            op_calls.append(1)
            return real_op(*args, **kwargs)

        monkeypatch.setattr(_cot_ops, "lightning_indexer", _spy_op)
        with _patched_attr(kpool_npu, "_INDEXER_ACLNN_V2", True), _patched_attr(kpool_npu, "_SCORE_ROW_CHUNK", 1024):
            out = _run_public_entry(q, k, gate_scores, weights, kpool_ape, ctx)

        assert sum(calls) >= 1, (
            f"{FIRST_OP_DOC_LEN}-token doc carries the first true op row (prefix "
            f"{SELECT_K + 1}) and must still dispatch the aclnn_v2 scorer"
        )
        # plan 几何锁：dense fill 盖 doc-local [0, 67)（67 行），op item 恰 [67, 68)。
        assert sum(op_calls) == 1, (
            f"68-token doc must produce exactly one op item at chunk=1024, got {sum(op_calls)}"
        )
        _assert_public_row_contract(out, row_prefix, ctx, label="first-op-row doc gate-on")

    def test_leak_window_v2_output_bitwise_equal_degenerate_closed_form(self):
        """全 67-token 文档负载直接调 ``_score_and_topk_aclnn_v2``：输出与退化闭式解
        ``torch.equal``（chunk=0 / chunk=1024 双布局）——数值等价钉在漏窗上沿（行前缀
        最大恰 select_k=16 的 dense 闭边界）。"""
        doc_lens = [67, 67, 67]
        for ln in doc_lens:
            assert -(-ln // INDEX_KPOOL) >= SELECT_K and ln < INDEX_KPOOL * (SELECT_K + 1), (
                "fixture precondition: workload must sit inside the ceil/floor leak window"
            )
        inputs = _build_aclnn_v2_inputs(doc_lens, seed=43)
        row_prefix = inputs["row_prefix"]
        assert int(row_prefix.max()) == SELECT_K, (
            "fixture precondition: widest row prefix must equal select_k exactly (dense "
            f"closed boundary), got {int(row_prefix.max())}"
        )

        expected = _degenerate_closed_form(row_prefix, SELECT_K, DEVICE)
        sel_whole = _run_aclnn_v2(inputs, chunk=0)
        sel_chunked = _run_aclnn_v2(inputs, chunk=1024)
        for label, sel in (("chunk=0", sel_whole), ("chunk=1024", sel_chunked)):
            assert sel.shape == (sum(doc_lens), SELECT_K), (
                f"{label}: sel shape {tuple(sel.shape)} != {(sum(doc_lens), SELECT_K)}"
            )
            assert sel.dtype == torch.int32, f"{label}: sel dtype {sel.dtype} != torch.int32"
            assert torch.equal(sel, expected), (
                f"{label}: leak-window V2 output differs from the degenerate closed form: "
                f"diff elems={(sel != expected).sum().item()} / {sel.numel()}"
            )


@_requires_accelerator
@_requires_fused_indexer
@pytest.mark.gpu
class TestAclnnV2ShardAwareGate:
    """【gate 语义升级：shard 感知】``has_op_rows`` 按 shard 实际盖到的行段判定：
    长文档只盖 dense 头部（shard 行 doc-local 全 < cap）的 shard 必须拦截，尽管该
    文档 ceil 池数远超 select_k（旧 ``select_k <= max_pools`` gate 不会拦）。"""

    def test_dense_head_shard_of_long_doc_intercepts_v2(self, monkeypatch: pytest.MonkeyPatch):
        """300 token 文档的 shard 只盖 doc-local [0, 60)：公开入口 spy 证实 V2 零调用，
        输出与 backend-off 逐位一致，行级契约（doc 锚回 global 文档头）成立。"""
        global_lens = DENSE_HEAD_SHARD_GLOBAL_LENS
        shard_start, shard_len = DENSE_HEAD_SHARD_START, DENSE_HEAD_SHARD_LEN
        q, k, gate_scores, weights, kpool_ape, ctx = _build_shard_public_inputs(
            global_lens, shard_start, shard_len, seed=47
        )
        ranges = kpool_npu._segment_ranges(ctx, shard_len, INDEX_KPOOL)
        assert len(ranges) == 1, f"fixture precondition: shard must intersect exactly one doc, got {ranges}"
        row_lo, row_hi, pool_lo, pool_hi, doc_start = ranges[0]
        assert doc_start == shard_start and row_lo == 0 and row_hi == shard_len, (
            f"fixture precondition: shard must cover the long doc's head rows [0, {shard_len})"
        )
        doc_ceil_pools = pool_hi - pool_lo
        assert doc_ceil_pools >= SELECT_K, (
            f"fixture precondition: the long doc's ceil pool count {doc_ceil_pools} must be "
            ">= select_k so the OLD ceil gate would dispatch (this is the regression target)"
        )
        assert shard_start + row_hi - doc_start < INDEX_KPOOL * (SELECT_K + 1), (
            "fixture precondition: shard rows must all sit below dense_row_cap + 1 so the "
            "new has_op_rows gate intercepts"
        )
        row_prefix, _row_doc_start, _slot_offsets, _prefix_host = kpool_npu._expand_meta(
            ctx, shard_len, INDEX_KPOOL, DEVICE
        )
        assert int(row_prefix.max()) <= SELECT_K - 1, (
            "fixture precondition: every shard row must be a dense row "
            f"(max floor prefix {int(row_prefix.max())})"
        )

        real = kpool_npu._score_and_topk_aclnn_v2
        calls: list[int] = []

        def _spy(*args: Any, **kwargs: Any) -> Tensor:
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(kpool_npu, "_score_and_topk_aclnn_v2", _spy)
        with _patched_attr(kpool_npu, "_INDEXER_ACLNN_V2", True), _patched_attr(kpool_npu, "_BATCHED_SCORE", True):
            out_on = _run_public_entry(q, k, gate_scores, weights, kpool_ape, ctx)
        assert sum(calls) == 0, (
            f"dense-head shard of a {DENSE_HEAD_SHARD_GLOBAL_LENS[1]}-token doc (ceil pools "
            f"{doc_ceil_pools} >= select_k, shard rows all dense) must be intercepted by the "
            f"shard-aware has_op_rows gate, got {sum(calls)} V2 calls"
        )

        monkeypatch.setattr(kpool_npu, "_score_and_topk_aclnn_v2", real)  # 门关跑不再 spy
        with _patched_attr(kpool_npu, "_INDEXER_ACLNN_V2", False), _patched_attr(kpool_npu, "_BATCHED_SCORE", True):
            out_off = _run_public_entry(q, k, gate_scores, weights, kpool_ape, ctx)

        assert torch.equal(out_on, out_off), (
            "dense-head shard: gate-on (skip) and gate-off (degenerate) outputs differ: "
            f"diff elems={(out_on != out_off).sum().item()} / {out_on.numel()}"
        )
        _assert_public_row_contract(
            out_on, row_prefix, ctx, label="dense-head shard gate-on", shard_start=shard_start
        )


@_requires_accelerator
@_requires_fused_indexer
@pytest.mark.gpu
class TestAclnnV2BatchedGateDegenerateShortCircuit:
    """【gate 语义升级：batched 分支同步】默认后端路径的 ``if select_k <= max_pools``
    同步改为 ``has_op_rows``：漏窗负载（ceil 池数 >= select_k 但零 op item）不再派发
    batched scorer 的死工作，直接走退化闭式构造。"""

    def test_batched_gate_skips_batched_scorer_on_leak_window(self, monkeypatch: pytest.MonkeyPatch):
        """``_INDEXER_ACLNN_V2=False`` + 全 67-token 文档负载：spy 证实
        ``_score_and_topk_batched`` 零调用（旧 ceil gate 会派发它），零 item 调用直通
        退化分支，输出行级契约（prefix 闭式解升序）成立。"""
        doc_lens = [67, 67, 67]
        for ln in doc_lens:
            assert -(-ln // INDEX_KPOOL) >= SELECT_K and ln < INDEX_KPOOL * (SELECT_K + 1), (
                "fixture precondition: workload must sit inside the ceil/floor leak window"
            )
        q, k, gate_scores, weights, kpool_ape, ctx = _build_public_inputs(doc_lens, seed=53)
        total = sum(doc_lens)
        pool_counts = _pack_pool_counts(kpool_npu._segment_ranges(ctx, total, INDEX_KPOOL))
        assert max(pool_counts) >= SELECT_K, (
            f"fixture precondition: old ceil gate must pass this workload (max pools "
            f"{max(pool_counts)}), otherwise the batched-scorer spy proves nothing"
        )
        row_prefix, _row_doc_start, _slot_offsets, _prefix_host = kpool_npu._expand_meta(
            ctx, total, INDEX_KPOOL, DEVICE
        )

        real = kpool_npu._score_and_topk_batched
        calls: list[int] = []

        def _spy(*args: Any, **kwargs: Any) -> Tensor:
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(kpool_npu, "_score_and_topk_batched", _spy)
        with _patched_attr(kpool_npu, "_INDEXER_ACLNN_V2", False), _patched_attr(kpool_npu, "_BATCHED_SCORE", True):
            out = _run_public_entry(q, k, gate_scores, weights, kpool_ape, ctx)
        assert sum(calls) == 0, (
            f"leak-window workload (ceil pools {max(pool_counts)} >= select_k but zero op "
            f"items) must short-circuit to the degenerate branch, got {sum(calls)} batched-scorer calls"
        )
        _assert_public_row_contract(out, row_prefix, ctx, label="batched gate leak-window")
