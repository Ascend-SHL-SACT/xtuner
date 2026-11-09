"""GLM-5.3 SFA skip-recompute（``XTUNER_SFA_SKIP_RECOMPUTE``）回归测试。

重算跳过：DSA 前向 ``npu_sparse_flash_attention`` 的核输出（attn_output + softmax 统计）在原始
前向（reentrant checkpoint 的 no_grad 段）入 stash，backward 内的 checkpoint 重放段直接取用而
不再重跑核。判别式 = checkpoint 重放态（``is_checkpoint_replay``，调用点尾参传入
``Function.forward``）：原始前向恒为 False（kernel+push），重放恒为 True（pop）；未包 checkpoint
的调用（求值/直连梯度）两种态都不是，一律走 kernel+push——孤儿自然沉底，绝不误弹活条目。
主机卸载模式（``_SFA_STASH_OFFLOAD``）：push 侧把核输出 D2H 进池化 pinned 缓冲（源张量
record_stream，在飞拷贝期间设备块不被重用），设备显存零驻留；重放入口先 H2D 预取（与索引
改写的发射窗口重叠），pop 等待事件后返回新分配的设备张量并把 pinned 缓冲归还池。

fixture 以单进程伪造 SP>1 分片（world=2 的 rank-1，shard_start=2728，跨 doc2 边界），走生产
offload Fn 路由（``XTUNER_SP_MLA_OFFLOAD=1`` + row-chunk 0），索引来自真实
``npu_kpool_topk_indices``（打包门必须开启——门关时生产者会在读窗内产 -1，SFA 反向 AICORE 崩溃）。
checkpoint 均经 ``apply_activation_checkpointing``（生产的固定 reentrant 包装，判别式的来源）。

TestSfaSkipRecompute
    test_skip_recompute_grad_parity_two_steps: 设备驻留模式，门开 vs 门关两轮全等——out/dq
        逐位（SFA 前向确定性）、dkv allclose（aclnn 反向原子累加跨 run 非位等价）。
    test_skip_recompute_offload_parity_two_steps: 主机卸载模式同上（D2H→H2D 往返必须无损）。
    test_two_pass_lifo_order: 两个同形不同数据的活图逆序 backward，LIFO 必须各取各的栈顶。
    test_stale_push_sinks_below_live_entries: no_grad 孤儿 push 沉在活条目之下，活图 parity
        不受污染，backward 后孤儿仍留守。
    test_inflight_cap_drops_oldest: 容量上界按最旧淘汰（设备模式，identity 断言）。
    test_inflight_cap_releases_pinned_on_eviction: 卸载模式下淘汰必须释放 pinned 缓冲，且
        新 park 立即复用被释放的缓冲（池计数 + identity 双断言）。
    test_offload_pool_reuse_roundtrip: park→prefetch→pop 数值往返逐位无损；污染池内缓冲后
        重推，拷贝必须覆盖污染（证明复用而非残留）。
    test_offload_device_block_reuse_in_flight: push 后立即释放源张量并再分配同形设备张量，
        在飞 D2H 不被毒化（record_stream 回归探针，竞态翻覆时最敏感）。
    test_orphan_batch_cap_eviction_keeps_live_parity: 孤儿批推满 lane 并触发越界淘汰后
        （淘汰只发生在孤儿之间），活图 parity 仍成立，活条目在栈顶被 pop。
    test_mtp_topology_replay_pops_own_entries: MTP 拓扑（主干三段 + 分支共享同一 lane）四路
        重放按依赖序（分支先、主干逆序）各弹各的 push，弹出值与门关参考逐位一致；
        设备驻留与主机卸载两种模式都必须过。
    test_replay_guard_paths_fall_back_to_recompute: 活图背后条目被清空 / 栈顶条目形状错位 /
        卸载模式 prefetch 后条目被外部清走——三种错配都优雅回退重算（rank0 告警 + 值与门关
        参考一致，miss 重算自身是重放不再 push），绝不弄崩训练；未包 checkpoint 的 grad 调用
        走 push 分支不误弹（值正确、条目守恒）。
    test_gate_off_leaves_stash_empty: 门关（生产默认）完全不经过 stash 管理器。
    test_gate_disabled_when_micro_batch_gt_one: 门开 + INTRA_LAYER_MICRO_BATCH>1 在 import 判定处
        强制关门并 rank0 告警——交错微批的 push/pop 会同形跨微批错配（LIFO 只对 IMB=1 成立）；
        门关不告警，IMB=1/缺省放行。
    test_prefetch_foreign_entry_returns_none: 卸载模式 lane 被外部塞入裸元组条目时 prefetch
        返回 None 而不 raise（重放路径零可 raise 语句），pop 的元组分支照常取值。
    test_pinned_pool_lru_family_cap: pinned 池按 LRU 淘汰超上限形状族的空闲链（变长求值下
        族数有界），热族 MRU 永不被自身触发的淘汰命中。
    test_pop_prunes_empty_lane: pop 排干即修剪 lane，变长求值不为出现过的形状残留空 deque。
"""

import os
from collections import deque

import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")

# kpool_npu 在 import 时把打包门冻结为模块常量，默认值须设在任何 xtuner 导入之前；
# 若本进程内该模块已被先行导入，由下方 autouse fixture 的 monkeypatch 兜底（调用期读取）。
os.environ.setdefault("XTUNER_GLM53_KPOOL_PACK_VISIBLE_TAIL", "1")
os.environ.setdefault("XTUNER_SP_MLA_OFFLOAD", "1")

from xtuner.v1.data_proto import SequenceContext  # noqa: E402
from xtuner.v1.model.utils.checkpointing import apply_activation_checkpointing  # noqa: E402
from xtuner.v1.ops.sparse_mla import kpool_npu  # noqa: E402
from xtuner.v1.ops.sparse_mla.npu_indexer import (  # noqa: E402
    _compute_prefix_extended_kv_slice,
    get_sp_mode_and_slice,
)
from xtuner.v1.ops.sparse_mla.npu_sparse_mla import npu_sparse_mla  # noqa: E402
from xtuner.v1.utils import activation_offload_npu  # noqa: E402
from xtuner.v1.utils.activation_offload_npu import (  # noqa: E402
    _MAX_INFLIGHT_SFA_STASH,
    _sfa_pinned_pool,
    _sfa_stash,
    _SfaStashParked,
    sfa_stash_key,
    sfa_stash_pop,
    sfa_stash_prefetch,
    sfa_stash_push,
)


DOC_LENS = [300, 2600, 40, 2500, 16]
S_TOTAL = sum(DOC_LENS)
WORLD = 2
SHARD_LEN = S_TOTAL // WORLD  # rank-1 均匀分片，跨 doc2（[300, 2900)）边界
SHARD_START = SHARD_LEN
K = 2048
KPOOL = 4
N_HEADS = 8
D_NOPE = 512
D_ROPE = 64
D_INDEX = 128
D_TOTAL = D_NOPE + D_ROPE
SEED = 1234
SCALE = D_TOTAL**-0.5
DEV = "npu:0"

AttnInputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _make_ctx(dev: str) -> SequenceContext:
    """构造 rank-1 伪造 SP>1 分片的 packed 上下文。

    Args:
        dev (str): 设备标签（如 ``"npu:0"``）。

    Returns:
        SequenceContext: cu_seq_lens 为全局文档边界、``shard_start`` 指向 rank-1 分片的上下文。
    """
    cu = [0]
    for ln in DOC_LENS:
        cu.append(cu[-1] + ln)
    cu_t = torch.tensor(cu, dtype=torch.int32, device=dev)
    return SequenceContext(
        input_ids=None,
        cu_seq_lens_q=cu_t,
        cu_seq_lens_k=cu_t,
        max_length_q=max(DOC_LENS),
        max_length_k=max(DOC_LENS),
        device=dev,
        shard_start=SHARD_START,
    )


def _produce_indices(ctx: SequenceContext, dev: str) -> torch.Tensor:
    """真实生产者产打包索引（固定 seed 确定性复现）。

    Args:
        ctx (SequenceContext): 分片上下文。
        dev (str): 设备标签。

    Returns:
        torch.Tensor: ``[SHARD_LEN, 1, K]`` int32 打包索引。
    """
    torch.manual_seed(SEED)
    torch_npu.npu.manual_seed(SEED)
    q_idx = torch.randn(1, SHARD_LEN, N_HEADS, D_INDEX, dtype=torch.bfloat16, device=dev)
    k_idx = torch.randn(1, S_TOTAL, D_INDEX, dtype=torch.bfloat16, device=dev)
    gate = torch.randn(1, S_TOTAL, D_INDEX, dtype=torch.bfloat16, device=dev)
    weights = torch.randn(1, SHARD_LEN, N_HEADS, dtype=torch.bfloat16, device=dev)
    ape = torch.randn(KPOOL, D_INDEX, dtype=torch.bfloat16, device=dev)
    indices = kpool_npu.npu_kpool_topk_indices(
        q_idx,
        k_idx,
        gate,
        weights,
        ape,
        ctx,
        index_head_dim=D_INDEX,
        index_topk=K,
        index_kpool=KPOOL,
    )
    assert indices.shape == (SHARD_LEN, 1, K), f"unexpected pack shape {tuple(indices.shape)}"
    return indices


def _make_case(dev: str = DEV) -> tuple[SequenceContext, torch.Tensor]:
    """跨分片边界用例：ctx + 打包索引，并断言 SP>1 路由成立。

    Args:
        dev (str): 设备标签。

    Returns:
        tuple[SequenceContext, torch.Tensor]: (ctx, indices)。
    """
    ctx = _make_ctx(dev)
    indices = _produce_indices(ctx, dev)
    is_sp1, _, _ = get_sp_mode_and_slice(ctx, SHARD_START, SHARD_LEN, dev, _compute_prefix_extended_kv_slice)
    assert not is_sp1, "SP>1 shard misdetected as SP=1"
    return ctx, indices


def _make_attn(seed: int, dev: str = DEV) -> AttnInputs:
    """按 seed 生成注意力输入（q/kv 压 0.05 尺度，g 为反向种子梯度）。

    Args:
        seed (int): 随机种子。
        dev (str): 设备标签。

    Returns:
        AttnInputs: (q [S, N, D_TOTAL], kv [S_TOTAL, 1, D_TOTAL], g [S, N, D_NOPE])，均 bf16。
    """
    torch.manual_seed(seed)
    torch_npu.npu.manual_seed(seed)
    q = (torch.randn(SHARD_LEN, N_HEADS, D_TOTAL, device=dev) * 0.05).to(torch.bfloat16)
    kv = (torch.randn(S_TOTAL, 1, D_TOTAL, device=dev) * 0.05).to(torch.bfloat16)
    g = torch.randn(SHARD_LEN, N_HEADS, D_NOPE, device=dev).to(torch.bfloat16)
    return q, kv, g


class _AttnBlock(torch.nn.Module):
    """(q, kv) -> raw_output 的 DSA 前向模块，可被 checkpoint 包装重放。

    Args:
        indices (torch.Tensor): 打包索引。
        ctx (SequenceContext): 分片上下文。
    """

    def __init__(self, indices: torch.Tensor, ctx: SequenceContext) -> None:
        super().__init__()
        self._indices = indices
        self._ctx = ctx

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        return npu_sparse_mla(q, kv, self._indices, SCALE, D_NOPE, seq_ctx=self._ctx).raw_output


def _make_block(indices: torch.Tensor, ctx: SequenceContext) -> _AttnBlock:
    """构造 (q, kv) -> raw_output 的 DSA 前向模块。

    Args:
        indices (torch.Tensor): 打包索引。
        ctx (SequenceContext): 分片上下文。

    Returns:
        _AttnBlock: 前向模块。
    """
    return _AttnBlock(indices, ctx)


def _run_checkpoint_arm(attn: AttnInputs, block: _AttnBlock) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """reentrant checkpoint 前向+反向，返回 (out, dq, dkv) 的 detached 拷贝。

    Args:
        attn (AttnInputs): ``_make_attn`` 的输出。
        block (_AttnBlock): ``_make_block`` 的输出。

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (out, dq, dkv)。
    """
    q, kv, g = attn
    qt = q.clone().requires_grad_()
    kvt = kv.clone().requires_grad_()
    out = apply_activation_checkpointing(block)(qt, kvt)
    out.backward(g)
    return out.detach(), qt.grad.detach(), kvt.grad.detach()


def _assert_gate_on_off_parity(monkeypatch: pytest.MonkeyPatch, block: _AttnBlock) -> None:
    """两轮 seed 下门开 vs 门关全等：out/dq 逐位（SFA 前向确定性），dkv allclose
    （aclnn 反向原子累加跨 run 非位等价）；门开臂必须真实经过 stash（计数 pop 佐证），
    每轮用尽即清（pop 排干即修剪 lane）。"""
    # pop 排干即修剪 lane：排空后的 stash 字典为空，不能再用字典非空证明门开臂用过
    # stash——包一层计数 pop（与 MTP 拓扑用例同法）。
    orig_pop = npu_sparse_mla.__globals__["sfa_stash_pop"]
    popped: list[object] = []

    def _count_pop(*args: object, **kwargs: object) -> object:
        popped.append(args[0])
        return orig_pop(*args, **kwargs)

    monkeypatch.setitem(npu_sparse_mla.__globals__, "sfa_stash_pop", _count_pop)
    for seed in (SEED + 1, SEED + 2):
        attn = _make_attn(seed)
        monkeypatch.setitem(npu_sparse_mla.__globals__, "_SFA_SKIP_RECOMPUTE", False)
        ref = _run_checkpoint_arm(attn, block)
        monkeypatch.setitem(npu_sparse_mla.__globals__, "_SFA_SKIP_RECOMPUTE", True)
        skip = _run_checkpoint_arm(attn, block)
        assert torch.equal(skip[0], ref[0])
        assert torch.equal(skip[1], ref[1])
        assert torch.allclose(skip[2].float(), ref[2].float(), atol=1e-2, rtol=1e-2)
        # 每轮用尽即清，轮间不残留活条目（lane 已被修剪）。
        assert all(not lane for lane in _sfa_stash.values())
    assert popped, "gate-on arms must have exercised the stash manager"


@pytest.fixture(autouse=True)
def _sfa_route(monkeypatch: pytest.MonkeyPatch):
    """钉住生产 offload Fn 路由与设备驻留默认（SP>1 + offload=1 + row-chunk 0），隔离 stash。"""
    monkeypatch.setenv("XTUNER_SP_MLA_OFFLOAD", "1")
    monkeypatch.setattr(kpool_npu, "_PACK_VISIBLE_TAIL", True)
    monkeypatch.setitem(npu_sparse_mla.__globals__, "_SFA_GRAD_ROW_CHUNK", 0)
    # 测试默认设备驻留模式；卸载用例自行翻转（防环境变量泄漏进无关用例）。
    monkeypatch.setattr(activation_offload_npu, "_SFA_STASH_OFFLOAD", False)
    _sfa_stash.clear()
    _sfa_pinned_pool.clear()
    yield
    torch.cuda.synchronize()
    _sfa_stash.clear()
    _sfa_pinned_pool.clear()


class TestSfaSkipRecompute:
    """SFA skip-recompute 的数值等价与 stash 生命周期回归。"""

    @pytest.mark.gpu
    def test_skip_recompute_grad_parity_two_steps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ctx, indices = _make_case()
        _assert_gate_on_off_parity(monkeypatch, _make_block(indices, ctx))

    @pytest.mark.gpu
    def test_skip_recompute_offload_parity_two_steps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 主机卸载模式：park（D2H）→ 预取（H2D）→ pop 全链路与重算路径全等。
        monkeypatch.setattr(activation_offload_npu, "_SFA_STASH_OFFLOAD", True)
        ctx, indices = _make_case()
        _assert_gate_on_off_parity(monkeypatch, _make_block(indices, ctx))

    @pytest.mark.gpu
    def test_two_pass_lifo_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ctx, indices = _make_case()
        block = _make_block(indices, ctx)
        attn_a, attn_b = _make_attn(SEED + 11), _make_attn(SEED + 12)
        monkeypatch.setitem(npu_sparse_mla.__globals__, "_SFA_SKIP_RECOMPUTE", False)
        ref_a = _run_checkpoint_arm(attn_a, block)
        ref_b = _run_checkpoint_arm(attn_b, block)
        monkeypatch.setitem(npu_sparse_mla.__globals__, "_SFA_SKIP_RECOMPUTE", True)
        # 两个同形不同数据的活图；逆序 backward——LIFO 必须各取各的栈顶。
        wrapped = apply_activation_checkpointing(block)
        leaves = []
        outs = []
        for attn in (attn_a, attn_b):
            qt = attn[0].clone().requires_grad_()
            kvt = attn[1].clone().requires_grad_()
            leaves.append((qt, kvt))
            outs.append(wrapped(qt, kvt))
        outs[1].backward(attn_b[2])
        outs[0].backward(attn_a[2])
        assert torch.equal(outs[1].detach(), ref_b[0])
        assert torch.equal(outs[0].detach(), ref_a[0])
        assert torch.equal(leaves[1][0].grad, ref_b[1])
        assert torch.equal(leaves[0][0].grad, ref_a[1])
        assert torch.allclose(leaves[1][1].grad.float(), ref_b[2].float(), atol=1e-2, rtol=1e-2)
        assert torch.allclose(leaves[0][1].grad.float(), ref_a[2].float(), atol=1e-2, rtol=1e-2)
        assert all(not lane for lane in _sfa_stash.values())

    @pytest.mark.gpu
    def test_stale_push_sinks_below_live_entries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ctx, indices = _make_case()
        block = _make_block(indices, ctx)
        monkeypatch.setitem(npu_sparse_mla.__globals__, "_SFA_SKIP_RECOMPUTE", True)
        # 孤儿：no_grad 求值调用只 push 不 pop（生产 = eval/probe 场景）。
        orphan = _make_attn(SEED + 21)
        with torch.no_grad():
            block(orphan[0], orphan[1])
        assert sum(len(lane) for lane in _sfa_stash.values()) == 1
        # 活图用不同数据：LIFO 栈顶必须是活条目，孤儿沉底。
        attn = _make_attn(SEED + 22)
        monkeypatch.setitem(npu_sparse_mla.__globals__, "_SFA_SKIP_RECOMPUTE", False)
        ref = _run_checkpoint_arm(attn, block)
        monkeypatch.setitem(npu_sparse_mla.__globals__, "_SFA_SKIP_RECOMPUTE", True)
        skip = _run_checkpoint_arm(attn, block)
        assert torch.equal(skip[0], ref[0])
        assert torch.equal(skip[1], ref[1])
        assert torch.allclose(skip[2].float(), ref[2].float(), atol=1e-2, rtol=1e-2)
        # 活条目已被 pop；孤儿仍留守同一条 lane。
        assert sum(len(lane) for lane in _sfa_stash.values()) == 1

    @pytest.mark.gpu
    def test_inflight_cap_drops_oldest(self) -> None:
        # 纯 stash 管理器单元测试（设备驻留模式）：容量上界按最旧淘汰，identity 判定。
        key = sfa_stash_key(torch.empty(4, 2, 8), torch.empty(16, 1, 8))
        entries = []
        for i in range(_MAX_INFLIGHT_SFA_STASH + 2):
            entry = (
                torch.full((4, 2, 8), float(i)),
                torch.full((2, 4), float(i)),
                torch.full((2, 4), float(i)),
            )
            entries.append(entry)
            sfa_stash_push(key, *entry)
        lane = _sfa_stash[key]
        assert len(lane) == _MAX_INFLIGHT_SFA_STASH
        # push 侧会把三元素重组为新元组入栈，identity 只对元素成立——用首元素判定归属。
        kept_firsts = [e[0] for e in lane]
        for evicted in entries[:2]:
            assert all(t is not evicted[0] for t in kept_firsts)
        for survivor in entries[2:]:
            assert any(t is survivor[0] for t in kept_firsts)

    @pytest.mark.gpu
    def test_inflight_cap_releases_pinned_on_eviction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 卸载模式纯管理器单测：淘汰必须释放 pinned 缓冲，且（池先前为空）新 park 恰好
        # 复用被释放的三个——释放既不泄漏也不迟滞。
        monkeypatch.setattr(activation_offload_npu, "_SFA_STASH_OFFLOAD", True)
        key = sfa_stash_key(torch.empty(4, 2, 8, device=DEV), torch.empty(16, 1, 8, device=DEV))

        def _entry(i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return (
                torch.full((4, 2, 8), float(i), device=DEV),
                torch.full((2, 4), float(i), device=DEV),
                torch.full((2, 4), float(i), device=DEV),
            )

        for i in range(_MAX_INFLIGHT_SFA_STASH):
            sfa_stash_push(key, *_entry(i))
        lane = _sfa_stash[key]
        assert len(lane) == _MAX_INFLIGHT_SFA_STASH
        assert all(isinstance(e, _SfaStashParked) for e in lane)
        assert sum(len(free) for free in _sfa_pinned_pool.values()) == 0
        oldest = lane[0]
        oldest_bufs = list(oldest.buffers)
        sfa_stash_push(key, *_entry(_MAX_INFLIGHT_SFA_STASH))
        assert oldest.buffers == []  # 淘汰即释放
        assert len(lane) == _MAX_INFLIGHT_SFA_STASH
        assert sum(len(free) for free in _sfa_pinned_pool.values()) == 0
        assert {id(b) for b in lane[-1].buffers} == {id(b) for b in oldest_bufs}

    @pytest.mark.gpu
    def test_offload_pool_reuse_roundtrip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # park→prefetch→pop 往返逐位无损；pinned 池先取后还；污染池内缓冲后重推，
        # D2H 拷贝必须覆盖污染（证明真复用而非残留值）。
        monkeypatch.setattr(activation_offload_npu, "_SFA_STASH_OFFLOAD", True)
        q_ref = torch.empty(4, 2, 8, device=DEV)
        kv_ref = torch.empty(16, 1, 8, device=DEV)
        key = sfa_stash_key(q_ref, kv_ref)
        src = (
            torch.full((4, 2, 8), 7.0, device=DEV),
            torch.full((2, 4), 7.0, device=DEV),
            torch.full((2, 4), 7.0, device=DEV),
        )
        sfa_stash_push(key, *src)
        lane = _sfa_stash[key]
        assert len(lane) == 1 and isinstance(lane[0], _SfaStashParked)
        assert sum(len(free) for free in _sfa_pinned_pool.values()) == 0
        entry = sfa_stash_prefetch(key)
        outs = sfa_stash_pop(key, q_ref, kv_ref, prefetch=entry)
        assert torch.equal(outs[0], src[0])
        assert torch.equal(outs[1], src[1])
        assert torch.equal(outs[2], src[2])
        assert outs[0] is not src[0]  # 恢复的是新分配的设备张量
        assert not lane
        assert sum(len(free) for free in _sfa_pinned_pool.values()) == 3
        # 第二轮：毒化池内全部缓冲后重推同值源，拷贝必须覆盖毒值。
        for free in _sfa_pinned_pool.values():
            for buf in free:
                buf.fill_(-1.0)
        sfa_stash_push(key, *src)
        entry = sfa_stash_prefetch(key)
        outs = sfa_stash_pop(key, q_ref, kv_ref, prefetch=entry)
        assert torch.equal(outs[0], torch.full((4, 2, 8), 7.0, device=DEV))
        assert torch.equal(outs[1], torch.full((2, 4), 7.0, device=DEV))
        assert torch.equal(outs[2], torch.full((2, 4), 7.0, device=DEV))
        assert sum(len(free) for free in _sfa_pinned_pool.values()) == 3

    @pytest.mark.gpu
    def test_offload_device_block_reuse_in_flight(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 回归探针：push 后立刻释放源张量并再分配同形设备张量。若 record_stream 缺失，
        # 设备块会被提前重用、填充核毒化在飞 D2H 的读源；record_stream 让块延迟到拷贝
        # 完成才回归分配池。该探针在缺陷回归时以竞态方式翻覆（最敏感处）。
        monkeypatch.setattr(activation_offload_npu, "_SFA_STASH_OFFLOAD", True)
        shape = (4096, 8, 512)
        src = torch.full(shape, 5.0, dtype=torch.bfloat16, device=DEV)
        smax = torch.zeros(8, 4096, device=DEV)
        ssum = torch.zeros(8, 4096, device=DEV)
        expected = src.clone()
        q_ref = torch.empty(shape[0], shape[1], shape[2], device=DEV)
        kv_ref = torch.empty(16384, 1, shape[2], device=DEV)
        key = sfa_stash_key(q_ref, kv_ref)
        sfa_stash_push(key, src, smax, ssum)
        del src, smax, ssum
        poison = torch.full(shape, -7.0, dtype=torch.bfloat16, device=DEV)
        torch.cuda.synchronize()
        assert poison.view(-1)[0].item() == -7.0  # 毒化确实执行过
        entry = sfa_stash_prefetch(key)
        outs = sfa_stash_pop(key, q_ref, kv_ref, prefetch=entry)
        assert torch.equal(outs[0], expected)
        assert poison.view(-1)[0].item() == -7.0

    @pytest.mark.gpu
    def test_orphan_batch_cap_eviction_keeps_live_parity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 孤儿批把 lane 推满并触发越界淘汰（淘汰只发生在孤儿之间）；随后活图 parity
        # 仍成立——活 push 淘汰一个孤儿，活条目仍在栈顶被 pop。
        monkeypatch.setattr(activation_offload_npu, "_SFA_STASH_OFFLOAD", True)
        monkeypatch.setitem(npu_sparse_mla.__globals__, "_SFA_SKIP_RECOMPUTE", True)
        ctx, indices = _make_case()
        block = _make_block(indices, ctx)
        for i in range(_MAX_INFLIGHT_SFA_STASH + 4):
            orphan = _make_attn(SEED + 100 + i)
            with torch.no_grad():
                block(orphan[0], orphan[1])
        assert sum(len(lane) for lane in _sfa_stash.values()) == _MAX_INFLIGHT_SFA_STASH
        attn = _make_attn(SEED + 200)
        monkeypatch.setitem(npu_sparse_mla.__globals__, "_SFA_SKIP_RECOMPUTE", False)
        ref = _run_checkpoint_arm(attn, block)
        monkeypatch.setitem(npu_sparse_mla.__globals__, "_SFA_SKIP_RECOMPUTE", True)
        skip = _run_checkpoint_arm(attn, block)
        assert torch.equal(skip[0], ref[0])
        assert torch.equal(skip[1], ref[1])
        assert torch.allclose(skip[2].float(), ref[2].float(), atol=1e-2, rtol=1e-2)
        # 活条目被 pop；留下的全是孤儿。
        assert sum(len(lane) for lane in _sfa_stash.values()) == _MAX_INFLIGHT_SFA_STASH - 1

    @pytest.mark.gpu
    def test_mtp_topology_replay_pops_own_entries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # MTP 拓扑：主干三段 + 分支共享同一 stash lane（同形）。四路重放按依赖序
        # （分支先、主干逆序）各弹各的 push，弹出值与门关参考逐位一致——lane 全局
        # 无序、按 LIFO 配对的结构性验证。设备驻留与主机卸载两种模式都必须过。
        ctx, indices = _make_case()
        block = _make_block(indices, ctx)
        wrapped = apply_activation_checkpointing(block)
        q, kv, _ = _make_attn(SEED + 50)

        def _pad(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.pad(x, (0, D_ROPE))

        # 门关全图反向基准（重算参考），与 stash 模式无关，只跑一次。
        monkeypatch.setitem(npu_sparse_mla.__globals__, "_SFA_SKIP_RECOMPUTE", False)
        qt_ref = q.clone().requires_grad_()
        kvt_ref = kv.clone().requires_grad_()
        o1 = wrapped(qt_ref, kvt_ref)
        o2 = wrapped(_pad(o1), kvt_ref)
        o3 = wrapped(_pad(o2), kvt_ref)
        om = wrapped(_pad(o3), kvt_ref)
        (o3.float().sum() + om.float().sum()).backward()
        dq_ref, dkv_ref = qt_ref.grad.detach(), kvt_ref.grad.detach()

        # 文件内 npu_sparse_mla 名字绑定的是函数；模块字典经 __globals__ 拿到，
        # Function 体对 sfa_stash_pop 的全局查找即落在该字典上。
        orig_pop = npu_sparse_mla.__globals__["sfa_stash_pop"]
        popped: list[torch.Tensor] = []

        def _rec_pop(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            outs = orig_pop(*args, **kwargs)
            popped.append(outs[0].detach().clone())
            return outs

        monkeypatch.setitem(npu_sparse_mla.__globals__, "sfa_stash_pop", _rec_pop)
        for offload in (False, True):
            _sfa_stash.clear()
            _sfa_pinned_pool.clear()
            popped.clear()
            monkeypatch.setattr(activation_offload_npu, "_SFA_STASH_OFFLOAD", offload)
            # 参考前向输出：门关 no_grad 直连（不进 stash）。
            monkeypatch.setitem(npu_sparse_mla.__globals__, "_SFA_SKIP_RECOMPUTE", False)
            with torch.no_grad():
                r1 = block(q, kv)
                r2 = block(_pad(r1), kv)
                r3 = block(_pad(r2), kv)
                rm = block(_pad(r3), kv)
            refs = (rm, r3, r2, r1)
            # 门开全图：push 顺序 [p1, p2, p3, pm]；重放/弹出顺序必须是依赖序逆推
            # [rm, r3, r2, r1]（分支图先于其主干来源反传）。
            monkeypatch.setitem(npu_sparse_mla.__globals__, "_SFA_SKIP_RECOMPUTE", True)
            qt = q.clone().requires_grad_()
            kvt = kv.clone().requires_grad_()
            o1 = wrapped(qt, kvt)
            o2 = wrapped(_pad(o1), kvt)
            o3 = wrapped(_pad(o2), kvt)
            om = wrapped(_pad(o3), kvt)
            (o3.float().sum() + om.float().sum()).backward()
            assert len(popped) == 4
            # 栈内条目是核输出的原形（可能带前导 1 维），raw_output 是其 squeeze 视图。
            for got, ref in zip(popped, refs):
                assert torch.equal(got.reshape(ref.shape), ref)
            assert torch.equal(qt.grad, dq_ref)
            assert torch.allclose(kvt.grad.float(), dkv_ref.float(), atol=1e-2, rtol=1e-2)
            assert all(not lane for lane in _sfa_stash.values())
            if offload:
                # 4 个 park 的 12 个 pinned 缓冲全部归还池。
                assert sum(len(free) for free in _sfa_pinned_pool.values()) == 12

    @pytest.mark.gpu
    def test_replay_guard_paths_fall_back_to_recompute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ctx, indices = _make_case()
        block = _make_block(indices, ctx)
        wrapped = apply_activation_checkpointing(block)
        monkeypatch.setitem(npu_sparse_mla.__globals__, "_SFA_SKIP_RECOMPUTE", True)
        # (a) 孤儿先入栈（lane 非空）；未包 checkpoint 的 grad 调用不是重放，必须走
        #     kernel+push 分支而非误弹孤儿——值正确、条目守恒。
        orphan = _make_attn(SEED + 30)
        with torch.no_grad():
            block(orphan[0], orphan[1])
        assert sum(len(lane) for lane in _sfa_stash.values()) == 1
        attn = _make_attn(SEED + 31)
        qt = attn[0].clone().requires_grad_()
        kvt = attn[1].clone().requires_grad_()
        monkeypatch.setitem(npu_sparse_mla.__globals__, "_SFA_SKIP_RECOMPUTE", False)
        ref_out = block(qt, kvt)
        ref_out.backward(attn[2])
        ref_dq, ref_dkv = qt.grad.detach(), kvt.grad.detach()
        qt2 = attn[0].clone().requires_grad_()
        kvt2 = attn[1].clone().requires_grad_()
        monkeypatch.setitem(npu_sparse_mla.__globals__, "_SFA_SKIP_RECOMPUTE", True)
        got_out = block(qt2, kvt2)
        got_out.backward(attn[2])
        assert torch.equal(got_out.detach(), ref_out.detach())
        assert torch.equal(qt2.grad, ref_dq)
        assert torch.allclose(kvt2.grad.float(), ref_dkv.float(), atol=1e-2, rtol=1e-2)
        # 孤儿仍在栈底、grad 调用新 push 在栈顶：2 条守恒（误弹孤儿会丢条目且值错）。
        assert sum(len(lane) for lane in _sfa_stash.values()) == 2

        # 之后所有守卫路径的告警都经 log_rank0，捕获之。
        warns: list[str] = []

        class _WarnRecorder:
            def warning(self, msg: str, *args: object, **kwargs: object) -> None:
                warns.append(msg)

        monkeypatch.setattr(activation_offload_npu, "log_rank0", _WarnRecorder())

        # (b) 活图背后条目被清空（如外部清理）：replay 优雅回退重算并告警，值与门关
        #     参考一致；miss 重算自身是重放，不再 push，lane 保持空。
        attn = _make_attn(SEED + 32)
        qt = attn[0].clone().requires_grad_()
        kvt = attn[1].clone().requires_grad_()
        monkeypatch.setitem(npu_sparse_mla.__globals__, "_SFA_SKIP_RECOMPUTE", False)
        ref_out = block(qt, kvt)
        ref_out.backward(attn[2])
        ref_dq, ref_dkv = qt.grad.detach(), kvt.grad.detach()
        qt2 = attn[0].clone().requires_grad_()
        kvt2 = attn[1].clone().requires_grad_()
        monkeypatch.setitem(npu_sparse_mla.__globals__, "_SFA_SKIP_RECOMPUTE", True)
        out = wrapped(qt2, kvt2)
        for lane in _sfa_stash.values():
            lane.clear()
        out.backward(attn[2])
        assert any("stash is empty" in w for w in warns)
        assert torch.equal(out.detach(), ref_out.detach())
        assert torch.equal(qt2.grad, ref_dq)
        assert torch.allclose(kvt2.grad.float(), ref_dkv.float(), atol=1e-2, rtol=1e-2)
        assert all(not lane for lane in _sfa_stash.values())
        # (c) 栈顶条目形状错位：pop 消费丢弃错配条目（自愈，lane 不被楔死）并回退重算。
        warns.clear()
        attn = _make_attn(SEED + 33)
        qt = attn[0].clone().requires_grad_()
        kvt = attn[1].clone().requires_grad_()
        monkeypatch.setitem(npu_sparse_mla.__globals__, "_SFA_SKIP_RECOMPUTE", False)
        ref_out = block(qt, kvt)
        ref_out.backward(attn[2])
        ref_dq, ref_dkv = qt.grad.detach(), kvt.grad.detach()
        qt2 = attn[0].clone().requires_grad_()
        kvt2 = attn[1].clone().requires_grad_()
        monkeypatch.setitem(npu_sparse_mla.__globals__, "_SFA_SKIP_RECOMPUTE", True)
        out = wrapped(qt2, kvt2)
        bad = (torch.zeros(2, 3, 4), torch.zeros(3), torch.zeros(3))
        for lane in _sfa_stash.values():
            lane.append(bad)
        out.backward(attn[2])
        assert any("misalignment" in w for w in warns)
        assert torch.equal(out.detach(), ref_out.detach())
        assert torch.equal(qt2.grad, ref_dq)
        assert torch.allclose(kvt2.grad.float(), ref_dkv.float(), atol=1e-2, rtol=1e-2)
        # 错配条目已消费丢弃，活条目留守栈底由后续容量淘汰收割。
        assert sum(len(lane) for lane in _sfa_stash.values()) == 1
        # (b2) 卸载模式：prefetch 到 pop 之间条目被外部清走——空 lane 上的 pop 必须
        #      释放该 prefetch 持有的 pinned 缓冲（防泄漏）并返回 None 回退。
        warns.clear()
        _sfa_stash.clear()
        _sfa_pinned_pool.clear()
        monkeypatch.setattr(activation_offload_npu, "_SFA_STASH_OFFLOAD", True)
        q_probe = torch.zeros(4, 2, 8, dtype=torch.bfloat16, device=DEV)
        kv_probe = torch.zeros(16, 1, 8, dtype=torch.bfloat16, device=DEV)
        probe_key = sfa_stash_key(q_probe, kv_probe)
        sfa_stash_push(
            probe_key,
            torch.zeros(4, 2, 8, dtype=torch.bfloat16, device=DEV),
            torch.zeros(4, 2, dtype=torch.float32, device=DEV),
            torch.zeros(4, 2, dtype=torch.float32, device=DEV),
        )
        handle = sfa_stash_prefetch(probe_key)
        assert handle is not None
        _sfa_stash[probe_key].clear()
        assert sfa_stash_pop(probe_key, q_probe, kv_probe, prefetch=handle) is None
        assert any("stash is empty" in w for w in warns)
        # 3 个 pinned 缓冲全部归还池。
        assert sum(len(free) for free in _sfa_pinned_pool.values()) == 3

    @pytest.mark.gpu
    def test_gate_off_leaves_stash_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(npu_sparse_mla.__globals__, "_SFA_SKIP_RECOMPUTE", False)
        ctx, indices = _make_case()
        block = _make_block(indices, ctx)
        _run_checkpoint_arm(_make_attn(SEED + 41), block)
        assert not _sfa_stash, "gate off (production default) must never touch the stash manager"

    @pytest.mark.gpu
    def test_gate_disabled_when_micro_batch_gt_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 门 + 微批拆分守卫：IMB>1 时 import 判定必须强制关门并 rank0 告警——交错
        # 微批的 push/pop 会跨微批同形错配、静默算错梯度（LIFO 配对只对 IMB=1
        # 成立）；门关不告警，IMB=1/缺省放行，非数字 IMB 交给 config 层去炸。
        warns: list[str] = []

        class _WarnRecorder:
            def warning(self, msg: str, *args: object, **kwargs: object) -> None:
                warns.append(msg)

        monkeypatch.setitem(npu_sparse_mla.__globals__, "log_rank0", _WarnRecorder())
        enabled = npu_sparse_mla.__globals__["_sfa_skip_recompute_enabled"]
        assert not enabled({"XTUNER_SFA_SKIP_RECOMPUTE": "1", "INTRA_LAYER_MICRO_BATCH": "2"})
        assert any("INTRA_LAYER_MICRO_BATCH" in w for w in warns)
        warns.clear()
        assert enabled({"XTUNER_SFA_SKIP_RECOMPUTE": "1", "INTRA_LAYER_MICRO_BATCH": "1"})
        assert enabled({"XTUNER_SFA_SKIP_RECOMPUTE": "1"})
        assert enabled({"XTUNER_SFA_SKIP_RECOMPUTE": "1", "INTRA_LAYER_MICRO_BATCH": "abc"})
        assert not enabled({"XTUNER_SFA_SKIP_RECOMPUTE": "0", "INTRA_LAYER_MICRO_BATCH": "2"})
        assert not warns

    @pytest.mark.gpu
    def test_prefetch_foreign_entry_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 卸载模式 lane 被外部塞入裸元组条目（设备驻留式 push 的形态）：prefetch
        # 必须不 raise 而返回 None（重放路径零可 raise 语句，assert 在 -O 下还会
        # 被剥掉）；pop 的元组分支照常返回其值，排干即修剪。
        monkeypatch.setattr(activation_offload_npu, "_SFA_STASH_OFFLOAD", True)
        q_ref = torch.empty(4, 2, 8, device=DEV)
        kv_ref = torch.empty(16, 1, 8, device=DEV)
        key = sfa_stash_key(q_ref, kv_ref)
        src = (
            torch.full((4, 2, 8), 3.0, device=DEV),
            torch.full((2, 4), 3.0, device=DEV),
            torch.full((2, 4), 3.0, device=DEV),
        )
        _sfa_stash.setdefault(key, deque()).append(src)
        assert sfa_stash_prefetch(key) is None
        outs = sfa_stash_pop(key, q_ref, kv_ref)
        assert outs is not None
        assert torch.equal(outs[0], src[0])
        assert torch.equal(outs[1], src[1])
        assert torch.equal(outs[2], src[2])
        assert key not in _sfa_stash

    @pytest.mark.gpu
    def test_pinned_pool_lru_family_cap(self) -> None:
        # 变长求值防护：形状族数超上限时按 LRU 淘汰最久未用族的空闲链（末引用
        # 落地，pinned 缓冲交还缓存分配器）；热族（刚释放）为 MRU，不被自身触发
        # 的淘汰命中；训练单形状族永远到不了淘汰路径。
        cap = activation_offload_npu._MAX_SFA_PINNED_FAMILIES
        keys: list[tuple[tuple[int, ...], torch.dtype]] = []

        def _release_family(i: int) -> None:
            buf = torch.empty(4 + i, 2, 8, dtype=torch.float32, pin_memory=True, device="cpu")
            activation_offload_npu._release_sfa_pinned(buf)
            keys.append((tuple(buf.shape), buf.dtype))

        for i in range(cap + 2):
            _release_family(i)
        assert len(_sfa_pinned_pool) == cap
        assert keys[0] not in _sfa_pinned_pool and keys[1] not in _sfa_pinned_pool
        assert keys[-1] in _sfa_pinned_pool
        # 热族重释放：自身成为 MRU，淘汰落在新的 LRU（第 3 族）上，热族保留。
        _release_family(0)
        assert keys[0] in _sfa_pinned_pool
        assert keys[2] not in _sfa_pinned_pool
        assert len(_sfa_pinned_pool) == cap

    @pytest.mark.gpu
    def test_pop_prunes_empty_lane(self) -> None:
        # 排干的 lane 即修剪：变长求值不再为每个出现过的形状残留空 deque；
        # 同形再 push 重建 lane 不受影响（守卫路径本就以空 lane 语义工作）。
        key = sfa_stash_key(torch.empty(4, 2, 8), torch.empty(16, 1, 8))
        sfa_stash_push(key, torch.zeros(4, 2, 8), torch.zeros(2, 4), torch.zeros(2, 4))
        assert key in _sfa_stash
        outs = sfa_stash_pop(key, torch.empty(4, 2, 8), torch.empty(16, 1, 8))
        assert outs is not None and torch.equal(outs[0], torch.zeros(4, 2, 8))
        assert key not in _sfa_stash
        sfa_stash_push(key, torch.zeros(4, 2, 8), torch.zeros(2, 4), torch.zeros(2, 4))
        assert key in _sfa_stash and len(_sfa_stash[key]) == 1
