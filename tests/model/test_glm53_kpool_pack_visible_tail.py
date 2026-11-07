"""GLM-5.3 kpool 行内可见 tail 打包（``XTUNER_GLM53_KPOOL_PACK_VISIBLE_TAIL``）回归测试。

打包门开启时 ``npu_kpool_topk_indices`` 输出行宽从 ``kpool_output_width`` 收缩到 ``index_topk``：行在
因果读窗 ``[0, min(qpos + 1, width))`` 内稠密无重复、``-1`` 只出现在读窗之外，尾 token 打包进行内
（短行 ``[0..p]`` + ``-1``；长行 = 所选池展开截去 t 槽 + 恰好 t 个以 query 结尾的文档尾）。消费端
``_rewrite_invalid_indices`` 门开时原样透传、``_global_to_local_indices`` 的 clamp 归一到 ``-1``。本文件
锁定：打包数学（合成输入单元测试）、生产者接线（门开/门关同输入的组合等价）、以及门约束。

TestPackVisibleTailUnit（CPU 或 NPU）
    test_short_rows_dense_then_neg1: 严格短行（``p + 1 < width``）= ``[0..p]`` + ``-1`` 补齐；边界行
        ``p + 1 == width`` 归长行（保持选择顺序），按集合断言；sel_all 垃圾值（越界）不被读取。
    test_long_rows_full_distinct_with_tail: 长行精确占满：前 ``width - t`` 列逐元素等于所选池展开的截断
        （保持选择顺序），后 t 列 = 以 query 结尾的文档尾（全局 id）；全行去重、因果。
    test_row_class_boundaries: 文档首 token（prefix=0，行 = [self] + -1）、t=0 行、p+1 == width 精确
        装满边界（prefix == select_k 且 t=0：行 = 所选池展开 == [0..p] 的集合）逐一锁定。
    test_kpool_one_empty_tail_guard: ``index_kpool == 1`` 时尾张量为 ``[S, 0]``，短行走 arange、长行
        整行 = 所选池（单 token 池）展开，打包不触碰尾分支。
TestPackVisibleTailProducer（NPU）
    test_pack_rows_match_unpacked_layout_composition: 同一输入门开/门关各跑一次（同打分路径）。长行：
        门开行逐元素等于门关布局的 ``out[:, :history_len]`` 拼 ``out[:, index_topk : index_topk + t]``
        （同选择、同顺序）。短行：门开历史段 = 规范序 ``arange(doc_start, doc_start + history_len)`` 且
        与门关历史段集合相等（topk 只保证集合），尾段逐元素相等。
    test_gate_on_row_invariants: 门开输出形状/dtype 契约 + 逐行性质：读窗内稠密非负去重、``-1`` 只在
        读窗外、长行精确占满。
    test_gate_on_requires_always_select_tail: 门开 + ``always_select_tail=False`` 立即 ValueError。
    test_gate_off_width_unchanged: 门关宽度契约不变（回归护栏）。
"""

import pytest
import torch
from torch import Tensor

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.sparse_mla import kpool, kpool_npu, npu_indexer
from xtuner.v1.utils.device import get_device


DEVICE = get_device()  # 本盒为 "npu"；纯 CPU 环境退化为 "cpu"
_requires_accelerator = pytest.mark.skipif(DEVICE == "cpu", reason="requires an NPU/CUDA accelerator")

if DEVICE == "npu":
    torch.npu.set_device("npu:0")


def _make_ctx(doc_lens: list[int], device: str | torch.device = DEVICE) -> SequenceContext:
    """构造单 rank（无 SP mesh）的 packed ``SequenceContext``。

    Args:
        doc_lens (list[int]): 各文档 token 数（packing 顺序即文档顺序）。
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
    num_heads: int = 8,
    head_dim: int = 128,
    index_kpool: int = 4,
    device: str | torch.device = DEVICE,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """按给定布局生成 ``npu_kpool_topk_indices`` 的全部输入。

    Args:
        doc_lens (list[int]): 各文档 token 数。
        seed (int): 随机种子。
        num_heads (int): indexer 头数 NH。
        head_dim (int): indexer 头维 D（也是 index_head_dim）。
        index_kpool (int): 池大小。
        device (str | torch.device): 张量设备。

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: ``(q, k, gate_scores, weights, kpool_ape)``，形状依次为
            ``[1, S, NH, D]`` / ``[1, S, D]`` / ``[1, S, D]`` / ``[1, S, NH]`` / ``[index_kpool, D]``。
    """
    torch.manual_seed(seed)
    total = sum(doc_lens)
    q = torch.randn(1, total, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, total, head_dim, dtype=torch.bfloat16, device=device)
    gate = torch.randn(1, total, head_dim, dtype=torch.bfloat16, device=device)
    weights = torch.randn(1, total, num_heads, dtype=torch.bfloat16, device=device)
    ape = torch.randn(index_kpool, head_dim, dtype=torch.bfloat16, device=device)
    return q, k, gate, weights, ape


def _row_layout(
    doc_lens: list[int],
    index_kpool: int,
    device: str | torch.device = DEVICE,
) -> tuple[Tensor, Tensor, Tensor, Tensor, list[tuple[int, int]]]:
    """从文档布局推导打包所需的逐行元数据（与 ``_expand_meta``/``_visible_tail_tokens`` 同数学）。

    Args:
        doc_lens (list[int]): 各文档 token 数。
        index_kpool (int): 池大小。
        device (str | torch.device): 张量设备。

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor, list[tuple[int, int]]]: ``(row_prefix, row_doc_start,
        slot_offsets, tail, local_pos)``——``row_prefix``/``row_doc_start`` 为 ``[S]`` int32，
        ``slot_offsets`` 为 ``[kpool]`` int32，``tail`` 为 ``[S, kpool - 1]`` int32（左对齐、``-1``
        垫，全局 token id），``local_pos`` 为每行 ``(doc, local_pos)`` 列表。
    """
    prefix, doc_start, local = [], [], []
    for d, ln in enumerate(doc_lens):
        start = sum(doc_lens[:d])
        for p in range(ln):
            prefix.append((p + 1) // index_kpool)
            doc_start.append(start)
            local.append((d, p))
    row_prefix = torch.tensor(prefix, dtype=torch.int32, device=device)
    row_doc_start = torch.tensor(doc_start, dtype=torch.int32, device=device)
    slot_offsets = torch.arange(index_kpool, dtype=torch.int32, device=device)

    token_ids = torch.tensor([s + p for s, (_, p) in zip(doc_start, local)], dtype=torch.int32, device=device)
    t = (token_ids.new_tensor([p + 1 for _, p in local]) % index_kpool).to(torch.int64)
    max_tail = index_kpool - 1
    if max_tail == 0:
        tail = torch.empty((len(local), 0), dtype=torch.int32, device=device)
    else:
        offsets = torch.arange(max_tail, device=device)
        tail_ids = (token_ids - t + 1).unsqueeze(-1) + offsets.unsqueeze(0)
        valid = offsets.unsqueeze(0) < t.unsqueeze(-1)
        tail = tail_ids.masked_fill(~valid, -1).to(torch.int32)
    return row_prefix, row_doc_start, slot_offsets, tail, local


def _synthetic_sel_all(
    row_prefix: Tensor,
    select_k: int,
    seed: int,
    device: str | torch.device = DEVICE,
) -> Tensor:
    """长行取 ``[0, prefix)`` 的随机无重采样（保持采样顺序），短行填越界垃圾值锁定"不被读取"。

    Args:
        row_prefix (Tensor): ``[S]`` int32 每行可见完整池数。
        select_k (int): 每行选择池数。
        seed (int): 随机种子。
        device (str | torch.device): 张量设备。

    Returns:
        Tensor: ``[S, select_k]`` int32 合成选择。
    """
    g = torch.Generator().manual_seed(seed)
    rows = []
    for prefix in row_prefix.tolist():
        if prefix >= select_k:
            rows.append(torch.randperm(prefix, generator=g)[:select_k])
        else:
            rows.append(torch.full((select_k,), prefix + 1000))
    return torch.stack(rows).to(torch.int32).to(device)


class TestPackVisibleTailUnit:
    """打包数学的合成输入单元测试（纯 torch，CPU 可跑）。"""

    def test_short_rows_dense_then_neg1(self) -> None:
        doc_lens, index_topk, index_kpool = [40], 32, 4
        row_prefix, row_doc_start, slot_offsets, tail, _ = _row_layout(doc_lens, index_kpool, device="cpu")
        select_k = index_topk // index_kpool
        sel_all = _synthetic_sel_all(row_prefix, select_k, seed=0, device="cpu")
        out = torch.full((sum(doc_lens), index_topk), -1, dtype=torch.int32)
        kpool_npu.pack_visible_tail_into(
            out,
            sel_all,
            row_prefix,
            row_doc_start,
            slot_offsets,
            tail,
            index_topk=index_topk,
            index_kpool=index_kpool,
        )
        for p in range(index_topk - 1):  # 严格短行: p + 1 < width（p=31 边界行归长行）
            expect = torch.full((index_topk,), -1, dtype=torch.int32)
            expect[: p + 1] = torch.arange(p + 1)
            assert torch.equal(out[p], expect), f"short row {p}"
        # 边界行 p + 1 == width: prefix == select_k 且 t = 0 -> 长行(选择顺序), 集合 == [0..p]
        assert torch.equal(out[index_topk - 1].sort().values, torch.arange(index_topk, dtype=torch.int32))
        # 短行与边界行的 sel_all 垃圾值（prefix + 1000）从未落地
        assert (out[:index_topk] < index_topk).all()

    def test_long_rows_full_distinct_with_tail(self) -> None:
        doc_lens, index_topk, index_kpool = [40, 200], 32, 4
        s = sum(doc_lens)
        row_prefix, row_doc_start, slot_offsets, tail, local = _row_layout(doc_lens, index_kpool, device="cpu")
        select_k = index_topk // index_kpool
        sel_all = _synthetic_sel_all(row_prefix, select_k, seed=1, device="cpu")
        out = torch.full((s, index_topk), -1, dtype=torch.int32)
        kpool_npu.pack_visible_tail_into(
            out,
            sel_all,
            row_prefix,
            row_doc_start,
            slot_offsets,
            tail,
            index_topk=index_topk,
            index_kpool=index_kpool,
        )
        for row, (d, p) in enumerate(local):
            prefix, t = row_prefix[row].item(), (p + 1) % index_kpool
            if prefix < select_k:
                continue
            # 长行: 精确占满, 前 width - t 列 = 所选池展开的截断(选择顺序), 后 t 列 = 文档尾(全局 id)
            doc_start = row_doc_start[row].item()
            expand = (sel_all[row].unsqueeze(-1) * index_kpool + doc_start + slot_offsets).view(-1)
            expect = torch.cat(
                [
                    expand[: index_topk - t],
                    torch.arange(doc_start + p - t + 1, doc_start + p + 1, dtype=torch.int32),
                ]
            )
            assert torch.equal(out[row], expect), f"long row {row} (doc {d} local {p})"
            assert out[row].unique().numel() == index_topk, f"duplicates in long row {row}"
            assert (out[row] >= doc_start).all() and (out[row] <= doc_start + p).all(), (
                f"non-causal slot in long row {row}"
            )

    def test_row_class_boundaries(self) -> None:
        # doc 边界首 token (prefix=0, t=1), t=0 行, p+1 == width 边界 (prefix == select_k, t=0)
        doc_lens, index_topk, index_kpool = [33, 5], 32, 4
        s = sum(doc_lens)
        row_prefix, row_doc_start, slot_offsets, tail, local = _row_layout(doc_lens, index_kpool, device="cpu")
        select_k = index_topk // index_kpool
        sel_all = _synthetic_sel_all(row_prefix, select_k, seed=2, device="cpu")
        out = torch.full((s, index_topk), -1, dtype=torch.int32)
        kpool_npu.pack_visible_tail_into(
            out,
            sel_all,
            row_prefix,
            row_doc_start,
            slot_offsets,
            tail,
            index_topk=index_topk,
            index_kpool=index_kpool,
        )
        for row, (d, p) in enumerate(local):
            doc_start = sum(doc_lens[:d])
            span = min(p + 1, index_topk)
            valid = out[row, :span]
            assert (valid >= 0).all() and (valid <= doc_start + p).all(), f"row {row}"
            assert valid.unique().numel() == valid.numel(), f"duplicates in row {row}"
            if span < index_topk:
                assert (out[row, span:] == -1).all(), f"-1 inside span in row {row}"
            if p == 0:  # 文档首 token: 行 = [self] + -1
                expect = torch.full((index_topk,), -1, dtype=torch.int32)
                expect[0] = doc_start
                assert torch.equal(out[row], expect), f"first-token row {row}"
            if p + 1 == index_topk:  # 边界: prefix == select_k 且 t = 0 -> 整行历史, 集合 == [0..p]
                assert torch.equal(out[row].sort().values, torch.arange(index_topk, dtype=torch.int32)), (
                    f"exact-fit boundary row {row}"
                )

    def test_kpool_one_empty_tail_guard(self) -> None:
        doc_lens, index_topk, index_kpool = [12], 8, 1
        row_prefix, row_doc_start, slot_offsets, tail, local = _row_layout(doc_lens, index_kpool, device="cpu")
        assert tail.shape[1] == 0
        select_k = index_topk // index_kpool
        sel_all = _synthetic_sel_all(row_prefix, select_k, seed=3, device="cpu")
        out = torch.full((sum(doc_lens), index_topk), -1, dtype=torch.int32)
        kpool_npu.pack_visible_tail_into(
            out,
            sel_all,
            row_prefix,
            row_doc_start,
            slot_offsets,
            tail,
            index_topk=index_topk,
            index_kpool=index_kpool,
        )
        for row, (d, p) in enumerate(local):
            if p + 1 < index_topk:  # 短行: [0..p] + -1（kpool=1 时池即单 token, 规范序 == arange）
                assert torch.equal(out[row, : p + 1], torch.arange(p + 1, dtype=torch.int32)), f"row {row}"
                assert (out[row, p + 1 :] == -1).all()
            else:  # 长行: 无尾, 整行 = 所选池展开, 保持选择顺序
                assert torch.equal(out[row], sel_all[row]), f"row {row}"


class TestPackVisibleTailProducer:
    """真实生产者接线：门开/门关同输入的组合等价与契约。"""

    doc_lens = [40, 200, 3, 65, 1, 129]
    index_topk = 32
    index_kpool = 4

    def _run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        pack: bool,
    ) -> tuple[Tensor, SequenceContext]:
        ctx = _make_ctx(self.doc_lens)
        inputs = _make_inputs(self.doc_lens, seed=7, index_kpool=self.index_kpool)
        monkeypatch.setattr(kpool_npu, "_BATCHED_SCORE", False)
        monkeypatch.setattr(kpool_npu, "_PACK_VISIBLE_TAIL", pack)
        q, k, gate, weights, ape = inputs
        out = kpool_npu.npu_kpool_topk_indices(
            q,
            k,
            gate,
            weights,
            ape,
            ctx,
            index_head_dim=q.shape[-1],
            index_topk=self.index_topk,
            index_kpool=self.index_kpool,
        )
        return out, ctx

    @_requires_accelerator
    def test_pack_rows_match_unpacked_layout_composition(self, monkeypatch: pytest.MonkeyPatch) -> None:
        packed, ctx = self._run(monkeypatch, pack=True)
        unpacked, _ = self._run(monkeypatch, pack=False)
        assert packed.shape == (sum(self.doc_lens), 1, self.index_topk)
        assert unpacked.shape == (
            sum(self.doc_lens),
            1,
            kpool.kpool_output_width(self.index_topk, self.index_kpool, 512),
        )

        flat_packed = packed.squeeze(1)
        flat_unpacked = unpacked.squeeze(1)
        select_k = self.index_topk // self.index_kpool
        cu = ctx.cu_seq_lens_q.tolist()
        for row in range(sum(self.doc_lens)):
            d = max(i for i in range(len(cu) - 1) if cu[i] <= row)
            p, doc_start = row - cu[d], cu[d]
            t = (p + 1) % self.index_kpool
            prefix = (p + 1) // self.index_kpool
            history_len = min(prefix * self.index_kpool, self.index_topk - t)
            tail_expect = flat_unpacked[row, self.index_topk : self.index_topk + t]
            # 尾段: 门开行内尾 == 门关尾列, 逐元素（对长短行都成立）
            assert torch.equal(flat_packed[row, history_len : history_len + t], tail_expect), (
                f"tail row {row} (doc {d} local {p})"
            )
            if prefix >= select_k:
                # 长行: 门开行 == 门关行的历史段前缀 + 尾段列（同选择、同顺序，逐元素）
                expect = torch.cat([flat_unpacked[row, :history_len], tail_expect])
                assert torch.equal(flat_packed[row], expect), f"long row {row} (doc {d} local {p})"
            else:
                # 短行: 门开历史段 = 规范序连续 arange; 与门关历史段集合相等（topk 只保证集合）
                hist = flat_packed[row, :history_len]
                expect = torch.arange(doc_start, doc_start + history_len, dtype=torch.int32, device=flat_packed.device)
                assert torch.equal(hist, expect), f"short hist row {row} (doc {d} local {p})"
                assert torch.equal(hist.sort().values, flat_unpacked[row, :history_len].sort().values), (
                    f"short hist set row {row} (doc {d} local {p})"
                )
                if history_len + t < self.index_topk:
                    assert (flat_packed[row, history_len + t :] == -1).all(), f"-1 inside span in short row {row}"

    @_requires_accelerator
    def test_gate_on_row_invariants(self, monkeypatch: pytest.MonkeyPatch) -> None:
        packed, _ = self._run(monkeypatch, pack=True)
        flat = packed.squeeze(1)
        assert packed.dtype == torch.int32
        cu = _make_ctx(self.doc_lens).cu_seq_lens_q.tolist()
        for row in range(sum(self.doc_lens)):
            d = max(i for i in range(len(cu) - 1) if cu[i] <= row)
            p, doc_start = row - cu[d], cu[d]
            span = min(p + 1, self.index_topk)
            valid = flat[row, :span]
            assert (valid >= doc_start).all() and (valid <= doc_start + p).all(), f"row {row}"
            assert valid.unique().numel() == valid.numel(), f"duplicates in row {row}"
            if span < self.index_topk:
                assert (flat[row, span:] == -1).all(), f"-1 inside span in row {row}"

    @_requires_accelerator
    def test_gate_on_requires_always_select_tail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(kpool_npu, "_PACK_VISIBLE_TAIL", True)
        ctx = _make_ctx([8])
        q, k, gate, weights, ape = _make_inputs([8], seed=11)
        with pytest.raises(ValueError, match="always_select_tail"):
            kpool_npu.npu_kpool_topk_indices(
                q,
                k,
                gate,
                weights,
                ape,
                ctx,
                index_head_dim=q.shape[-1],
                index_topk=32,
                index_kpool=4,
                always_select_tail=False,
            )

    def test_glm52_dsa_producer_rejects_pack_gate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # GLM-5.2 生产者在读跨度内部任意位置填 -1, 门开直通会被 backward 读到; 守卫必须
        # 在任何设备操作之前触发, 因此本测试无需加速器。
        monkeypatch.setattr(kpool_npu, "_PACK_VISIBLE_TAIL", True)
        ctx = _make_ctx([4], device="cpu")
        q = torch.zeros(1, 4, 2, 128, dtype=torch.bfloat16)
        k = torch.zeros(1, 4, 128, dtype=torch.bfloat16)
        weights = torch.zeros(1, 4, 2, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="PACK_VISIBLE_TAIL"):
            npu_indexer.npu_dsa_topk_indices(
                q, k, weights, ctx, index_head_dim=q.shape[-1], index_topk=2
            )

    @_requires_accelerator
    def test_gate_off_width_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        unpacked, _ = self._run(monkeypatch, pack=False)
        assert unpacked.shape[-1] == kpool.kpool_output_width(self.index_topk, self.index_kpool, 512)
        assert unpacked.dtype == torch.int32
