# Copyright (c) OpenMMLab. All rights reserved.
"""Regression tests for the masked-load-OOB clamps in the KDA triton kernels.

Masked MTE loads on this backend can be lowered to unmasked issues, so a
masked-off lane's address still reaches the memory system -- the class that
faulted the conv-bwd kernel with 507035 vector-core aborts. Three sibling
kernels carried the same pattern:

- ``_chunk_cumsum_kernel``: the tail chunk's tokens overshoot ``stop`` by up
  to BT-1 rows and the last feature block overshoots ``K`` by up to BK-1
  columns; for the final document those offsets leave the tensor;
- ``_seg_copy_rows_kernel``: pad rows load with ``in_seg`` false, and for the
  last segment ``start + rows`` runs past the source tensor;
- ``_seg_copy_rows_gather_kernel``: the tail tile's rows overshoot the index
  table and pad rows carry the sentinel source row ``n_src``, exactly one
  row past the source tensor.

No load address may leave its tensor. The seg-copy kernels keep per-lane
``tl.minimum`` clamps on every load index (masked-off lanes then read
in-bounds values the mask discards; active lanes are unchanged) while stores
keep the raw index. ``_chunk_cumsum_kernel`` instead moves its tile through
``make_block_ptr`` + ``boundary_check`` over the (stop, K) window -- the
same valid mask without the per-lane clamp. The tests pin both mechanisms in
source and the identical outputs on overshoot shapes against independent
references.
"""

import inspect

import pytest
import torch
import torch.nn.functional as F

from xtuner.v1.ops.kda import chunk_cumsum as chunk_cumsum_mod
from xtuner.v1.ops.kda import kda_op
from xtuner.v1.ops.kda.chunk_cumsum import chunk_cumsum
from xtuner.v1.ops.kda.kda_op import _padded_pack_maps, _repack_padded


pytestmark = pytest.mark.gpu


def _flat(source: str) -> str:
    return " ".join(source.split())


def _chunk_indices(cu: list[int]) -> torch.Tensor:
    pairs: list[tuple[int, int]] = []
    for doc, (start, end) in enumerate(zip(cu[:-1], cu[1:])):
        length = end - start
        pairs.extend((doc, i) for i in range((length + 63) // 64))
    return torch.tensor(pairs, dtype=torch.int32, device="cuda")


def _cumsum_reference(g: torch.Tensor, cu: list[int] | None, scale: float) -> torch.Tensor:
    """Per-chunk-local fp32 prefix sums, reset at each chunk/document start."""
    out = torch.zeros_like(g, dtype=torch.float32)
    gf = g.float()
    if cu is None:
        bounds = [(b * g.shape[1], (b + 1) * g.shape[1]) for b in range(g.shape[0])]
    else:
        bounds = list(zip(cu[:-1], cu[1:]))
    for start, end in bounds:
        for c0 in range(start, end, 64):
            c1 = min(c0 + 64, end)
            out[:, c0:c1] = torch.cumsum(gf[:, c0:c1], dim=1) * scale
    return out


class TestChunkCumsumOobClamp:
    def test_block_ptr_bounds_in_kernel_source(self):
        source = _flat(inspect.getsource(chunk_cumsum_mod._chunk_cumsum_kernel.fn))
        assert "tl.make_block_ptr" in source
        assert "boundary_check=(0, 1)" in source
        assert 'padding_option="zero"' in source
        assert "tl.minimum(token" not in source

    @pytest.mark.parametrize(
        "cu,heads,features",
        [
            ([0, 100, 355, 512], 4, 96),  # lengths not 64-aligned: token overshoots stop
            ([0, 100, 355, 512], 4, 80),  # 96 -> 80: last feature block overshoots K by 16
            ([0, 64, 128, 192], 2, 32),  # every boundary 64/K aligned
            (None, 2, 96),  # non-varlen B=2, T=100 overshoots the last chunk
        ],
    )
    def test_varlen_overshoot_matches_reference(self, cu: list[int] | None, heads: int, features: int):
        torch.manual_seed(0)
        scale = 0.37
        length = 100 if cu is None else cu[-1]
        g = (
            torch.randn(2, length, heads, features, dtype=torch.bfloat16, device="cuda")
            if cu is None
            else torch.randn(1, length, heads, features, dtype=torch.bfloat16, device="cuda")
        )
        cu_seqlens = None if cu is None else torch.tensor(cu, dtype=torch.int32, device="cuda")
        chunk_indices = None if cu is None else _chunk_indices(cu)
        out = chunk_cumsum(g, cu_seqlens, chunk_indices, scale)
        assert out.shape == g.shape and out.dtype == torch.float32
        ref = _cumsum_reference(g, cu, scale)
        assert torch.allclose(out, ref, rtol=1e-4, atol=1e-4)


class TestSegCopyOobClamp:
    def test_clamps_present_in_kernel_source(self):
        scatter = _flat(inspect.getsource(kda_op._seg_copy_rows_kernel.fn))
        assert "tl.minimum(rows_t, rem - 1)" in scatter
        assert "tl.load(src, mask=in_seg" in scatter
        assert "tl.store(dst, vals, mask=(rows_t < rem + pad_len)" in scatter
        gather = _flat(inspect.getsource(kda_op._seg_copy_rows_gather_kernel.fn))
        assert "tl.minimum(rows, n_dst - 1)" not in gather  # idx table is host-padded
        assert "tl.minimum(src_idx, n_src - 1)" in gather  # sentinel row clamp stays
        assert "tl.load(idx_ptr + rows, mask=in_dst" in gather
        assert "tl.load(src_ptr + src_idx_ld" in gather
        assert "tl.store(dst_ptr + rows" in gather

    @pytest.mark.parametrize(
        "cu",
        [
            [0, 100, 355, 512],  # lengths 100/255/157: real pad tails on every segment
            [0, 100, 100, 355, 512],  # zero-length segment is dropped
        ],
    )
    @pytest.mark.parametrize(
        "shape",
        [
            (2, 128),  # wide scatter, row_elems=256, one D_BLOCK
            (4, 128),  # wide scatter, row_elems=512, two D_BLOCKs
            (1, 64),  # narrow gather, row_elems=64
            (1, 1),  # narrow gather, row_elems=1 (W=1 tile)
        ],
    )
    def test_repack_paths_match_padded_reference(self, cu: list[int], shape: tuple[int, int]):
        torch.manual_seed(0)
        device = torch.device("cuda")
        cu_padded, _real_pos, gather_pos, cu_t, pad_cu_t, max_padded_len = _padded_pack_maps(cu, device)
        tensor = torch.randn(1, cu[-1], *shape, dtype=torch.bfloat16, device=device)
        out = _repack_padded(tensor, gather_pos, cu_t, pad_cu_t, cu_padded[-1], max_padded_len)
        assert out.shape[1] == cu_padded[-1]
        pad_spec = (0, 0, 0, 1) if tensor.dim() == 3 else (0, 0, 0, 0, 0, 1)
        ref = F.pad(tensor, pad_spec).index_select(1, gather_pos[: cu_padded[-1]])
        assert torch.equal(out, ref)

    @pytest.mark.parametrize(
        "label,cu,shape,dtype",
        [
            # fp32 halves the tile rows (ROWS=32): seg_len=96 gives rem=0 with a
            # pad tail, so the tail clamp hits its rem-1 = -1 branch
            ("fp32 ROWS=32 rem=0 pad=32", [0, 96, 200, 300], (2, 256), torch.float32),
            # bf16 W=512 also compiles ROWS=32 tiles over the same geometry
            ("bf16 W512 ROWS=32", [0, 96, 200, 300], (4, 128), torch.bfloat16),
            # a pack whose padded length crosses the table's row quantum
            ("n_padded>1024 20seg", list(range(0, 2001, 100)), (2, 128), torch.bfloat16),
        ],
    )
    def test_repack_extended_boundaries(self, label: str, cu: list[int], shape: tuple[int, int], dtype: torch.dtype):
        torch.manual_seed(0)
        device = torch.device("cuda")
        cu_padded, _real_pos, gather_pos, cu_t, pad_cu_t, max_padded_len = _padded_pack_maps(cu, device)
        tensor = torch.randn(1, cu[-1], *shape, dtype=dtype, device=device)
        out = _repack_padded(tensor, gather_pos, cu_t, pad_cu_t, cu_padded[-1], max_padded_len)
        assert out.shape[1] == cu_padded[-1]
        pad_spec = (0, 0, 0, 1) if tensor.dim() == 3 else (0, 0, 0, 0, 0, 1)
        ref = F.pad(tensor, pad_spec).index_select(1, gather_pos[: cu_padded[-1]])
        assert torch.equal(out, ref)
