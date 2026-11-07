"""Minimal verbatim excerpts of the vllm-ascend SFA C8 dynamic-block quantization contract.

Pinned: vllm-ascend@9300a2f9bfee2fec19c0f7fa2634490d77cde9de (vllm_ascend/attention/utils.py,
vllm_ascend/attention/sfa_v1.py). Serving GLM-5.3-Flash with ``--kv-cache-dtype int8`` stores
the RMSNorm'd ckv latent through ``custom_kv_rmsnorm_rope``: the latent's ``kv_lora_rank``
channels are quantized by ``torch_npu.npu_dynamic_block_quant`` with ``row_block_size=1`` /
``col_block_size=SFA_QSFA_TILE_SIZE`` -- one symmetric-int8 grid with one fp32 scale per
(1, 128) token tile -- and the fp32 scales are byte-transported (``view(torch.int8)``) in the
cache row next to the int8 values. The training kernel ``int8_fake_quant_dynamic_block`` must
sit on exactly that grid; the tests assert our kernel against these consumer excerpts.

On-device probe record (claude_added/probe_npu_dynamic_block_20261106.py, A3): the op derives
``scale = amax * (1/127)`` in fp32 (bitwise; NOT ``amax / 127``), emits int8 levels in
[-127, 127] with ``round_mode="rint"``, and yields ``scale == 0`` / ``y == 0`` for an all-zero
tile. Dequantization applies ``y * scale`` per tile (probe round-trip error bounded by
``scale / 2``). The op's quantize step is CANN-internal and not reproducible bitwise in torch:
on random data ~0.03% of levels sit on a rounding knife-edge and differ by exactly one level,
which is immaterial against the ``scale / 2`` grid error; scales match bitwise.

Deviations from the source (mirroring ``_modelslim_consumer_ref.py`` conventions):
- ``import torch_npu`` moved into the function body: the module must import on hosts without
  the NPU stack; the op only runs on-device, so the parity test is device-marked and skips
  elsewhere.
- Only the int8 branch of ``custom_kv_rmsnorm_rope`` is copied (the A3/GLM-5.3-Flash path);
  the fp8/A5 branch and the RoPE half are out of scope for 910C and untouched by this kernel;
  consequently the source's 3-tuple return (int8 values, scales, rope'd k_pe) is narrowed to
  the 2-tuple below.
- Two ``npu_dynamic_block_quant`` arguments are inlined: the source passes
  ``dst_type=dst_type`` and ``col_block_size=tile_size`` (injected by the caller as
  ``self.c8_cache_dtype`` / ``self.sfa_qsfa_tile_size``); this excerpt hardcodes
  ``torch.int8`` / ``SFA_QSFA_TILE_SIZE`` -- exactly the values the GLM-5.3-Flash A3 path
  supplies, so the call is semantically identical on that path.
"""

import torch


# Verbatim: vllm_ascend/attention/utils.py:45 (the deployment kernel's only supported
# column block size).
SFA_QSFA_TILE_SIZE = 128


def custom_kv_rmsnorm_rope_quant_int8(k_nope: torch.Tensor, kv_lora_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the ckv latent exactly as the deployed SFA C8 path does (verbatim excerpt).

    Excerpt of ``vllm_ascend/attention/sfa_v1.py::custom_kv_rmsnorm_rope`` (int8 branch): the
    source takes the post-RMSNorm ``k_nope`` and returns int8 values plus fp32 scales whose
    bytes the caller packs into the cache row (the ``view(torch.int8)`` transport below is the
    consumer's own code).

    Args:
        k_nope (torch.Tensor): Post-RMSNorm latent, shape ``[..., kv_lora_rank]``.
        kv_lora_rank (int): The latent channel count (512 for GLM-5.3-Flash).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(k_nope int8 view [..., kv_lora_rank], scales
        byte-transported as int8 with fp32 payload, shape [..., kv_lora_rank // 128])``.
    """
    import torch_npu  # lazy: the op only exists on the NPU stack

    prefix_shape = k_nope.shape[:-1]
    k_nope_q, knope_scale = torch_npu.npu_dynamic_block_quant(
        k_nope.view(-1, 1, kv_lora_rank),
        dst_type=torch.int8,
        row_block_size=1,
        col_block_size=SFA_QSFA_TILE_SIZE,
    )
    # Return byte views so the caller can concatenate all three components.
    return (
        k_nope_q.view(*prefix_shape, kv_lora_rank),
        knope_scale.to(torch.float32).view(*prefix_shape, -1).contiguous().view(torch.int8),
    )
