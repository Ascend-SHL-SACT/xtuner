"""Int8 W8A8 dynamic quantization modules for dense and grouped linear layers.

Both modules mirror their parent classes' forward semantics (DTensor unwrapping, bias
handling, grouped-GEMM call signature) and insert the deployed ``W8A8_DYNAMIC`` grids on
top: activations are fake-quantized per token (dynamic int8, straight-through estimator)
and weights per output channel (int8 levels implicit in the fake-quantized value). With
the gate off the factories build the original classes, so these modules only ever run in
quantized training.

The grouped variant optionally runs the real aclnn op (MindSpeed ``npu_quant_gmm``) as a
dual-path forward: ``real.detach() + (fake - fake.detach())``. The correction term is
exactly zero, so the value matches the real quantized op bit-for-bit unconditionally,
while the backward graph stays bit-identical to the fake engine, because only the fake
term carries gradient and the real arm runs under ``torch.no_grad()``.
"""

from collections.abc import Callable
from typing import Literal

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.nn import functional as F

from xtuner.v1.module.grouped_linear.moe_group_linear import GroupedLinear, GroupedLinearParallelStyle
from xtuner.v1.module.linear.linear import _Linear
from xtuner.v1.ops import group_gemm

from .kernels import (
    per_channel_weight_fake_quant,
    per_channel_weight_quant,
    per_token_act_fake_quant,
    per_token_act_quant,
)


class Int8W8A8DynamicLinear(_Linear):
    """Dense linear with W8A8 dynamic fake quantization.

    Extends :class:`~xtuner.v1.module.linear.linear._Linear` (same ``nn.Linear`` parameter
    layout, same DTensor unwrapping) by fake-quantizing the input per token and the weight
    per output channel before the GEMM. Both fake quantizers pass gradients through
    unchanged (STE), so the master weights adapt to the deployed int8 grids while the
    forward stays in bf16. The fused a2a+gmm path may hand the shared experts' gate/up an
    already fake-quantized input (the same tensor feeds the routed experts) and scope the
    skip of the internal per-token quant around that forward via ``_input_prequantized``.

    Args:
        in_features (int): Input feature dimension.
        out_features (int): Output feature dimension.
        bias (bool): Whether the layer has an additive bias.
        device (torch.device | None): Parameter device.
        dtype (torch.dtype | None): Parameter dtype.
    """

    # Set (and always reset) by ``fused_a2a_gmm.fused_forward`` around a shared-experts
    # forward whose input was already fake-quantized on this grid: the STE node then sits
    # upstream, so the internal per-token quant would only recompute identical values. A
    # class-level default keeps the attribute readable without touching ``__init__``.
    _input_prequantized: bool = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(self.weight, DTensor):
            w = self.weight.to_local()
            if self.bias is not None:
                assert isinstance(self.bias, DTensor), "Bias should be a DTensor if weight is a DTensor"
                b = self.bias.to_local()
            else:
                b = None
        else:
            w = self.weight
            b = self.bias
        if self._input_prequantized:
            xq = input
        else:
            xq = per_token_act_fake_quant(input)
        return F.linear(xq, per_channel_weight_fake_quant(w), b)


class Int8W8A8DynamicGroupedLinear(GroupedLinear):
    """Grouped linear with W8A8 dynamic fake quantization for routed experts.

    Extends :class:`~xtuner.v1.module.grouped_linear.moe_group_linear.GroupedLinear` (same
    weight layout, same DTensor placements, same bias tail) by fake-quantizing the tokens
    per token and each local expert's weight per output channel before the grouped GEMM.
    The per-channel scale is computed over the full input axis, so row-parallel expert TP
    (which shards ``in_features``) would produce a rank-local grid that matches neither the
    export nor the deployed one; that layout fails fast at construction.

    Args:
        in_features (int): Input feature dimension.
        out_features (int): Output feature dimension.
        num_routed_experts (int): Global number of routed experts.
        moe_bias (bool): Whether each expert has an additive bias.
        ep_mesh (DeviceMesh | None): Expert-parallel mesh.
        expert_tp_mesh (DeviceMesh | None): Expert tensor-parallel mesh.
        parallel_style (GroupedLinearParallelStyle | None): Column/row parallel style.
        ep_tp_mesh (DeviceMesh | None): 2D (ep, tp) sub-mesh for DTensor weights.
        num_fused_projections (int): Number of projections fused per expert (2 for
            fused_w1w3).
        gmm_engine (Literal["fake", "npu_quant_gmm"]): Grouped-GEMM engine. ``"fake"``
            computes the quantized grouped GEMM with the native ``group_gemm`` (autograd
            flows through the STE graph). ``"npu_quant_gmm"`` runs the real aclnn op
            (MindSpeed ``npu_quant_gmm``) as a dual-path forward: the value comes from the
            real op, the backward graph from the fake path.

    Raises:
        NotImplementedError: If expert TP is enabled with row parallelism — the per-channel
            weight scale needs the full input axis (see class docstring).
        ImportError: If ``gmm_engine="npu_quant_gmm"`` but MindSpeed is not importable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_routed_experts: int,
        moe_bias: bool = False,
        ep_mesh: DeviceMesh | None = None,
        expert_tp_mesh: DeviceMesh | None = None,
        parallel_style: GroupedLinearParallelStyle | None = None,
        ep_tp_mesh: DeviceMesh | None = None,
        num_fused_projections: int = 1,
        gmm_engine: Literal["fake", "npu_quant_gmm"] = "fake",
    ):
        super().__init__(
            in_features,
            out_features,
            num_routed_experts,
            moe_bias=moe_bias,
            ep_mesh=ep_mesh,
            expert_tp_mesh=expert_tp_mesh,
            parallel_style=parallel_style,
            ep_tp_mesh=ep_tp_mesh,
            num_fused_projections=num_fused_projections,
        )
        if self.tp_enabled and self.parallel_style == "row":
            raise NotImplementedError(
                "Int8 W8A8 dynamic quantization computes the per-output-channel weight scale over the "
                "full input axis, which row-parallel expert TP shards across ranks; the local scale "
                "would match neither the exported nor the deployed grid. Run experts with EP only, or "
                "set XTUNER_QUANT_LINEAR off."
            )
        self.gmm_engine: Literal["fake", "npu_quant_gmm"] = gmm_engine
        self._quant_gmm: Callable[..., torch.Tensor] | None = None
        if gmm_engine == "npu_quant_gmm":
            try:
                from mindspeed.ops.quant_gmm import npu_quant_gmm
            except ImportError as exc:
                raise ImportError(
                    "XTUNER_QUANT_LINEAR_GMM='npu_quant_gmm' requires the MindSpeed npu_quant_gmm op; "
                    "install MindSpeed or set XTUNER_QUANT_LINEAR_GMM=fake."
                ) from exc
            self._quant_gmm = npu_quant_gmm

    def forward(self, x: torch.Tensor, tokens_per_expert: torch.Tensor, decoding: bool = False) -> torch.Tensor:
        weight = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        weight = weight.view(-1, self.local_out_features, self.local_in_features)
        wq = per_channel_weight_fake_quant(weight)
        xq = per_token_act_fake_quant(x)
        out = group_gemm(xq, wq, tokens_per_expert)
        if self._quant_gmm is not None:
            out = self._dual_path_gmm(x, weight, tokens_per_expert, out)

        if self.moe_bias:
            bias = self.bias.to_local() if isinstance(self.bias, DTensor) else self.bias
            bias = bias.view(self.local_num_routed_experts, -1)
            if self.tp_enabled and self.parallel_style == "row" and self.tp_rank != 0:
                return out
            out = out + bias.repeat_interleave(tokens_per_expert, dim=0)  # TODO: 无法 compile
        return out

    def fused_weight_fake_quant(self, weight: torch.Tensor) -> torch.Tensor:
        """Fake-quantize a native-layout weight for the fused a2a+gmm path.

        Called by ``fused_a2a_gmm.fused_forward`` on the FSDP2-local native weight view: the
        returned straight-through node becomes the fused op's ``weight`` argument, so the
        forward consumes the quantized grid and the hand-written backward's ``weight_grad``
        flows through the STE to the master weight (the exact module-path math).

        Args:
            weight (torch.Tensor): native grouped weight ``[E_local, out, in]``.

        Returns:
            torch.Tensor: the per-output-channel fake-quantized weight, same layout.
        """
        if self.gmm_engine == "npu_quant_gmm":
            raise ValueError(
                "XTUNER_QUANT_LINEAR_GMM='npu_quant_gmm' runs the real aclnn quant op in the "
                "separated module path only; the fused a2a+gmm path has no real-op arm. Set "
                "XTUNER_MOE_FUSED_A2A_GMM=0."
            )
        return per_channel_weight_fake_quant(weight)

    def fused_act_fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        """Fake-quantize a fused-path activation per token (STE).

        Applied by ``fused_a2a_gmm.fused_dispatch_mlp_combine`` to the FC1 input (pre-dispatch;
        per-token scales are row-local, so this equals the module path's post-dispatch quant)
        and the FC2 input. All-zero rows -- the chunked path's structural pad -- stay exact
        zeros on every grid (the eager chain floors the scale, the NPU kernel zeroes it).

        Args:
            x (torch.Tensor): activation rows ``[..., in_features]``.

        Returns:
            torch.Tensor: the per-token fake-quantized activation, same shape and dtype.
        """
        return per_token_act_fake_quant(x)

    def _dual_path_gmm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        fake_out: torch.Tensor,
    ) -> torch.Tensor:
        # Value from the real aclnn op, backward graph from the fake path. The real arm
        # quantizes on the same grids as the fake arm (per-token amax/127 for activations,
        # per-output-channel amax/127 for weights), with the deployment-mirror dtype combo
        # probed on 910C: bf16 weight scales + fp32 per-token scale + bf16 output.
        if x.shape[0] == 0:
            # Shape-only check (no device sync); the real op is pointless on an empty batch
            # and an all-zero group list is an unprobed aclnn corner.
            return fake_out
        quant_gmm = self._quant_gmm
        assert quant_gmm is not None  # only reachable when gmm_engine="npu_quant_gmm" (forward gate)
        group_list = tokens_per_expert.cumsum(0).to(torch.int64)
        with torch.no_grad():
            qx, sx = per_token_act_quant(x)
            qw, sw = per_channel_weight_quant(weight)
            real = quant_gmm(
                qx,
                qw.transpose(1, 2).contiguous(),
                sw.squeeze(-1).to(torch.bfloat16),
                offset=None,
                per_token_scale=sx,
                bias=None,
                group_list=group_list,
                output_dtype=torch.bfloat16,
            )
        return real.detach() + (fake_out - fake_out.detach())


__all__ = [
    "Int8W8A8DynamicGroupedLinear",
    "Int8W8A8DynamicLinear",
]
