import contextlib
import os
from functools import partial
from typing import Literal, Protocol, TypeAlias, cast, Dict

import torch
import torch.nn as nn
import torch.utils.checkpoint
from pydantic import BaseModel, ConfigDict
from torch.autograd.function import Function
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.nn import functional as F

from xtuner.v1.config.generate import GenerateConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8 import Float8Config
from xtuner.v1.module import (
    AttnOutputs,
    GatedDeltaNet,
    GatedDeltaNetConfig,
    GreedyRouterConfig,
    MHAConfig,
    MLAConfig,
    MultiHeadAttention,
    MultiLatentAttention,
    NoAuxRouterConfig,
    RMSNorm,
    RouterResults,
)
from xtuner.v1.module.dispatcher import (
    CombineResult,
    DispatchResult,
    PostDispatchResult,
    PreCombineResult,
    PreDispatchResult,
    build_dispatcher,
)
from xtuner.v1.module.grouped_linear.moe_group_linear import build_grouped_linear
from xtuner.v1.module.rope import RopeScalingConfig
from xtuner.v1.ops.act_fn import get_act_fn
from xtuner.v1.utils import ForwardState
from xtuner.v1.utils.activation_offload import async_save_on_cpu

from ..linear import build_linear


RouterLogits: TypeAlias = torch.Tensor
RouterWeights: TypeAlias = torch.Tensor
HiddenStates: TypeAlias = torch.Tensor


class MoEActFnProtocol(Protocol):
    def __call__(self, fused_x: torch.Tensor, split_dim: int = -1) -> torch.Tensor: ...


class MoEActFnConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    act_type: Literal["clipped_swiglu", "swiglu"] = "swiglu"

    clip_alpha: float | None = None
    clip_limit: float | None = None

    def build(self) -> MoEActFnProtocol:
        act_fn = get_act_fn(self.act_type)

        if self.act_type == "clipped_swiglu":
            act_fn = partial(act_fn, alpha=self.clip_alpha, limit=self.clip_limit)
        return act_fn


class MoEMLP(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        n_shared_experts: int,
        moe_intermediate_size: int,
        hidden_act: str,
        mlp_bias: bool = False,
        float8_cfg: Float8Config | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = moe_intermediate_size * n_shared_experts
        self.gate_proj = build_linear(self.hidden_size, self.intermediate_size, bias=mlp_bias, float8_cfg=float8_cfg)
        self.up_proj = build_linear(self.hidden_size, self.intermediate_size, bias=mlp_bias, float8_cfg=float8_cfg)
        self.down_proj = build_linear(self.intermediate_size, self.hidden_size, bias=mlp_bias, float8_cfg=float8_cfg)
        self.act_fn = get_act_fn(hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MoEGate(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        n_routed_experts: int,
        num_experts_per_tok: int,
        router_config: GreedyRouterConfig | NoAuxRouterConfig,
        gate_bias: bool = False,
        router_compute_dtype: Literal["float32", "native"] = "float32",
    ):
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.router_compute_dtype = router_compute_dtype

        self.gating_dim = hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))

        self.router = router_config.build(
            n_routed_experts=self.n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
        )

        self.gate_bias = gate_bias
        if self.gate_bias:
            self.bias = nn.Parameter(torch.zeros(self.n_routed_experts))

    def forward(
        self, hidden_states: torch.Tensor, rollout_routed_experts: torch.Tensor | None = None
    ) -> RouterResults:
        _, _, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)

        if isinstance(self.weight, DTensor):
            weight = self.weight.to_local()
        else:
            weight = self.weight

        bias = None
        if self.gate_bias:
            bias = self.bias.to_local() if isinstance(self.bias, DTensor) else self.bias

        if self.router_compute_dtype == "native":
            logits = F.linear(hidden_states, weight, bias)
        else:
            bias = bias.float() if bias is not None else None
            logits = F.linear(hidden_states.float(), weight.float(), bias)
        return self.router(logits, rollout_routed_experts)

        # Debug for aligning with hf implementation.
        # logits = F.linear(hidden_states, weight, bias)
        # gate = self.router(logits, rollout_routed_experts)
        # gate['topk_weights'] = gate['topk_weights'].float()
        # return gate


class MoEBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        moe_intermediate_size: int,
        n_routed_experts: int,
        moe_bias: bool = False,
        ep_mesh: DeviceMesh | None = None,
        float8_cfg: Float8Config | None = None,
        moe_act_fn_cfg: MoEActFnConfig,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = moe_intermediate_size
        self.num_routed_experts = n_routed_experts
        self.ep_mesh = ep_mesh
        # self.fused_w1 = GroupedLinear(self.hidden_size, self.intermediate_size, self.num_routed_experts, ep_mesh)
        # self.fused_w3 = GroupedLinear(self.hidden_size, self.intermediate_size, self.num_routed_experts, ep_mesh)
        self.fused_w1w3 = build_grouped_linear(
            self.hidden_size,
            2 * self.intermediate_size,
            self.num_routed_experts,
            moe_bias=moe_bias,
            ep_mesh=self.ep_mesh,
            float8_cfg=float8_cfg,
        )
        self.fused_w2 = build_grouped_linear(
            self.intermediate_size,
            self.hidden_size,
            self.num_routed_experts,
            moe_bias=moe_bias,
            ep_mesh=self.ep_mesh,
            float8_cfg=float8_cfg,
        )
        self.moe_act = moe_act_fn_cfg.build()

    def forward(self, x, tokens_per_expert, decoding):
        # short cut for dispatching 0 token in ep_size >1 case
        if x.numel() == 0:
            return x

        gate_up_out = self.fused_w1w3(x, tokens_per_expert, decoding)
        out = self.moe_act(gate_up_out, split_dim=-1)
        res = self.fused_w2(out, tokens_per_expert, decoding)
        return res


class MoEAttention:
    """Attention computation of the MoE decoder layer.

    Encapsulates the self-attention + post-attention layernorm forward so it
    can be recomputed independently from the MoE expert computation (e.g.
    wrapped with ``torch.utils.checkpoint``). The attention submodules
    (``self_attn`` / ``input_layernorm`` / ``post_attention_layernorm``) are
    owned and registered by the parent :class:`MoEDecoderLayer` to keep
    parameter names -- and therefore HF weight loading -- unchanged; this
    class only holds references to them.
    """

    def __init__(
        self,
        self_attn: MultiHeadAttention | MultiLatentAttention | GatedDeltaNet,
        input_layernorm: RMSNorm,
        post_attention_layernorm: RMSNorm,
    ):
        self.self_attn = self_attn
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        state: ForwardState,
        past_key_values: list[list[torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE: In order to allow `torch.compile` to compile the ops before and after attention as much as possible,
        # attention and post-layernorm are implemented in one function.
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        if state == ForwardState.TRAINING:
            attn_outputs: AttnOutputs = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
            )
            hidden_states = attn_outputs["projected_output"]
        elif state == ForwardState.PREFILLING:
            assert past_key_values is not None, "past_key_values should be provided in pre-filling state"
            hidden_states = self.self_attn.prefilling(  # type: ignore
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
                past_key_values=past_key_values,
            )
        elif state == ForwardState.DECODING:
            assert past_key_values is not None, "past_key_values should be provided in decoding state"
            hidden_states = self.self_attn.decoding(  # type: ignore
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
                past_key_values=past_key_values,
            )
        hidden_states = residual + hidden_states

        # Fully Connected (post-attention layernorm)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual, hidden_states

    def __call__(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(*args, **kwargs)


class MoEDecoderLayer(nn.Module):
    """MoE decoder layer."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        moe_intermediate_size: int,
        mlp_bias: bool = False,
        gate_bias: bool = False,
        moe_bias: bool = False,
        hidden_act: str,
        rms_norm_eps: float = 1e-6,
        rms_norm_type: Literal["default", "zero_centered"] = "default",
        num_experts_per_tok: int,
        n_routed_experts: int,
        n_shared_experts: int,
        with_shared_expert_gate: bool = False,
        hidden_factor: float = 1.0,
        attention_config: MHAConfig | MLAConfig | GatedDeltaNetConfig,
        rope_scaling_cfg: RopeScalingConfig | None = None,
        layer_type: Literal["full_attention", "sliding_attention"] | None = None,
        generate_config: GenerateConfig | None = None,
        router_config: GreedyRouterConfig | NoAuxRouterConfig,
        router_compute_dtype: Literal["float32", "native"] = "float32",
        moe_act_fn_cfg: MoEActFnConfig,
        float8_cfg: Float8Config | None = None,
        layer_idx: int = 0,
        dispatcher: Literal["deepep", "all2all", "agrs"] | None,
        ep_mesh: DeviceMesh | None = None,
    ):
        super().__init__()
        self.ep_mesh = ep_mesh
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.hidden_factor = hidden_factor

        self.self_attn: MultiHeadAttention | MultiLatentAttention | GatedDeltaNet = attention_config.build(
            hidden_size=hidden_size,
            layer_idx=layer_idx,
            generate_config=generate_config,
            rope_scaling_cfg=rope_scaling_cfg,
            layer_type=layer_type,
            float8_cfg=float8_cfg,
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)
        self.layer_idx = layer_idx
        # When True, attention and MoE expert computation are each wrapped with
        # ``torch.utils.checkpoint`` inside ``_forward`` so they are recomputed
        # independently during backward (two separate recompute units) instead of
        # wrapping the whole layer with ``checkpoint_wrapper``. Set by the model
        # (``fully_shard``) for layers that should use gradient checkpointing.
        self.use_recompute: bool = False
        # Number of chunks to split the MoE part along the sequence dimension
        # (1 = no split). Configurable via XTUNER_MOE_CHUNK_SIZE on the model.
        self.moe_chunk_size: int = 1
        self.offload_stream: torch.cuda.Stream | None = None
        self.with_shared_expert_gate = with_shared_expert_gate
        self.shared_expert_gate: nn.Module | None
        self.shared_experts: MoEMLP | None

        if n_shared_experts > 0:
            self.shared_experts = MoEMLP(
                hidden_size=hidden_size,
                n_shared_experts=n_shared_experts,
                moe_intermediate_size=moe_intermediate_size,
                hidden_act=hidden_act,
                mlp_bias=mlp_bias,
                float8_cfg=float8_cfg,
            )
            if with_shared_expert_gate:
                self.shared_expert_gate = build_linear(hidden_size, 1, bias=False)
        else:
            self.shared_experts = None
            self.shared_expert_gate = None

        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)

        # Attention computation is factored out into ``MoEAttention`` so that it
        # can be recomputed independently from the MoE expert computation. The
        # submodules remain registered on this layer (see MoEAttention docstring).
        self.attention = MoEAttention(
            self_attn=self.self_attn,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
        )

        self.gate = MoEGate(
            hidden_size=hidden_size,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            router_config=router_config,
            gate_bias=gate_bias,
            router_compute_dtype=router_compute_dtype,
        )
        self.experts = MoEBlock(
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            n_routed_experts=n_routed_experts,
            moe_bias=moe_bias,
            ep_mesh=ep_mesh,
            float8_cfg=float8_cfg,
            moe_act_fn_cfg=moe_act_fn_cfg,
        )
        # TODO: (yehaochen) Maybe should be replaced by build_dispatcher
        process_group = ep_mesh.get_group() if ep_mesh is not None else None
        self.dispatcher = build_dispatcher(
            dispatcher=dispatcher,
            n_routed_experts=n_routed_experts,
            ep_group=process_group,
            training_dtype="fp8" if float8_cfg is not None else "bf16",
            generate_dtype=generate_config.dtype if generate_config is not None else "bf16",
        )

    def forward(
        self,
        *hidden_states: torch.Tensor,
        seq_ctx: SequenceContext | list[SequenceContext],
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        layer_idx: int = -1,
    ) -> tuple[HiddenStates, RouterResults] | tuple[torch.Tensor, ...]:
        """Forward pass of the MoE decoder layer.

        Args:
            hidden_states (torch.Tensor): Input hidden states.
            seq_ctx (SequenceContext): Sequence context.
            position_embeddings (tuple[torch.Tensor, torch.Tensor]): Position embeddings.
            past_key_values (list[list[torch.Tensor]], optional): Past key values for pre-filling or decoding.

        Returns:
            tuple[torch.Tensor, RouterResults]: Output hidden states and router results.
        """
        if len(hidden_states) == 1:
            assert isinstance(seq_ctx, SequenceContext), (
                f"seq_ctx should be a SequenceContext instance but got {seq_ctx}"
            )
            assert isinstance(position_embeddings, tuple) and len(position_embeddings) == 2, (
                "position_embeddings should be a tuple of two tensors (position_ids, position_embeds)"
            )
            return self._forward(
                hidden_states=hidden_states[0],
                seq_ctx=seq_ctx,
                position_embeddings=position_embeddings,
                layer_idx = layer_idx,
            )
        else:
            assert isinstance(seq_ctx, list) and len(seq_ctx) == len(hidden_states), (
                "seq_ctx should be a list of SequenceContext instances with the same length as hidden_states"
            )
            assert isinstance(position_embeddings, list) and len(position_embeddings) == len(hidden_states), (
                "position_embeddings should be a list of tuples with the same length as hidden_states"
            )

            return self._micro_batch_forward(
                hidden_states_list=list(hidden_states),
                seq_ctx_list=seq_ctx,
                position_embeddings_list=position_embeddings,
                layer_idx = layer_idx,
            )

    def _hf_expert_forward_for_debug(self, hidden_states: torch.Tensor, router_results: RouterResults, origin_shape):
        # xtuner: num_experts * 2 * expert_dim, hidden_size
        # hf: num_experts, 2 * expert_dim, hidden_size
        origin_gate_up_proj = self.experts.fused_w1w3.weight
        gate_up_proj = origin_gate_up_proj.view(
            self.n_routed_experts, 2 * self.experts.intermediate_size, self.hidden_size
        )

        # xtuner: num_experts * hidden_size, expert_dim
        # hf: num_experts, hidden_size, expert_dim
        origin_down_proj = self.experts.fused_w2.weight
        down_proj = origin_down_proj.view(self.n_routed_experts, self.hidden_size, self.experts.intermediate_size)

        from transformers.activations import ACT2FN

        act_fn = ACT2FN["silu"]

        hidden_states_reshaped = hidden_states.view(-1, hidden_states.size(-1))
        combined_hidden_states = torch.zeros_like(hidden_states_reshaped)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(router_results["topk_ids"], num_classes=self.n_routed_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.n_routed_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states_reshaped[token_idx]
            gate, up = nn.functional.linear(current_state, gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, down_proj[expert_idx])
            current_hidden_states = current_hidden_states * router_results["topk_weights"][token_idx, top_k_pos, None]
            combined_hidden_states.index_add_(0, token_idx, current_hidden_states.to(combined_hidden_states.dtype))

        combined_hidden_states = combined_hidden_states.view(*origin_shape)
        return combined_hidden_states

    def _token_distribution_across_experts(self, tokens_per_expert: torch.Tensor, skip: bool = True):
        if skip:
            return
        """Token distribution across experts."""
        if self.ep_mesh is not None and self.ep_mesh.size() > 1:
            tokens_per_expert_local = tokens_per_expert
            # Create a list to hold gathered tensors from all EP ranks
            tokens_per_expert_list = [torch.zeros_like(tokens_per_expert_local) for _ in range(self.ep_mesh.size())]
            # AllGather within EP group
            ep_group = self.ep_mesh.get_group()
            torch.distributed.all_gather(tokens_per_expert_list, tokens_per_expert_local, group=ep_group)
            # Stack all gathered tensors
            tokens_per_expert_gathered = torch.stack(tokens_per_expert_list, dim=0)
            tokens_per_expert_gathered = tokens_per_expert_gathered.reshape(-1)

            # Save to CSV only on expert 1's rank (rank 1 within EP group)
            ep_rank = torch.distributed.get_rank(group=ep_group)
            if ep_rank == 1:
                import csv
                import os
                # Create output directory under log_dir
                log_dir = os.environ.get("LOG_DIR", "/mnt/huawei/hyf/xtuner_logs_0410_397b")
                # output_dir = os.path.join(log_dir, "expert_stats")
                # os.makedirs(output_dir, exist_ok=True)
                
                # Use a single file for all layers
                rank = torch.distributed.get_rank()
                filename = os.path.join(log_dir, f"all_layers_tokens_per_expert_{rank}.csv")
                
                # Check if file exists to determine if we need to write header
                file_exists = os.path.exists(filename)
                
                # Write to CSV in append mode
                with open(filename, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Write header only if file is newly created
                    if not file_exists:
                        writer.writerow(['layer_idx'] + [f'expert_id_{i}' for i in range(self.n_routed_experts)])
                    # Write data for each EP rank and expert
                    writer.writerow([self.layer_idx] + tokens_per_expert_gathered.tolist())
                
                # print(f"[EP Rank {ep_rank}] Appended layer {self.layer_idx} tokens_per_expert to {filename}")

    def moe_op(
        self,
        hidden_states: torch.Tensor,
        router_results: RouterResults,
        origin_shape: torch.Size,
    ) -> torch.Tensor:
        pre_dispatched = self.dispatcher.dispatch_preprocess(
            hidden_states=hidden_states.view(-1, hidden_states.shape[-1]),
            topk_ids=router_results["topk_ids"],
        )
        dispatched = self.dispatcher.dispatch(
            pre_dispatched=pre_dispatched,
            topk_weights=router_results["topk_weights"],
            decoding=False,
        )  # type: ignore[call-overload]
        post_dispatched = self.dispatcher.dispatch_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
        )
        # ProberList.after_dispatch(
        #     self.layer_idx,
        #     post_dispatched["hidden_states"],
        #     post_dispatched["tokens_per_expert"],
        #     post_dispatched.get("row_ids_map"),  # type: ignore[arg-type]
        #     dispatched["topk_weights"],
        # )
        self._token_distribution_across_experts(post_dispatched["tokens_per_expert"], skip=True)

        experts_out = self.experts(
            post_dispatched["hidden_states"],
            post_dispatched["tokens_per_expert"],
            decoding=False,
        )
        # ProberList.before_combine(
        #     self.layer_idx,
        #     experts_out,
        #     post_dispatched.get("row_ids_map"),  # type: ignore[arg-type]
        #     dispatched["topk_weights"],
        # )
        pre_combined = self.dispatcher.combine_preprocess(
            hidden_states=experts_out,
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            decoding=False,
        )

        combined = self.dispatcher.combine(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            pre_combined=pre_combined,
            decoding=False,
        )
        post_combined = self.dispatcher.combine_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            pre_combined=pre_combined,
            combined=combined,
        )
        combined_hidden_states = post_combined["hidden_states"]
        combined_hidden_states = combined_hidden_states.view(*origin_shape)
        return combined_hidden_states

    def _moe_forward(
        self,
        chunk_hidden: torch.Tensor,
        chunk_residual: torch.Tensor,
        seq_ctx: SequenceContext,
        chunk_start: int,
        chunk_end: int,
        router_results: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """MoE expert computation unit for a single chunk of attention output.

        Runs (optional) skip-pad -> moe_op -> shared experts -> post-moe
        layernorm on ``chunk_hidden`` (a contiguous slice of the attention
        output along the sequence dimension). This is the unit that is wrapped
        with ``torch.utils.checkpoint`` for independent recomputation from the
        attention unit. ``chunk_start`` / ``chunk_end`` are used to align the
        skip-pad mask to the chunk's token range.

        ``router_results`` must be provided; the gate is computed
        on the full sequence inside ``_attention_with_gate`` to avoid GPU kernel
        divergence from different input shapes.
        """
        skip_pad_tokens = (os.environ.get("SKIP_PAD_TOKENS", "False") == "True")
        if skip_pad_tokens:
            chunk_mask = seq_ctx.mask[:, chunk_start:chunk_end]
            nonpad_indices = torch.nonzero(chunk_mask, as_tuple=True)[1]
            pad_indices = torch.nonzero(~chunk_mask, as_tuple=True)[1]
            origin_hidden_states = chunk_hidden
            chunk_hidden = origin_hidden_states[:, nonpad_indices, :]
            chunk_residual = chunk_residual[:, nonpad_indices, :]
            pad_hidden_states = origin_hidden_states[:, pad_indices, :]
            aligned_topk_ids = router_results["topk_ids"][nonpad_indices, :]
            aligned_topk_weights = router_results["topk_weights"][nonpad_indices, :]
        else:
            aligned_topk_ids = router_results["topk_ids"]
            aligned_topk_weights = router_results["topk_weights"]
        origin_shape = chunk_hidden.shape

        chunk_router_results = {
            "topk_ids": aligned_topk_ids,
            "topk_weights": aligned_topk_weights,
        }
        combined_hidden_states = self.moe_op(
            hidden_states=chunk_hidden,
            router_results=chunk_router_results,
            origin_shape=origin_shape,
        )

        if self.n_shared_experts > 0:
            shared_experts_out = self._shared_experts_forward(hidden_states=chunk_hidden)
        else:
            shared_experts_out = None

        chunk_hidden = self._post_moe_forward(
            combined_hidden_states=combined_hidden_states,
            residual=chunk_residual,
            shared_experts_out=shared_experts_out,
        )
        if skip_pad_tokens:
            result = torch.zeros_like(origin_hidden_states)
            result[:, nonpad_indices, :] = chunk_hidden
            result[:, pad_indices, :] = pad_hidden_states
            chunk_hidden = result
        return chunk_hidden

    def _attention_with_gate(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        state: ForwardState,
    ) -> tuple[HiddenStates, HiddenStates, RouterResults]:
        """Attention + gate computed as a single unit for checkpoint compatibility.

        The gate is computed on the full attention output so that the router
        sees the same input shape regardless of ``moe_chunk_size``, avoiding
        GPU kernel divergence in the linear layer.
        """
        residual, hidden_states = self.attention(
            hidden_states=hidden_states,
            seq_ctx=seq_ctx,
            position_embeddings=position_embeddings,
            state=state,
        )
        rollout_routed_experts = None
        if seq_ctx.rollout_routed_experts is not None and self.layer_idx < seq_ctx.rollout_routed_experts.shape[1]:
            rollout_routed_experts = seq_ctx.rollout_routed_experts[:, self.layer_idx, :]
        router_results: RouterResults = self.gate(hidden_states, rollout_routed_experts)
        return residual, hidden_states, router_results

    def _forward(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        layer_idx: int,
    ) -> tuple[HiddenStates, RouterLogits, RouterWeights]:
        # Attention unit (recomputed independently when use_recompute is set).
        # The gate is computed inside the same checkpoint as attention so that
        # the router sees the full attention output regardless of moe_chunk_size.
        if self.use_recompute:
            residual, hidden_states, router_results = torch.utils.checkpoint.checkpoint(
                self._attention_with_gate,
                hidden_states,
                seq_ctx,
                position_embeddings,
                ForwardState.TRAINING,
                use_reentrant=False,
            )
        else:
            residual, hidden_states, router_results = self._attention_with_gate(
                hidden_states=hidden_states,
                seq_ctx=seq_ctx,
                position_embeddings=position_embeddings,
                state=ForwardState.TRAINING,
            )
        # MoE unit: the attention output is split into ``moe_chunk_size`` chunks
        # along the sequence dimension and each chunk is fed to
        # ``_moe_forward`` (which runs skip-pad -> moe_op -> shared -> post-moe).
        # The gate is computed on the full sequence inside
        # ``_attention_with_gate``, and the router results are sliced and passed
        # to each chunk to avoid GPU kernel divergence.
        #
        # When ``offload_stream`` is set (XTUNER_ACTIVATION_OFFLOAD=1), the
        # entire chunk loop is wrapped with ``async_save_on_cpu`` (group="moe")
        # so that the checkpoint-saved inputs to ``_moe_forward`` -- i.e. the
        # attention output slices -- are offloaded to CPU during forward and
        # fetched back during backward. This complements the decoder-layer-level
        # offload (group="text") driven from ``moe.py`` which offloads the
        # decoder layer's input activation.
        seq_len = hidden_states.shape[1]
        moe_chunk_size = self.moe_chunk_size
        chunk_size = seq_len // moe_chunk_size if moe_chunk_size > 1 else seq_len

        outputs: list[torch.Tensor] = []
        # print(f"{torch.distributed.get_rank()==0}:moe_chunk_size={moe_chunk_size},hidden_states.shape={hidden_states.shape}")
        for i in range(moe_chunk_size):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < moe_chunk_size - 1 else seq_len
            chunk_hidden = hidden_states[:, start:end, :].clone()
            chunk_residual = residual[:, start:end, :].clone()
            # Slice the pre-computed full-sequence router results to this chunk.
            chunk_router_results = {
                "topk_ids": router_results["topk_ids"][start:end, :].clone(),
                "topk_weights": router_results["topk_weights"][start:end, :].clone(),
            }
            # Only offload the _moe_forward *inputs* (chunk_hidden / chunk_residual /
            # chunk_router_results tensors) to CPU; skip all intermediate activations
            # produced inside _moe_forward (e.g. npu_moe_token_permute outputs whose
            # storage layout is incompatible with the naive storage copy). Matched by
            # data_ptr, mirroring the decoder-layer-level offload in moe.py.
            if self.offload_stream is not None:
                router_ptrs = [chunk_router_results["topk_ids"].data_ptr(), chunk_router_results["topk_weights"].data_ptr()]
                offload_ctx = async_save_on_cpu(
                    h2d_stream=self.offload_stream,
                    d2h_stream=self.offload_stream,
                    block_idx=self.layer_idx,
                    group=f"moe_chunk_{i}",
                    custom_check_fn=lambda x, ch=chunk_hidden, cr=chunk_residual, rp=router_ptrs: (
                        x.data_ptr() == ch.data_ptr()
                        or x.data_ptr() == cr.data_ptr()
                        or x.data_ptr() in rp
                    ),
                )
            else:
                offload_ctx = contextlib.nullcontext()
            with offload_ctx:
                if self.use_recompute:
                    out_h = torch.utils.checkpoint.checkpoint(
                        self._moe_forward,
                        chunk_hidden,
                        chunk_residual,
                        seq_ctx,
                        start,
                        end,
                        chunk_router_results,
                        use_reentrant=False,
                    )
                else:
                    out_h = self._moe_forward(
                        chunk_hidden=chunk_hidden,
                        chunk_residual=chunk_residual,
                        seq_ctx=seq_ctx,
                        chunk_start=start,
                        chunk_end=end,
                        router_results=chunk_router_results,
                    )
            outputs.append(out_h)
        # if self.offload_stream is not None:
        #     torch.cuda.current_stream().wait_stream(self.offload_stream)
        hidden_states = torch.cat(outputs, dim=1)
        return hidden_states, router_results["logits"], router_results["router_weights"]

    def _micro_batch_forward(
        self,
        hidden_states_list: list[torch.Tensor],
        seq_ctx_list: list[SequenceContext],
        position_embeddings_list: list[tuple[torch.Tensor, torch.Tensor]],
        layer_idx: int,
    ) -> tuple[torch.Tensor, ...]:  # (HiddenStates, HiddenStates, RouterLogits, RouterLogits)
        origin_shape = hidden_states_list[0].shape
        assert all(hidden_states.shape == origin_shape for hidden_states in hidden_states_list), (
            "All hidden states should have the same shape"
        )
        intra_layer_micro_batch = len(hidden_states_list)
        residual_list: list[torch.Tensor] = []
        router_results_list: list[RouterResults] = []

        pre_dispatched_list: list[PreDispatchResult] = []
        dispatched_list: list[DispatchResult] = []
        pre_moe_forward_out_list: list[torch.Tensor] = []

        # Attention + gate + pre-dispatch
        for (
            hidden_states,
            seq_ctx,
            position_embeddings,
        ) in zip(
            hidden_states_list,
            seq_ctx_list,
            position_embeddings_list,
        ):
            residual, hidden_states, router_results = self._pre_moe_forward(
                hidden_states=hidden_states,
                seq_ctx=seq_ctx,
                position_embeddings=position_embeddings,
                state=ForwardState.TRAINING,
            )
            pre_moe_forward_out_list.append(hidden_states)
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            pre_dispatched = self.dispatcher.dispatch_preprocess(
                hidden_states=hidden_states,
                topk_ids=router_results["topk_ids"],
                async_op=True,
            )
            pre_dispatched_list.append(pre_dispatched)
            residual_list.append(residual)
            router_results_list.append(router_results)

        post_dispatched_list: list[PostDispatchResult] = []
        experts_out_list: list[torch.Tensor] = []
        pre_combined_list: list[PreCombineResult] = []
        combined_list: list[CombineResult] = []

        # dispatch + experts + pre-combine
        for router_results, pre_dispatched in zip(
            router_results_list,
            pre_dispatched_list,
        ):
            dispatched = self.dispatcher.dispatch(
                pre_dispatched=pre_dispatched,
                topk_weights=router_results["topk_weights"],
                async_op=True,
            )
            # wait for pre-dispatch event
            post_dispatched = self.dispatcher.dispatch_postprocess(
                pre_dispatched=pre_dispatched,
                dispatched=dispatched,
                async_op=True,
            )
            experts_out = self.experts(
                post_dispatched["hidden_states"],
                post_dispatched["tokens_per_expert"],
                decoding=False,
            )

            pre_combined = self.dispatcher.combine_preprocess(
                hidden_states=experts_out,
                pre_dispatched=pre_dispatched,
                dispatched=dispatched,
                post_dispatched=post_dispatched,
                async_op=True,
            )

            post_dispatched_list.append(post_dispatched)
            experts_out_list.append(experts_out)
            dispatched_list.append(dispatched)
            pre_combined_list.append(pre_combined)

        for pre_combined, pre_dispatched, dispatched, post_dispatched in zip(
            pre_combined_list,
            pre_dispatched_list,
            dispatched_list,
            post_dispatched_list,
        ):
            combined = self.dispatcher.combine(
                pre_combined=pre_combined,
                pre_dispatched=pre_dispatched,
                dispatched=dispatched,
                post_dispatched=post_dispatched,
                async_op=True,
            )
            combined_list.append(combined)

        shared_experts_out_list: list[torch.Tensor | None]

        if self.n_shared_experts > 0:
            shared_experts_out_list = []
            for pre_moe_forward_out in pre_moe_forward_out_list:
                shared_experts_out = self._shared_experts_forward(
                    hidden_states=pre_moe_forward_out,
                )
                shared_experts_out_list.append(shared_experts_out)
        else:
            shared_experts_out_list = [None] * intra_layer_micro_batch

        hidden_states_out_list: list[torch.Tensor] = []
        for i in range(intra_layer_micro_batch):
            post_combined = self.dispatcher.combine_postprocess(
                pre_dispatched=pre_dispatched_list[i],
                dispatched=dispatched_list[i],
                post_dispatched=post_dispatched_list[i],
                pre_combined=pre_combined_list[i],
                combined=combined_list[i],
                async_op=True,
            )
            hidden_states = self._post_moe_forward(
                # hidden_states=pre_moe_forward_out_list[i],
                combined_hidden_states=post_combined["hidden_states"].view(*pre_moe_forward_out_list[i].shape),
                residual=residual_list[i],
                shared_experts_out=shared_experts_out_list[i],
            )
            hidden_states_out_list.append(hidden_states)

        router_logits = [router_results["logits"] for router_results in router_results_list]
        router_weights = [router_results["router_weights"] for router_results in router_results_list]
        return tuple(hidden_states_out_list + router_logits + router_weights)

    def _pre_moe_forward(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        state: ForwardState,
        past_key_values: list[list[torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, RouterResults]:
        # NOTE: The attention + post-attention-layernorm computation is factored
        # out into ``self.attention`` (MoEAttention) so that it can be recomputed
        # independently from the MoE expert computation. The router gate stays
        # here since it is part of the MoE routing.
        residual, hidden_states = self.attention(
            hidden_states=hidden_states,
            seq_ctx=seq_ctx,
            position_embeddings=position_embeddings,
            state=state,
            past_key_values=past_key_values,
        )

        if seq_ctx.rollout_routed_experts is not None and self.layer_idx < seq_ctx.rollout_routed_experts.shape[1]:
            rollout_routed_experts = seq_ctx.rollout_routed_experts[:, self.layer_idx, :]  # seq_l, expert
        else:
            rollout_routed_experts = None
        router_results: RouterResults = self.gate(hidden_states, rollout_routed_experts)
        return residual, hidden_states, router_results

    def _shared_experts_forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        assert self.shared_experts is not None, "Shared experts should be initialized when n_shared_experts > 0"
        shared_experts_out = self.shared_experts(hidden_states)

        if self.with_shared_expert_gate:
            assert self.shared_expert_gate is not None, (
                "Shared expert gate should be initialized when with_shared_expert_gate is True"
            )
            shared_experts_out = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_experts_out

        return shared_experts_out

    def _post_moe_forward(
        self,
        combined_hidden_states: torch.Tensor,
        residual: torch.Tensor,
        shared_experts_out: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.n_shared_experts > 0:
            shared_experts_out = cast(torch.Tensor, shared_experts_out)
            combined_hidden_states = combined_hidden_states + shared_experts_out
        return combined_hidden_states * self.hidden_factor + residual

    def build_kv_cache(
        self, max_batch_size: int | None = None, max_length: int | None = None, block_size: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.self_attn.build_kv_cache(  # type: ignore
            max_batch_size=max_batch_size,
            max_length=max_length,
            block_size=block_size,
        )


class _BackwardSync(Function):
    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        previous_backward_event: torch.cuda.Event | None = None,
        finished_backward_event: torch.cuda.Event | None = None,
        name=None,
    ) -> torch.Tensor:
        ctx.previous_backward_event = previous_backward_event
        ctx.finished_backward_event = finished_backward_event
        ctx.name = name
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        current_stream = torch.cuda.current_stream()

        if ctx.previous_backward_event is not None:
            current_stream.wait_event(ctx.previous_backward_event)
        if ctx.finished_backward_event is not None:
            current_stream.record_event(ctx.finished_backward_event)

        return grad_output, None, None, None


backward_sync = _BackwardSync.apply
