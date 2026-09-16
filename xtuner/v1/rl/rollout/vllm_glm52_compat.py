"""GLM-5.2 vLLM rollout/weight-sync compatibility layer, gated by ``XTUNER_VLLM_GLM52_COMPAT``.

Everything in this module is new behavior that ``vllm.py`` (upstream) does not have; with the
gate unset, ``vllm.py`` keeps its upstream code paths byte-for-byte. Export
``XTUNER_VLLM_GLM52_COMPAT=1`` in the launch environment (it must be visible to the rollout
worker process; the gate is also forwarded to the vLLM engine worker processes through the
server env dict).

The bundle carries the GLM-5.2-validated behaviors:

- Receive-side weight sync for vllm_ascend (``update_weight_npu_ipc_compat``):
  ``AscendUnquantizedFusedMoEMethod.process_weights_after_loading`` replaces ``w13_weight``/
  ``w2_weight`` with bare transposed parameters (``weight_loader`` lost), and the SFA MLA
  backend (``sparse_mla_backend="torch_npu"``) splits ``kv_b_proj`` into ``W_UV``/``W_UK_T``
  and disposes the parameter to shape ``(0,)`` (``mla_v1`` splits without disposing); the
  parameter must be restored before a sync ``load_weights`` and the split re-applied after.
  Also: vllm 0.23 exposes the in-engine TP rank
  as ``self.rank`` (no ``global_rank``), NPU IPC rebuild signatures carry the sender's device
  index while CPU shm rebuilds do not, and shm-backed host tensors must not be unmapped
  before their H2D copies complete.
- Token-level rollout through ``/v1/completions``: the base ``RolloutWorker.generate``
  template plus a payload builder and a completions-to-native response translation.
- Engine env overrides (applied in ``vllm.py`` behind the same gate): clear
  ``PYTORCH_NPU_ALLOC_CONF`` (vllm_ascend's sleep mode uses CaMemAllocator, which forbids
  ``expandable_segments``) and stop spoofing ``VLLM_VERSION`` (vllm_ascend's
  ``vllm_version_is()`` consumes it to select version-specific code paths).
"""

import base64
import json
import os
from multiprocessing.reduction import ForkingPickler
from typing import TYPE_CHECKING, Any, cast

import httpx
import torch

from xtuner.v1.utils.device import get_torch_device_module


if TYPE_CHECKING:
    from xtuner.v1.data_proto.rl_data import RolloutState
    from xtuner.v1.rl.rollout.vllm import vLLMWorker

GLM52_COMPAT_ENV = "XTUNER_VLLM_GLM52_COMPAT"

DEVICE_MODULE = get_torch_device_module()


def glm52_compat_enabled() -> bool:
    """Return whether the GLM-5.2 compat bundle is enabled via the gate env var."""
    return os.environ.get(GLM52_COMPAT_ENV) == "1"


def apply_engine_env(env: dict[str, str]) -> None:
    """Apply the GLM-5.2-validated engine env overrides onto the vLLM server env dict.

    Must run on the dict that ``vllm.py`` packs into ``parallel_config.ray_runtime_env``:
    the engine-side processes (APIServer, EngineCore and RayWorkerProc actors) build their
    environment from that dict explicitly, so launch-shell exports never reach them. Two
    overrides, both vllm_ascend-specific:

    - Clear ``PYTORCH_NPU_ALLOC_CONF``: the job env sets ``expandable_segments:True``
      globally, but vllm_ascend's sleep mode uses CaMemAllocator, which forbids
      expandable segments (camem.py).
    - Drop ``VLLM_VERSION``: vllm_ascend's ``vllm_version_is()`` consumes it to select
      version-specific code paths, so a spoofed version selects the wrong ones. Note: Ray
      merges runtime ``env_vars`` on top of the inherited process environment, so a launch
      shell that still exports ``VLLM_VERSION`` keeps the spoof alive in the engine
      processes; the launch environment must not export it alongside the gate.

    The gate itself is propagated into the dict so the engine-side
    ``WorkerWrap.update_weight_npu_ipc`` sees the same gate decision when it re-checks
    ``glm52_compat_enabled`` in the engine process. The caller gates the call (``vllm.py``
    invokes this under ``glm52_compat_enabled``).

    Args:
        env (dict[str, str]): The env dict forwarded to the engine actor processes via
            ``ray_runtime_env``; mutated in place.
    """
    env.pop("VLLM_VERSION", None)
    env["PYTORCH_NPU_ALLOC_CONF"] = ""
    # Propagate the gate itself to the engine worker processes.
    env[GLM52_COMPAT_ENV] = "1"


def install_glm52_compat(worker: "vLLMWorker") -> None:
    """Install the token-level ``/v1/completions`` rollout behavior onto the worker class.

    Overrides ``_get_request_payload``/``_safe_handle_response`` with the compat versions so the
    base-class ``RolloutWorker.generate`` template runs (the ``generate``/``get_logprobs``
    pass-through stubs are deleted at ``vllm.py`` import time, before the ActorClass is
    created). Idempotent: the class is patched once, while the per-instance endpoint switch is
    applied on every call.

    Args:
        worker (vLLMWorker): The rollout worker instance being initialized.
    """
    cls = type(worker)
    worker.endpoints["generate"] = "v1/completions"
    if getattr(cls, "_xtuner_glm52_compat", False):
        return
    # Runtime method installation (setattr: the compat implementations take the worker as
    # their first argument, so they bind exactly like the methods they replace).
    setattr(cls, "_xtuner_glm52_compat", True)
    setattr(cls, "_get_request_payload", _get_request_payload)
    setattr(cls, "_safe_handle_response", _safe_handle_response)
    # The generate/get_logprobs pass-through stubs are deleted at vllm.py import time
    # (module-level gate block): Ray captures actor-method signatures at ActorClass
    # creation, so an __init__-time del here would be too late for that binding.


def update_weight_npu_ipc_compat(worker: Any, data: dict | str) -> None:
    """GLM-5.2-validated replacement for ``WorkerWrap.update_weight_npu_ipc``.

    See the module docstring for the vllm_ascend behaviors this covers.

    Args:
        worker (Any): The engine-side worker extension instance. This method runs on the
            engine's ``NPUWorker`` (vllm injects ``worker_extension_cls`` dynamically), which
            shares no static type with the driver-side classes, hence ``Any``.
        data (dict | str): The sync payload (a JSON string or an already-parsed dict).
    """
    payload: dict[Any, Any] = json.loads(data) if isinstance(data, str) else data

    def _construct(item: tuple[Any, Any]) -> torch.Tensor:
        func, args = item
        args = list(args)
        # NPU tensor IPC rebuilds carry the sender's device index at args[6];
        # host-memory tensors (merged on CPU by the sender, shared via shm
        # filename) rebuild through rebuild_tensor(meta, storage, options) with a
        # shorter signature and must be rebuilt as-is on the CPU side.
        if len(args) > 6:
            args[6] = DEVICE_MODULE.current_device()
        return cast(torch.Tensor, func(*args))

    serialized_data = payload["serialized_named_tensors"]
    if isinstance(serialized_data, list):
        # vllm 0.23 injects worker_extension_cls methods into NPUWorker, which
        # exposes its in-engine TP rank as ``self.rank`` (no ``global_rank``).
        # The sender gathers one shard per rollout_tp rank, so index by TP rank.
        serialized_data = serialized_data[worker.rank]
    weights = ForkingPickler.loads(base64.b64decode(serialized_data))
    weights = [(k, _construct(v)) for k, v in weights]
    DEVICE_MODULE.synchronize()
    model = worker.model_runner.model
    if weights:
        hidden_size = worker.model_runner.vllm_config.model_config.hf_text_config.hidden_size
        _restore_fused_moe_expert_params(model, hidden_size)
        _prepare_kv_b_proj_for_sync(model)
    try:
        model.load_weights(weights=weights)
    except Exception:
        # Log per-weight diagnostics so a failed sync identifies the exact
        # tensor and the receiving parameter's state (name, shapes, and the
        # loader-relevant attrs) instead of surfacing as a bare 500.
        params_dict = dict(model.named_parameters())
        print(
            f"[XTuner][update_weight_npu_ipc] load_weights failed; batch has {len(weights)} weights",
            flush=True,
        )
        for name, tensor in weights:
            param = params_dict.get(name)
            param_shape = tuple(param.shape) if param is not None else None
            attrs = {
                key: getattr(param, key, None)
                for key in ("output_dim", "input_dim", "is_sharded_weight", "packed_dim", "shard_id")
            }
            has_loader = param is not None and hasattr(param, "weight_loader")
            print(
                f"[XTuner][update_weight_npu_ipc] {name}: incoming={tuple(tensor.shape)} "
                f"param={param_shape} attrs={attrs} weight_loader={'yes' if has_loader else 'NO'}",
                flush=True,
            )
        raise
    if payload.get("finished") or not weights:
        # The finished marker batch carries no weights; also treat an empty
        # payload as the end of the sync so the re-transpose cannot be missed.
        _reapply_fused_moe_postprocess(model)
        act_dtype = worker.model_runner.vllm_config.model_config.dtype
        _reapply_kv_b_postprocess(model, act_dtype)
    # Synchronize before dropping the rebuilt tensors: host-memory weights are
    # backed by sender-shared shm pages (and NPU IPC tensors by the sender's
    # storage), so pending H2D copies must complete before the mappings go away.
    DEVICE_MODULE.synchronize()
    del weights
    DEVICE_MODULE.empty_cache()


def _get_request_payload(worker: "vLLMWorker", rollout_state: "RolloutState") -> dict:
    sample_params = rollout_state.sample_params

    if sample_params.return_token_ids:
        # Token-level path: vLLM /v1/completions accepts raw prompt token ids and, with
        # return_token_ids=True, returns the generated ids (CompletionResponseChoice.token_ids).
        if rollout_state.tokens is not None:
            payload: dict[str, Any] = {
                "model": worker.config.model_path,
                "prompt": rollout_state.tokens,
                "input_ids": rollout_state.tokens,  # consumed by RolloutWorker.generate early-exit checks
            }
        else:
            text_prompt = worker.tokenizer.apply_chat_template(
                rollout_state.message, tokenize=False, add_generation_prompt=True
            )
            payload = {"model": worker.config.model_path, "prompt": text_prompt}

        payload.update(
            {
                "stream": sample_params.stream,
                "n": sample_params.n,
                "temperature": sample_params.temperature,
                "top_p": sample_params.top_p,
                "top_k": sample_params.top_k,
                "max_tokens": sample_params.max_tokens,
                "min_tokens": sample_params.min_tokens,
                "repetition_penalty": sample_params.repetition_penalty,
                "presence_penalty": sample_params.presence_penalty,
                "frequency_penalty": sample_params.frequency_penalty,
                "stop": sample_params.stops,
                "stop_token_ids": sample_params.stop_token_ids,
                "skip_special_tokens": sample_params.skip_special_tokens,
                "spaces_between_special_tokens": sample_params.spaces_between_special_tokens,
                # Mirror the chat path's _transform_sample_params mapping: no_stop_trim is the
                # sample-params flag that controls stop-string retention for the vLLM backend.
                "include_stop_str_in_output": sample_params.no_stop_trim,
                "return_token_ids": True,
                "logprobs": 0,
            }
        )
        if sample_params.sampling_seed is not None:
            payload["seed"] = sample_params.sampling_seed
        return payload

    payload = {"model": worker.config.model_path, "stream": sample_params.stream}

    if rollout_state.tools is not None:
        payload["tools"] = rollout_state.tools
    if rollout_state.tool_choice is not None:
        payload["tool_choice"] = rollout_state.tool_choice

    vllm_sample_params = worker._transform_sample_params(sample_params.model_dump())

    if (
        worker.enable_return_routed_experts
        and sample_params.return_routed_experts
        and not rollout_state.extra_fields.get("disable_routed_experts", False)
    ):
        payload["return_routed_experts"] = True

    payload["messages"] = rollout_state.message
    payload.update(vllm_sample_params)
    return payload


async def _safe_handle_response(
    worker: "vLLMWorker", rollout_state: "RolloutState", http_response: httpx.Response
) -> "RolloutState":
    """Translate the vLLM completions response into the token-out shape the base parser expects."""
    if rollout_state.sample_params.return_token_ids:
        response = http_response.json()
        if response.get("choices"):
            choice = response["choices"][0]
            token_ids = choice.get("token_ids") or []
            token_logprobs = (choice.get("logprobs") or {}).get("token_logprobs") or []
            # prompt_tokens backs the base parser's partial-rollout path, which reads
            # meta_info.prompt_tokens when enable_partial_rollout is set; vLLM reports the
            # prompt token count in the usage block, falling back to the local prompt ids.
            prompt_tokens = response.get("usage", {}).get("prompt_tokens") or len(rollout_state.tokens or [])
            native: dict[str, Any] = {
                "text": choice.get("text", ""),
                "output_ids": token_ids,
                "meta_info": {"completion_tokens": len(token_ids), "prompt_tokens": prompt_tokens},
            }
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                native["meta_info"]["finish_reason"] = {"type": finish_reason}
            if token_ids and len(token_logprobs) != len(token_ids):
                print(
                    f"[XTuner][_safe_handle_response] token_logprobs ({len(token_logprobs)}) and token_ids "
                    f"({len(token_ids)}) length mismatch; output_token_logprobs omitted",
                    flush=True,
                )
            if token_ids and len(token_logprobs) == len(token_ids):
                native["meta_info"]["output_token_logprobs"] = [
                    [logprob, token_id] for logprob, token_id in zip(token_logprobs, token_ids)
                ]
            http_response = httpx.Response(status_code=http_response.status_code, json=native)
    # Explicit base-class dispatch instead of super(): the function is installed onto the
    # vLLMWorker class at runtime, so zero-arg super() is unavailable here. Lazy import keeps
    # the module-level import graph acyclic.
    from xtuner.v1.rl.rollout.worker import RolloutWorker

    return await RolloutWorker._safe_handle_response(worker, rollout_state, http_response)


def _is_fused_moe_module(module: torch.nn.Module) -> bool:
    """Duck-type check for FusedMoE-like modules holding fused expert weights."""
    w13 = getattr(module, "w13_weight", None)
    w2 = getattr(module, "w2_weight", None)
    quant_method = getattr(module, "quant_method", None)
    return (
        isinstance(w13, torch.Tensor)
        and w13.dim() == 3
        and isinstance(w2, torch.Tensor)
        and w2.dim() == 3
        and hasattr(quant_method, "process_weights_after_loading")
    )


def _restore_fused_moe_expert_params(model: torch.nn.Module, hidden_size: int) -> None:
    """Recover ``weight_loader`` on FusedMoE expert params before a sync ``load_weights``.

    vllm_ascend's ``AscendUnquantizedFusedMoEMethod.process_weights_after_loading`` replaces
    ``w13_weight``/``w2_weight`` with bare transposed ``nn.Parameter`` instances (the layout
    consumed by the NPU grouped matmul), which strips ``weight_loader``; ``wake_up`` then
    flips them back to the standard layout. ``model.load_weights`` resolves expert weights
    through ``param.weight_loader`` on the standard layout, so restore both here. The layout
    is detected from the shape, so the restore is idempotent and moves no data when the
    params already sit in the standard layout. Assumes ``2 * moe_intermediate_size !=
    hidden_size`` so the two layouts cannot be confused.
    """
    for module in model.modules():
        if not _is_fused_moe_module(module):
            continue
        loader = getattr(module, "weight_loader", None)
        if loader is None:
            continue
        for attr in ("w13_weight", "w2_weight"):
            param: torch.nn.Parameter = getattr(module, attr)
            # w13: standard [E, 2I, H] vs transposed [E, H, 2I];
            # w2: standard [E, H, I] vs transposed [E, I, H].
            if attr == "w13_weight":
                needs_transpose = param.shape[-2] == hidden_size
            else:
                needs_transpose = param.shape[-1] == hidden_size
            if needs_transpose:
                param.data = param.data.transpose(1, 2)
            if not hasattr(param, "weight_loader"):
                param.weight_loader = loader  # type: ignore[attr-defined]


def _reapply_fused_moe_postprocess(model: torch.nn.Module) -> None:
    """Re-run vllm_ascend's MoE post-processing after a sync ``load_weights``.

    The NPU grouped matmul consumes the transposed layout produced by
    ``AscendUnquantizedFusedMoEMethod.process_weights_after_loading``; call it again on the
    freshly synced expert weights so the next generation step sees the expected layout.
    """
    for module in model.modules():
        if not _is_fused_moe_module(module):
            continue
        module.quant_method.process_weights_after_loading(module)  # type: ignore[union-attr, operator]


def _get_sfa_mla_impls(model: torch.nn.Module) -> list:
    """Collect attention impls that hold vllm_ascend's split kv_b state.

    vllm_ascend's ``AscendSFAImpl`` (selected by ``sparse_mla_backend="torch_npu"``) and
    ``AscendMLAImpl`` (the ``mla_v1`` backend) store ``W_UV``/``W_UK_T`` views split from
    ``kv_b_proj`` on the impl object reachable as ``Attention.impl``.
    """
    impls = []
    for module in model.modules():
        impl = getattr(module, "impl", None)
        if impl is not None and hasattr(impl, "W_UV") and hasattr(impl, "kv_b_proj"):
            impls.append(impl)
    return impls


def _prepare_kv_b_proj_for_sync(model: torch.nn.Module) -> None:
    """Make the split-kv_b impls loadable again and mark them for the post-sync re-split.

    vllm_ascend's ``AscendSFAImpl.process_weights_after_loading`` splits ``kv_b_proj`` into
    ``W_UV``/``W_UK_T`` buffers, then calls ``dispose_layer`` to shrink the original
    parameter to empty (``dispose_tensor`` -> shape ``(0,)``); the ``mla_v1`` backend
    (``AscendMLAImpl``) splits the same way but keeps the parameter. A sync ``load_weights``
    on a disposed parameter hits ``assert param_data.shape == loaded_weight.shape`` in the
    ColumnParallelLinear weight loader. Re-allocate disposed parameters with the analytic
    TP-shard shape — zeros, so that a sync which never delivers ``kv_b_proj`` cannot leak
    uninitialized memory into the split buffers — so the loader can copy into them again.
    Every split-holding impl is marked so ``_reapply_kv_b_postprocess`` re-runs the
    backend's split for it after the sync; skipping the mark would leave the consumed
    ``W_UV``/``W_UK_T`` buffers stale while the parameter holds the new weights.
    """
    for impl in _get_sfa_mla_impls(model):
        weight: torch.nn.Parameter = impl.kv_b_proj.weight
        if weight.numel() == 0:
            shape = (impl.local_num_heads * (impl.qk_nope_head_dim + impl.v_head_dim), impl.kv_lora_rank)
            weight.data = torch.zeros(shape, dtype=weight.dtype, device=weight.device)
        impl._xtuner_kv_b_restored = True


def _reapply_kv_b_postprocess(model: torch.nn.Module, act_dtype: torch.dtype) -> None:
    """Re-run the backend kv_b split after a sync ``load_weights``.

    The split-kv_b backends (SFA and ``mla_v1``) consume ``W_UV``/``W_UK_T`` (whose
    addresses must stay stable across iterations), so after the freshly synced weights
    land in ``kv_b_proj``, call the backend's ``process_weights_after_loading`` again:
    it copies the new weight into the existing buffers and, for the SFA backend,
    disposes the parameter once more.
    """
    for impl in _get_sfa_mla_impls(model):
        if not getattr(impl, "_xtuner_kv_b_restored", False):
            continue
        if impl.kv_b_proj.weight.numel() == 0:
            continue
        impl.process_weights_after_loading(act_dtype)
        impl._xtuner_kv_b_restored = False
