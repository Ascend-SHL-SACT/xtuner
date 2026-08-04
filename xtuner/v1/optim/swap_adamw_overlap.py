"""Copy-stream H2D/D2H overlap for SwapAdamW.step (no persistent device buffer).

Dispatched from swap_adamw.step when any of XTUNER_SWAP_BF16_STATE,
XTUNER_SWAP_D2H_OVERLAP, or XTUNER_SWAP_H2D_OVERLAP is set. m/v stream between
pinned CPU and a fresh per-step device temp; nothing is kept on device across
steps, so swap still saves the m/v device footprint.

Knobs (independent, combinable):
  XTUNER_SWAP_BF16_STATE=1   m/v bf16 on host/PCIe; upcast fp32 on device before
                              adam; the post-adam fp32->bf16 downcast runs on the
                              HOST (D2H fp32->fp32 to a pinned scratch, then a
                              daemon worker casts scratch->bf16 cpu m/v on the
                              CPU, overlapping device fwd+bwd). Never a device
                              cast temp: a copy-stream-allocated temp is never
                              reused by NPU's allocator (the fragmentation source).
  XTUNER_SWAP_D2H_OVERLAP=1  D2H writeback on a copy stream (gated by an
                              adam-done event), overlapping next chunk + next
                              step's fwd+bwd (step-end event).
  XTUNER_SWAP_H2D_OVERLAP=1  H2D on a copy stream one chunk ahead, overlapping
                              the current chunk's adam + D2H.

With no knob set, swap_adamw.step takes its stock serial path. Chunking
(XTUNER_SWAP_OVERLAP_NCHUNKS, default 16) bounds per-chunk in-flight memory.

Memory notes (verified): max_memory is set by fwd/bwd and is invariant across
swap knobs (optimizer temps are chunked and fit in freed-activation space).
reserved_memory varies with allocator-pool fragmentation: any device block
allocated ON the copy stream is never reused (climbs toward the device limit);
record_stream'd blocks (allocated on default, read on copy) are reclaimable.
So D2H-overlap alone is mild (only the fp32 adam temp is record_stream'd);
H2D-overlap (prefetch blocks allocated on the copy stream) fragments hard.
Host-casting the bf16 downcast removes the one copy-stream-allocated D2H block;
the H2D-prefetch block cannot be relocated (the cross-stream write itself taints
regardless of alloc stream -- relocating it fragments worse, not better).
"""

import os
import queue
import threading
from collections.abc import Callable

import torch
from torch.optim.adam import adam as torch_adam

from xtuner.v1.utils import get_device, get_torch_device_module

DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


def _get_host_scratch(optimizer, e: dict) -> dict:
    """Lazy pinned fp32 host scratch for m/v(max); reused across steps."""
    scratch = getattr(optimizer, "_swap_host_scratch", None)
    if scratch is None:
        scratch = {}
        optimizer._swap_host_scratch = scratch
    key = e["param"]
    if key not in scratch:

        def pin(t: torch.Tensor) -> torch.Tensor:
            return torch.empty(t.shape, dtype=torch.float32, device="cpu").pin_memory()

        d = {"m": pin(e["cpu_exp_avg"]), "v": pin(e["cpu_exp_avg_sq"])}
        if e["amsgrad"]:
            d["max"] = pin(e["cpu_max"])
        scratch[key] = d
    return scratch[key]


def _d2h_to_scratch(optimizer, entries: list, stream, cast_pairs: list) -> None:
    """D2H fp32 dev temps to pinned fp32 host scratch on ``stream`` (same-dtype,
    fast). The fp32 adam temp is record_stream'd to ``stream`` (reclaimable).
    Collects (cpu_dst, scratch) pairs holding NO device refs so the fp32 temps
    free at step end (no max_memory rise)."""
    for e in entries:
        sc = _get_host_scratch(optimizer, e)
        sc["m"].copy_(e["exp_avg"], non_blocking=True)
        sc["v"].copy_(e["exp_avg_sq"], non_blocking=True)
        e["exp_avg"].record_stream(stream)
        e["exp_avg_sq"].record_stream(stream)
        if e["amsgrad"] and e["max"] is not None:
            sc["max"].copy_(e["max"], non_blocking=True)
            e["max"].record_stream(stream)
        cast_pairs.append((
            e["cpu_exp_avg"], sc["m"],
            e["cpu_exp_avg_sq"], sc["v"],
            e["cpu_max"], sc.get("max"),
            e["amsgrad"],
        ))


def _d2h_plain(entries: list, stream) -> None:
    """Same-dtype fp32 D2H (no cast) on ``stream``."""
    for e in entries:
        e["cpu_exp_avg"].copy_(e["exp_avg"], non_blocking=True)
        e["cpu_exp_avg_sq"].copy_(e["exp_avg_sq"], non_blocking=True)
        e["exp_avg"].record_stream(stream)
        e["exp_avg_sq"].record_stream(stream)
        if e["amsgrad"] and e["max"] is not None:
            e["cpu_max"].copy_(e["max"], non_blocking=True)
            e["max"].record_stream(stream)


def _submit_host_cast(optimizer, d2h_ev, pairs: list) -> None:
    """Submit the fp32->bf16 host cast to a single daemon worker.

    The worker waits d2h_ev (copy-stream D2H landed), then CPU-casts scratch ->
    bf16 cpu m/v, masked by device fwd+bwd. Daemonesque: does not block process
    teardown (the last step's cast may be abandoned at exit, which is fine since
    the checkpoint path does not read swapped cpu m/v mid-cast)."""
    q = getattr(optimizer, "_swap_cast_q", None)
    if q is None:
        q = queue.Queue()
        optimizer._swap_cast_q = q

        def _loop() -> None:
            while True:
                job = q.get()
                if job is None:
                    return
                ev, prs, done = job
                ev.synchronize()
                for cm, sm, cv, sv, cmx, smx, amg in prs:
                    cm.copy_(sm)
                    cv.copy_(sv)
                    if amg and cmx is not None:
                        cmx.copy_(smx)
                done.set()

        threading.Thread(target=_loop, daemon=True).start()
    done = threading.Event()
    q.put((d2h_ev, pairs, done))
    optimizer._swap_cast_pending = done


@torch.no_grad()
def step_overlap(
    optimizer: torch.optim.Optimizer,
    closure: Callable[[], torch.Tensor] | None = None,
) -> torch.Tensor | None:
    """Run the swap step with optional copy-stream H2D/D2H overlap.

    Args:
        optimizer (SwapAdamW): the wrapped optimizer.
        closure (Callable[[], torch.Tensor] | None): optional closure.

    Returns:
        torch.Tensor | None: the loss from the closure, or None.
    """
    loss: torch.Tensor | None = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    bf16 = bool(int(os.environ.get("XTUNER_SWAP_BF16_STATE", "0")))
    d2h_ov = bool(int(os.environ.get("XTUNER_SWAP_D2H_OVERLAP", "0")))
    h2d_ov = bool(int(os.environ.get("XTUNER_SWAP_H2D_OVERLAP", "0")))

    copy_s = getattr(optimizer, "_swap_copy_stream", None)
    if copy_s is None:
        copy_s = DEVICE_MODULE.Stream()
        optimizer._swap_copy_stream = copy_s
        d2h_ev = DEVICE_MODULE.Event()
        with DEVICE_MODULE.stream(copy_s):
            d2h_ev.record()  # prime: first step's wait is a no-op
        optimizer._swap_d2h_event = d2h_ev
    d2h_ev = optimizer._swap_d2h_event

    default_s = DEVICE_MODULE.current_stream()
    # Prior step's D2H must land before this step reads CPU m/v. With bf16 the
    # host cast runs on the daemon worker; wait for it so the bf16 cpu m/v it
    # wrote is ready for this step's H2D (by now, after fwd+bwd, it is done).
    if bf16:
        pending = getattr(optimizer, "_swap_cast_pending", None)
        if pending is not None:
            pending.wait()
    elif d2h_ov:
        default_s.wait_event(d2h_ev)

    params_list = [p for p in optimizer._param_to_group_map if p.grad is not None]
    n = len(params_list)
    nchunks = max(1, int(os.environ.get("XTUNER_SWAP_OVERLAP_NCHUNKS", "16")))
    nchunks = min(nchunks, n)

    def chunk_slice(ci: int) -> list:
        lo = round(ci * n / nchunks)
        hi = round((ci + 1) * n / nchunks)
        return params_list[lo:hi]

    def h2d_chunk(chunk: list) -> list:
        # H2D m+v to fresh device temps (on the active stream); upcast bf16->fp32.
        entries = []
        for param in chunk:
            if param.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")
            group = optimizer._param_to_group_map[param]
            param_state = optimizer.state[param]
            step_tensor = param_state.get("step")
            if step_tensor is None:
                step_tensor = torch.tensor(0.0, dtype=torch.float32, device="cpu")
                param_state["step"] = step_tensor
            amsgrad = bool(group["amsgrad"])
            cpu_state = optimizer._param_to_cpu_states_map[param]
            cpu_exp_avg = cpu_state["exp_avg"]
            cpu_exp_avg_sq = cpu_state["exp_avg_sq"]
            cpu_max = cpu_state.get("max_exp_avg_sq", None)

            assert isinstance(cpu_exp_avg, torch.Tensor)
            assert isinstance(cpu_exp_avg_sq, torch.Tensor)
            exp_avg = cpu_exp_avg.to(device=DEVICE, non_blocking=True)
            exp_avg_sq = cpu_exp_avg_sq.to(device=DEVICE, non_blocking=True)
            max_exp_avg_sq = None
            if amsgrad:
                assert isinstance(cpu_max, torch.Tensor)
                max_exp_avg_sq = cpu_max.to(device=DEVICE, non_blocking=True)

            if bf16:
                exp_avg = exp_avg.to(torch.float32)
                exp_avg_sq = exp_avg_sq.to(torch.float32)
                if max_exp_avg_sq is not None:
                    max_exp_avg_sq = max_exp_avg_sq.to(torch.float32)

            entries.append({
                "param": param,
                "group": group,
                "amsgrad": amsgrad,
                "step_tensor": step_tensor,
                "cpu_exp_avg": cpu_exp_avg,
                "cpu_exp_avg_sq": cpu_exp_avg_sq,
                "cpu_max": cpu_max,
                "exp_avg": exp_avg,
                "exp_avg_sq": exp_avg_sq,
                "max": max_exp_avg_sq,
            })
        return entries

    def adam_chunk(entries: list) -> None:
        # One foreach torch_adam call per param group in the chunk.
        groups: dict = {}
        for e in entries:
            group = e["group"]
            gid = id(group)
            if gid not in groups:
                groups[gid] = [group, [], [], [], [], [], []]
            slot = groups[gid]
            slot[1].append(optimizer._to_local_tensor(e["param"]))
            slot[2].append(optimizer._to_local_tensor(e["param"].grad))
            slot[3].append(optimizer._to_local_tensor(e["exp_avg"]))
            slot[4].append(optimizer._to_local_tensor(e["exp_avg_sq"]))
            slot[5].append(
                optimizer._to_local_tensor(e["max"]) if e["max"] is not None else None
            )
            slot[6].append(e["step_tensor"])
        for slot in groups.values():
            g = slot[0]
            mx_nonnull = [x for x in slot[5] if x is not None]
            torch_adam(
                slot[1],
                slot[2],
                slot[3],
                slot[4],
                mx_nonnull,
                slot[6],
                amsgrad=bool(g["amsgrad"]),
                has_complex=any(torch.is_complex(p) for p in slot[1]),
                lr=g["lr"],
                beta1=g["betas"][0],
                beta2=g["betas"][1],
                weight_decay=g["weight_decay"],
                eps=g["eps"],
                maximize=g["maximize"],
                foreach=g["foreach"],
                capturable=g["capturable"],
                differentiable=g["differentiable"],
                fused=g["fused"],
                grad_scale=getattr(optimizer, "grad_scale", None),
                found_inf=getattr(optimizer, "found_inf", None),
                decoupled_weight_decay=g["decoupled_weight_decay"],
            )

    cast_pairs: list = []

    def d2h_chunk(entries: list) -> None:
        # bf16: host cast (D2H fp32->scratch, daemon worker casts on CPU).
        # fp32: same-dtype D2H (no cast). Copy stream only when d2h_ov.
        if bf16:
            if d2h_ov:
                ev = DEVICE_MODULE.Event()
                ev.record()
                with DEVICE_MODULE.stream(copy_s):
                    copy_s.wait_event(ev)
                    _d2h_to_scratch(optimizer, entries, copy_s, cast_pairs)
            else:
                _d2h_to_scratch(optimizer, entries, default_s, cast_pairs)
        else:
            if d2h_ov:
                ev = DEVICE_MODULE.Event()
                ev.record()
                with DEVICE_MODULE.stream(copy_s):
                    copy_s.wait_event(ev)
                    _d2h_plain(entries, copy_s)
            else:
                _d2h_plain(entries, default_s)

    # Pipeline driver. With h2d_ov, each chunk's H2D is launched one chunk ahead
    # on copy_s so it overlaps the previous chunk's adam + D2H. The matching
    # h2d_done event is waited on the default stream before adam.
    next_entries: list | None = None
    next_h2d_ev = None
    if h2d_ov and nchunks > 0:
        with DEVICE_MODULE.stream(copy_s):
            next_entries = h2d_chunk(chunk_slice(0))
        next_h2d_ev = DEVICE_MODULE.Event()
        with DEVICE_MODULE.stream(copy_s):
            next_h2d_ev.record()

    for ci in range(nchunks):
        if h2d_ov:
            entries = next_entries
            default_s.wait_event(next_h2d_ev)
            if ci + 1 < nchunks:
                with DEVICE_MODULE.stream(copy_s):
                    next_entries = h2d_chunk(chunk_slice(ci + 1))
                next_h2d_ev = DEVICE_MODULE.Event()
                with DEVICE_MODULE.stream(copy_s):
                    next_h2d_ev.record()
            else:
                next_entries = None
                next_h2d_ev = None
        else:
            entries = h2d_chunk(chunk_slice(ci))

        adam_chunk(entries)
        d2h_chunk(entries)

    # All D2H launched. bf16: record d2h_ev + submit the host cast worker (it
    # waits d2h_ev, then CPU-casts scratch->bf16 cpu m/v, masked by next fwd+bwd;
    # next step's pending.wait() ensures it is done before H2D). fp32: record
    # d2h_ev for next step's default-stream wait.
    if bf16:
        if d2h_ov:
            with DEVICE_MODULE.stream(copy_s):
                d2h_ev.record()
        else:
            d2h_ev.record()
        if cast_pairs:
            _submit_host_cast(optimizer, d2h_ev, cast_pairs)
    elif d2h_ov:
        with DEVICE_MODULE.stream(copy_s):
            d2h_ev.record()
    return loss
