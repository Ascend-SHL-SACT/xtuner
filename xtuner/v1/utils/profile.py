import time
from contextlib import contextmanager

import torch

from xtuner.v1.utils import get_logger, get_torch_device_module


logger = get_logger()


@contextmanager
def profile_time(desc):
    start_t = time.time()

    yield

    cost_time = time.time() - start_t
    logger.success(f"{desc} Elapsed time {cost_time:.2f} seconds")


@contextmanager
def profile_time_and_memory(desc):
    torch_device = get_torch_device_module()
    start_t = time.time()
    torch_device.reset_peak_memory_stats()

    yield

    max_memory = torch_device.max_memory_allocated()
    cost_time = time.time() - start_t

    logger.success(f"{desc} Elapsed time {cost_time:.2f} seconds, peak gpu memory {max_memory / 1024**3:.1f}G")


# Adapted from https://github.com/volcengine/verl/blob/main/verl/utils/profiler/performance.py
@contextmanager
def timer(name: str, timer_dict: dict[str, float]):
    # TODO: install codetiming in xtuner latest images
    from codetiming import Timer

    with Timer(name=name, logger=None) as t:
        yield
    if name not in timer_dict:
        timer_dict[name] = 0.0
    timer_dict[name] += t.last


def timer_logger(time_dict: dict[str, float]):
    report_lines = [f"  - {name:<25}: {duration:.2f}s" for name, duration in time_dict.items()]
    total_duration = sum(time_dict.values())
    report_lines.append(f"  - {'Total':<25}: {total_duration:.2f}s")
    return "\n".join(report_lines)


def _in_autograd_backward() -> bool:
    """Return True when executing inside the autograd engine's backward pass.

    This includes the ``torch.utils.checkpoint`` NO_REENTRANT recompute, which
    runs under ``enable_grad`` inside an ``unpack_hook`` -- so ``is_grad_enabled``
    cannot tell it apart from the real forward. ``torch._C.
    _current_graph_task_id`` returns ``-1`` outside any backward and a valid id
    inside one, which is the same signal ``torch.utils.checkpoint`` itself uses
    to gate its unpack path. It is thread-aware, so it works inside the autograd
    worker threads that run the recompute (unlike a main-thread flag).
    """
    try:
        return torch._C._current_graph_task_id() != -1
    except Exception:
        return False
