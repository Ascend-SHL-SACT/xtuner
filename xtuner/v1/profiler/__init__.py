import os
from contextlib import contextmanager
from pathlib import Path

from xtuner.v1.utils import get_device


if get_device() == "cuda":
    from .cuda_profile import profiling_memory, profiling_time
elif get_device() == "npu":
    if os.environ.get("XTUNER_PROFILE_ENABLE", "0") == "1":
        from .npu_profile import profiling_memory
        from .profiler_v2 import (
            Profiler,
            ProfilingConfig,
            analyse,
            profiling_config_from_env,
            profiling_time,
        )

    else:
        from .npu_profile import profiling_memory, profiling_time

else:

    @contextmanager
    def profiling_time(profile_dir: Path):
        yield

    @contextmanager
    def profiling_memory(profile_dir: Path):
        yield


__all__ = [
    "profiling_time",
    "profiling_memory",
]
