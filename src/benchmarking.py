"""Utilities for profling and benchmarking."""


from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from deepspeed.profiling.flops_profiler import FlopsProfiler
from torch import nn
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity
from torch.profiler import profile as torch_profile
from torch.profiler import record_function


@dataclass
class ProfilerOutputs:
    profiler: Union[FlopsProfiler, torch_profile]
    _json: Optional[Dict[str, Any]] = None

    def to_json(self):
        raise NotImplementedError


@dataclass
class DeepspeedProfilerOutputs(ProfilerOutputs):
    def __post_init__(self):
        # Save profile info since end_profile cleans up.
        self._json = {
            "flops": self.profiler.get_total_flops(),
            "macs": self.profiler.get_total_macs(),
            "params": self.profiler.get_total_params(),
            "duration": self.profiler.get_total_duration(),
        }

    def to_json(self):
        return self._json


class TorchProfilerOutputs(ProfilerOutputs):
    def to_json(self):
        events = self.profiler.key_averages()
        # https://github.com/pytorch/pytorch/blob/5b1cedacde7f3f93fd5b59e9a7a42ba13c8b5bfc/torch/autograd/profiler_util.py#L823-L832  # noqa
        total_cpu_time = sum([event.self_cpu_time_total for event in events])
        total_flops = sum([event.flops for event in events])
        total_cuda_time = 0
        for evt in events:
            if evt.device_type == DeviceType.CPU:
                # in legacy profiler, kernel info is stored in cpu events
                if evt.is_legacy:
                    total_cuda_time += evt.self_cuda_time_total
            elif evt.device_type == DeviceType.CUDA:
                # in kineto profiler, there're events with the correct device
                # type (e.g. CUDA)
                total_cuda_time += evt.self_cuda_time_total

        return {
            "cpu_time": total_cpu_time,
            "cuda_time": total_cuda_time,
            "flops": total_flops,
        }


def _ds_profile(func_to_profile):
    """Wrap any function that takes a model as first arg, and profile it."""

    def profiled_func(model: nn.Module, *args, **kwargs):
        prof = FlopsProfiler(model)
        prof.start_profile()

        outputs = func_to_profile(model, *args, **kwargs)

        prof.stop_profile()

        profiler_outputs = DeepspeedProfilerOutputs(prof)

        prof.end_profile()

        return outputs, profiler_outputs

    return profiled_func


def _pytorch_profile(func_to_profile):
    """Wrap any function that takes a model as first arg, and profile it."""

    def profiled_func(*args, **kwargs):
        with torch_profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True
        ) as prof:
            with record_function("model_inference"):
                outputs = func_to_profile(*args, **kwargs)
            profiler_outputs = TorchProfilerOutputs(prof)
        return outputs, profiler_outputs

    return profiled_func


def profile(func_to_profile, profiler_type="deepspeed"):
    """Wrap any function that takes a model as first arg, and profile it."""
    if profiler_type == "deepspeed":
        return _ds_profile(func_to_profile)
    elif profiler_type == "pytorch":
        return _pytorch_profile(func_to_profile)
    else:
        raise ValueError(f"Unknown profiler type {profiler_type})")
