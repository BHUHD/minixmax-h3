"""NPU stream helpers for layer overlap / async HCCL."""
from __future__ import annotations

from typing import Optional

_streams: dict[str, object] = {}


def npu_stream(name: str):
    import torch

    if not torch.npu.is_available():
        return torch.npu.current_stream()
    if name not in _streams:
        _streams[name] = torch.npu.Stream()
    return _streams[name]


def record_event(stream=None):
    import torch

    evt = torch.npu.Event()
    evt.record(stream or torch.npu.current_stream())
    return evt


def wait_event(evt, stream=None) -> None:
    import torch

    (stream or torch.npu.current_stream()).wait_event(evt)


def maybe_sync(enabled: bool) -> None:
    if enabled:
        import torch

        torch.npu.synchronize()
