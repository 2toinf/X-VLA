"""Device selection for the inference entry points.

Detection order is XPU, then CUDA, then CPU. An explicit device always wins.
"""

from typing import Optional, Union

import torch


def is_available(device_type: str) -> bool:
    """Whether ``device_type`` is usable in this torch build and on this host."""
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "cuda":
        return torch.cuda.is_available()
    return device_type == "cpu"


def default_device_type() -> str:
    """Best available backend: ``xpu``, then ``cuda``, then ``cpu``."""
    for device_type in ("xpu", "cuda"):
        if is_available(device_type):
            return device_type
    return "cpu"


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve ``None`` or ``"auto"`` to the best backend, else honour ``device``.

    A bare ``xpu`` or ``cuda`` becomes index 0.
    """
    if device is None or device == "auto":
        device_type = default_device_type()
        return torch.device(device_type, 0) if device_type != "cpu" else torch.device("cpu")
    device = torch.device(device)
    if device.type in ("xpu", "cuda") and device.index is None:
        device = torch.device(device.type, 0)
    return device
