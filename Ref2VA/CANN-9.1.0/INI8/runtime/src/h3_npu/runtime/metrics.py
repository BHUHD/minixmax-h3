"""Step-level wall time / NPU HBM / host RAM snapshots."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import torch


def _host_rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def _host_mem_mb() -> dict[str, float]:
    out = {"MemTotal": float("nan"), "MemAvailable": float("nan"), "MemFree": float("nan")}
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                key = line.split(":")[0]
                if key in out:
                    out[key] = int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return {k: round(v, 1) for k, v in out.items()}


def npu_mem_mb(device: Optional[torch.device] = None) -> dict[str, float]:
    if not torch.npu.is_available():
        return {"allocated": float("nan"), "reserved": float("nan"), "max_allocated": float("nan")}
    return {
        "allocated": round(torch.npu.memory_allocated(device) / (1024**2), 1),
        "reserved": round(torch.npu.memory_reserved(device) / (1024**2), 1),
        "max_allocated": round(torch.npu.max_memory_allocated(device) / (1024**2), 1),
    }


def snapshot(tag: str, *, device: Optional[torch.device] = None, extra: Optional[dict] = None) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "tag": tag,
        "ts": time.time(),
        "host_rss_mb": round(_host_rss_mb(), 1),
        "host_mem_mb": _host_mem_mb(),
        "npu_mem_mb": npu_mem_mb(device),
        "pid": os.getpid(),
        "rank": int(os.environ.get("RANK", "0")),
    }
    if extra:
        rec.update(extra)
    return rec


class MetricsLog:
    def __init__(self):
        self.records: list[dict[str, Any]] = []
        self._t0 = time.time()

    def add(self, tag: str, *, device=None, extra=None) -> dict[str, Any]:
        rec = snapshot(tag, device=device, extra=extra)
        rec["elapsed_s"] = round(time.time() - self._t0, 3)
        self.records.append(rec)
        return rec

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")
