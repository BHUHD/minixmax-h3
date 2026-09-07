#!/usr/bin/env python3
"""启动 FastAPI Worker（本地调试 / 容器内）。"""
from __future__ import annotations

import os
import sys

# 确保可 import api 包
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_ROOT = os.path.join(ROOT, "api")
for p in (ROOT, API_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import uvicorn  # noqa: E402

from api.config import get_settings  # noqa: E402


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "api.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
