#!/usr/bin/env python3
"""Resolve actual Parquet shard paths for veRL's file cache."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
for split in ("train", "test"):
    files = []
    for capability in ("capability1", "capability2", "capability3", "capability4"):
        paths = sorted((root / capability / split).glob("*.parquet"))
        if not paths:
            raise SystemExit(f"Missing data: {root / capability / split}")
        files.extend(str(p) for p in paths)
    print(json.dumps(files))
