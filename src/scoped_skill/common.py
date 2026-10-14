"""Configuration and atomic, provenance-checked intermediate artifacts."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[2]
CAPABILITIES = tuple(f"capability{i}" for i in range(1, 5))


def load_config(path=None):
    config = yaml.safe_load(Path(path or ROOT / "configs/skill_extraction.yaml").read_text())
    if not math.isfinite(float(config.get("reward_threshold", 0.5))):
        raise ValueError("reward_threshold must be finite")
    if config["sample_per_capability"] != -1 and config["sample_per_capability"] < 1:
        raise ValueError("sample_per_capability must be positive or -1 for all training rows")
    for key in ("rollout_k", "rollout_k_small", "rollout_batch",
                "online_max_workers", "author_attempts", "evidence_per_kind", "server_tp",
                "server_max_images", "max_skill_lines"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    return config


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def signature(**values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()


@contextmanager
def atomic_text(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yield handle
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, value):
    with atomic_text(path) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completed(path, expected):
    path = Path(path)
    metadata = path.with_name(path.name + ".meta.json")
    if not path.is_file() or not metadata.is_file():
        return False
    try:
        record = json.loads(metadata.read_text())
        return record["signature"] == expected and record["sha256"] == digest(path)
    except (OSError, ValueError, KeyError):
        return False


def mark_completed(path, expected):
    path = Path(path)
    write_json(path.with_name(path.name + ".meta.json"), {"signature": expected, "sha256": digest(path)})
