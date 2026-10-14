"""Shared, authenticated HF dataset preparation for both training backends."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
import yaml

ROOT = Path(__file__).resolve().parents[2]
CAPABILITIES = ("capability1", "capability2", "capability3", "capability4")


def settings():
    result = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    repo_id = os.environ.get("SCOPED_DATASET", result["repo_id"])
    revision = os.environ.get("SCOPED_DATASET_REVISION", result["revision"] if repo_id == result["repo_id"] else "main")
    if repo_id != result["repo_id"] or revision != result["revision"]:
        result["manifest_sha256"] = None
    result.update(repo_id=repo_id, revision=revision)
    return result


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative_path(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid relative dataset path: {value!r}")
    return path


def remote_manifest(repo_id, revision):
    path = hf_hub_download(repo_id, "manifest.json", repo_type="dataset", revision=revision)
    manifest = json.loads(Path(path).read_text())
    if not manifest.get("complete") or manifest.get("limited_export"):
        raise ValueError("HF repository must contain a complete full-dataset export")
    return manifest, Path(path)


def prepare(output, capabilities=CAPABILITIES, repo_id=None, revision=None, offline=False):
    config = settings()
    repo_id = repo_id or config["repo_id"]
    revision = revision or (config["revision"] if repo_id == config["repo_id"] else "main")
    expected_manifest_hash = config["manifest_sha256"] if (
        repo_id == config["repo_id"] and revision == config["revision"]
    ) else None
    output = Path(output).resolve()
    manifest_path = output / "manifest.json"
    # User-created smoke or experimental subsets are usable without HF access.
    # Their own manifest must explicitly mark them as limited exports.
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    limited = bool(manifest and manifest.get("limited_export"))
    if manifest is None:
        if offline:
            raise ValueError(f"No local dataset manifest: {manifest_path}")
        manifest, _ = remote_manifest(repo_id, revision)
    if not manifest.get("complete"):
        raise ValueError(f"Incomplete dataset export: {output}")
    if manifest_path.is_file() and expected_manifest_hash and not limited:
        if file_sha256(manifest_path) != expected_manifest_hash:
            raise ValueError("Local dataset manifest differs from the pinned HF revision; choose another DATA_DIR")
    required = []
    specs = {}
    for capability in capabilities:
        if capability not in CAPABILITIES:
            raise ValueError(f"Unknown capability: {capability}")
        for split in ("train", "test"):
            split_spec = manifest["capabilities"][capability][split]
            if not split_spec["shards"] or sum(s["rows"] for s in split_spec["shards"]) != split_spec["rows"]:
                raise ValueError(f"Invalid shard counts: {capability}/{split}")
            for shard in split_spec["shards"]:
                filename = str(safe_relative_path(f"{capability}/{split}/{shard['file']}"))
                required.append(filename)
                specs[filename] = shard
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        if offline or limited:
            raise ValueError(f"Missing {len(missing)} local shards under {output}")
        print(f"Downloading {len(missing)} shards from {repo_id}@{revision}", flush=True)
        snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                          local_dir=output, allow_patterns=["README.md", "manifest.json", *required])
        manifest_path = output / "manifest.json"
    if expected_manifest_hash and not limited and file_sha256(manifest_path) != expected_manifest_hash:
        raise ValueError("Downloaded manifest differs from the pinned HF revision")
    if not manifest_path.is_file():
        raise ValueError(f"Dataset manifest missing after preparation: {output}")
    cache_path = output / ".scoped_verified.json"
    try:
        cached = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cached = {}
    for name in required:
        path = output / name
        stat = path.stat()
        spec = specs[name]
        if stat.st_size != spec["bytes"]:
            raise ValueError(f"Shard byte-size mismatch: {name}")
        fingerprint = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": spec["sha256"]}
        if cached.get(name) != fingerprint:
            if file_sha256(path) != spec["sha256"]:
                raise ValueError(f"Shard checksum mismatch: {name}")
            cached[name] = fingerprint
    temporary = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(cached, indent=2) + "\n")
    temporary.replace(cache_path)
    print(f"Data ready: {', '.join(capabilities)} ({len(required)} verified shards)", flush=True)
    return output
