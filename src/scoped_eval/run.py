#!/usr/bin/env python3
"""Eight-benchmark baseline evaluation using HF data and a persistent vLLM API."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import islice
import json
from pathlib import Path
import sys

import yaml

from .data import CONFIG_NAMES, iter_rows
from .config import load_configs, server_settings
from .protocol import OnlineEngine
from .scoring import score_record, summarize
from .server import api_url, check_server, start_server, stop_server

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/lib"))
from scoped_data import file_sha256
from storage import storage_root


def prepare_data(root, names, offline=False):
    config = yaml.safe_load((ROOT / "configs/eval_data.yaml").read_text())
    root = Path(root)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text()) if path.is_file() else None
    if manifest is not None:
        if file_sha256(path) != config["manifest_sha256"]:
            raise ValueError("Evaluation manifest differs from the pinned HF revision")
        missing = [shard["file"] for name in names for shard in manifest["benchmarks"][CONFIG_NAMES[name]]["shards"]
                   if not (root / shard["file"]).is_file()]
    else:
        missing = ["manifest.json"]
    if missing:
        if offline:
            raise FileNotFoundError(f"Missing evaluation data under {root}; remove --offline to download it automatically from HF")
        from huggingface_hub import snapshot_download
        configs = [CONFIG_NAMES[name] for name in names]
        print(f"Downloading evaluation data from {config['repo_id']}: {', '.join(names)}", flush=True)
        snapshot_download(repo_id=config["repo_id"], repo_type="dataset", revision=config["revision"], local_dir=root,
                          allow_patterns=["README.md", "manifest.json", *[f"{name}/**/*.parquet" for name in configs]])
        if file_sha256(path) != config["manifest_sha256"]:
            raise ValueError("Downloaded evaluation manifest differs from pinned revision")
        manifest = json.loads(path.read_text())
    if not manifest.get("complete") or manifest["ccocr_scope"] != "recognition":
        raise ValueError("A complete recognition-only evaluation export is required")
    cache_path = root / ".eval_verified.json"
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cache = {}
    for name in names:
        for shard in manifest["benchmarks"][CONFIG_NAMES[name]]["shards"]:
            file = root / shard["file"]
            stat = file.stat()
            fingerprint = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": shard["sha256"]}
            if stat.st_size != shard["bytes"]:
                raise ValueError(f"Evaluation shard size mismatch: {file}")
            if cache.get(shard["file"]) != fingerprint:
                if file_sha256(file) != shard["sha256"]:
                    raise ValueError(f"Evaluation shard checksum mismatch: {file}")
                cache[shard["file"]] = fingerprint
    cache_path.write_text(json.dumps(cache, indent=2) + "\n")
    print(f"Evaluation data ready: {', '.join(names)} ({root})", flush=True)
    return manifest, config


def run_benchmark(name, data_root, manifest, out_dir, config, base_url, model_name, concurrency, dp, batch_size, limit, resume):
    path = out_dir / f"{name}_predictions.jsonl"
    previous = []
    if path.exists() and path.stat().st_size:
        if not resume:
            raise ValueError(f"Results already exist: {path}; use --resume or another --out-dir")
        previous = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    completed = {record["sample_id"] for record in previous}
    rows = (row for row in iter_rows(name, data_root, manifest, limit) if row["sample_id"] not in completed)
    engine = OnlineEngine(config, api_url(base_url), concurrency, dp, model_name)
    count = len(previous)
    with path.open("a" if resume else "w") as output:
        while batch := list(islice(rows, batch_size)):
            pending = []
            for row in batch:
                if row.get("_skip"):
                    record = {"id": row["id"], "sample_id": row["sample_id"], "source_split": row["source_split"],
                              "skipped": True, "skip_reason": row["_skip"]}
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                else:
                    pending.append(row)
            output.flush()
            if pending:
                requests = [(row["_images"], row["_prompt"]) for row in pending]

                def on_done(index, response):
                    nonlocal count
                    record = score_record(name, pending[index], response)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    count += 1

                engine.generate_stream(requests, on_done)
            print(f"{name}: {count} results saved", flush=True)
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({row["sample_id"] for row in records}) != len(records):
        raise ValueError(f"Duplicate result IDs: {path}")
    return summarize(name, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="xiongyq/SCOPED-8B", help="HF model ID or local export; defaults to the released xiongyq/SCOPED-8B")
    parser.add_argument("--datasets", default="all", help="Comma-separated names or all")
    parser.add_argument("--eval-root", type=Path, default=storage_root() / "data/hf_eval")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--online", action="store_true", help="Compatibility flag; this runner always uses the online API")
    parser.add_argument("--base-url", help="Use an existing OpenAI-compatible server without starting one")
    parser.add_argument("--served-model-name", default="eval-model")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--serve-tp", type=int, default=4)
    parser.add_argument("--serve-dp", type=int, default=1)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--batch-size", type=int, help="Override per-benchmark request window (default 100)")
    parser.add_argument("--max-model-len", type=int, help="Override per-benchmark context limit")
    parser.add_argument("--max-num-seqs", type=int, help="Override per-benchmark server sequence limit")
    parser.add_argument("--max-num-batched-tokens", type=int, help="Override per-benchmark server token budget")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, help="Optional override for smoke checks; defaults preserve original per-benchmark settings")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-data", action="store_true", help="Prepare and inspect prompts without starting a server")
    parser.add_argument("--print-config", action="store_true", help="Print resolved sampling and runtime settings without downloading data or starting inference")
    args = parser.parse_args()
    names = list(CONFIG_NAMES) if args.datasets == "all" else [name.strip() for name in args.datasets.split(",")]
    if not names or any(name not in CONFIG_NAMES for name in names) or len(set(names)) != len(names):
        parser.error(f"Choose from: {','.join(CONFIG_NAMES)}")
    if min(args.serve_tp, args.serve_dp, args.concurrency, args.startup_timeout) < 1:
        parser.error("Parallel sizes, concurrency, batch size and limits must be positive")
    for key in ("max_tokens", "batch_size", "max_model_len", "max_num_seqs", "max_num_batched_tokens"):
        if getattr(args, key) is not None and getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    configs = load_configs(names, args)
    requested_runtime = {"gpus": args.gpus, "tp": args.serve_tp, "dp": args.serve_dp,
                         "concurrency": args.concurrency, "base_url": args.base_url,
                         "server": {name: server_settings(configs[name], args) for name in names}}
    if args.print_config:
        print(json.dumps({"model": args.model, "generation": configs, "runtime": requested_runtime}, indent=2))
        return 0
    manifest, data_config = prepare_data(args.eval_root, names, args.offline)
    if args.check_data:
        for name in names:
            rows = list(iter_rows(name, args.eval_root, manifest, limit=max(1, args.limit)))
            available = sum(not row.get("_skip") for row in rows)
            print(f"{name}: inspected {len(rows)} rows, {available} available; original split={rows[0]['source_split']}", flush=True)
        return 0
    out_dir = args.out_dir or storage_root() / "outputs/evaluation" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {"model": args.model, "dataset": data_config, "datasets": names, "generation": {name: configs[name] for name in names},
                  "limit": args.limit, "served_model_name": args.served_model_name,
                  "runtime": requested_runtime}
    config_path = out_dir / "run_config.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != run_config:
            raise ValueError("Output directory belongs to another run; choose a new path or resume the same configuration")
    else:
        if any(out_dir.glob("*_predictions.jsonl")):
            raise ValueError("Result files exist without matching run_config.json")
        config_path.write_text(json.dumps(run_config, indent=2) + "\n")
    process, base_url, active_settings = None, args.base_url, None
    summary = {}
    try:
        if base_url:
            check_server(base_url, args.served_model_name)
        else:
            import torch
            gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
            if len(gpu_ids) < args.serve_tp * args.serve_dp or torch.cuda.device_count() < args.serve_tp * args.serve_dp:
                raise ValueError("Not enough visible GPUs for serve-tp * serve-dp")
        for name in names:
            settings = requested_runtime["server"][name]
            if not args.base_url and settings != active_settings:
                stop_server(process)
                process, base_url = None, None
                process, base_url = start_server(args.model, args.served_model_name, args.port, args.serve_tp, args.serve_dp,
                                                args.gpus, settings, out_dir / f"{name}_vllm_serve.log", args.startup_timeout)
                active_settings = settings
            score = run_benchmark(name, args.eval_root, manifest, out_dir, configs[name], base_url,
                                  args.served_model_name, args.concurrency, args.serve_dp,
                                  configs[name]["batch_size"], args.limit, args.resume)
            summary[name] = score
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(f"{name}: {json.dumps(score)}", flush=True)
    finally:
        stop_server(process)
    print(f"Results saved under {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
