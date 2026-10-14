"""Run the four extraction steps for one capability expert."""
import argparse
import json
from pathlib import Path

from .common import CAPABILITIES, load_config, write_json
from .prepare import prepare


def model_identity(model):
    path = Path(model).expanduser()
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix in (".json", ".safetensors", ".bin"))
        if not files:
            raise ValueError(f"Not an exported HF model: {path}")
        return {"path": str(path.resolve()), "files": {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in files}}
    from huggingface_hub import HfApi
    info = HfApi().model_info(model)
    return {"repo_id": model, "revision": info.sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capability", required=True, choices=CAPABILITIES)
    parser.add_argument("--model", required=True, help="This capability's exported RL expert")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--sample", type=int)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--server-tp", type=int)
    parser.add_argument("--base-url", help="Reuse an endpoint serving --capability; do not manage its process")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.sample is not None:
        if args.sample != -1 and args.sample < 1:
            parser.error("--sample must be positive or -1 for all training rows")
        config["sample_per_capability"] = args.sample
    if args.server_tp is not None:
        if args.server_tp < 1:
            parser.error("--server-tp must be positive")
        config["server_tp"] = args.server_tp
    if args.dry_run:
        print(json.dumps({"capability": args.capability, "expert": args.model,
                          "train": str(args.data_dir / args.capability / "train"),
                          "steps": ["prepare", "rollout", "analyst", "author"],
                          "skill": str(args.work_dir / "skills" / args.capability / "SKILL.md")}, indent=2))
        return
    identity = model_identity(args.model)
    prepared = prepare(args.data_dir, args.work_dir, args.capability, config, args.resume)
    from .analyst import analyze
    from .author import author
    from .rollout import rollout
    from .server import expert_server
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=identity.get("revision"))
    with expert_server(args.model, args.capability, args.work_dir, config, args.port, args.base_url,
                       revision=identity.get("revision")) as engine:
        engine.tokenizer = tokenizer
        traces = rollout(prepared, args.work_dir, args.capability, config, engine, identity, args.resume)
        patches = analyze(traces, args.work_dir, args.capability, config, engine, identity, args.resume)
        skill = author(patches, args.work_dir, args.capability, config, engine, identity, args.resume)
    write_json(args.work_dir / "runs" / f"{args.capability}.json", {
        "capability": args.capability, "model": identity, "training_rows": str(prepared),
        "traces": str(traces), "patches": [str(p) for p in patches], "skill": str(skill), "config": config})


if __name__ == "__main__":
    main()
