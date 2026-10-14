#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import shlex
import subprocess
import sys
from storage import output_root

ROOT = Path(__file__).resolve().parents[2]


def complete_actor(path):
    if not (path / "huggingface/config.json").is_file():
        return False
    groups = {}
    for file in path.glob("model_world_size_*_rank_*.pt"):
        match = re.fullmatch(r"model_world_size_(\d+)_rank_(\d+)\.pt", file.name)
        if match and file.stat().st_size:
            groups.setdefault(int(match[1]), set()).add(int(match[2]))
    return any(ranks == set(range(size)) for size, ranks in groups.items())


def main():
    parser = argparse.ArgumentParser(description="Export selected or latest complete specialist checkpoints")
    parser.add_argument("--checkpoint-root", type=Path, default=output_root() / "rl")
    parser.add_argument("--steps", nargs="*", default=[])
    parser.add_argument("--env-file", type=Path, default=output_root() / "teachers.env")
    args = parser.parse_args()
    requested = {}
    for item in args.steps:
        capability, value = item.split("=", 1)
        if capability not in ("capability1", "capability2", "capability3", "capability4") or not value.isdecimal():
            parser.error(f"Invalid step selection: {item}")
        requested[capability] = int(value)
    selected = {}
    for index in range(1, 5):
        capability = f"capability{index}"
        candidates = []
        for path in (args.checkpoint_root / capability).glob("global_step_*"):
            match = re.fullmatch(r"global_step_(\d+)", path.name)
            if match and complete_actor(path / "actor"):
                step = int(match[1])
                if capability not in requested or step == requested[capability]:
                    candidates.append((step, path / "actor"))
        if not candidates:
            parser.error(f"No complete checkpoint found for {capability}")
        selected[capability] = max(candidates)[1].resolve()
    lines = []
    for index, (capability, actor) in enumerate(selected.items(), 1):
        subprocess.run([sys.executable, str(ROOT / "src/easyr1/scripts/model_merger.py"), "--local_dir", str(actor)], check=True)
        model = actor / "huggingface"
        if not any(model.glob("*.safetensors")):
            raise RuntimeError(f"Exported model weights missing: {model}")
        lines.append(f"export CAPABILITY{index}_TEACHER={shlex.quote(str(model))}")
    args.env_file.parent.mkdir(parents=True, exist_ok=True)
    args.env_file.write_text("\n".join(lines) + "\n")
    print(f"Teacher paths ready: {args.env_file}", flush=True)


if __name__ == "__main__":
    main()
