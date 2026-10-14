#!/usr/bin/env python3
"""Fail early on incompatible backend versions or insufficient visible GPUs."""
import argparse
import importlib.metadata as metadata
import math

from packaging.version import Version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("rl", "distillation"), required=True)
    parser.add_argument("--gpus", type=float, default=0)
    parser.add_argument("--attention", choices=("flash_attention_2", "sdpa"), default="flash_attention_2")
    args = parser.parse_args()
    errors = []
    versions = {}
    for name in ("torch", "vllm", "transformers", "flash-attn", "datasets", "huggingface_hub"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            if name != "flash-attn" or args.attention == "flash_attention_2":
                errors.append(f"Missing dependency: {name}")
    if "vllm" in versions:
        version = Version(versions["vllm"])
        if args.stage == "rl" and version.base_version != "0.11.0":
            errors.append("EasyR1 requires the tested vLLM 0.11.0 runtime; activate the RL environment")
        if args.stage == "distillation" and not Version("0.18") <= version < Version("0.25"):
            errors.append("The modified veRL backend requires vLLM >=0.18,<0.25")
    if "transformers" in versions:
        version = Version(versions["transformers"])
        if args.stage == "rl" and not Version("4.57") <= version < Version("5"):
            errors.append("EasyR1 requires Transformers >=4.57,<5")
        if args.stage == "distillation" and not Version("5.5.3") <= version < Version("5.11"):
            errors.append("The modified veRL backend requires Transformers >=5.5.3,<5.11")
    if "torch" in versions:
        import torch
        count = torch.cuda.device_count()
        print(f"Visible GPUs: {count}; required: {math.ceil(args.gpus)}", flush=True)
        if count < math.ceil(args.gpus):
            errors.append("Insufficient visible GPUs; adjust CUDA_VISIBLE_DEVICES and NGPUS_PER_NODE")
    print("Runtime: " + ", ".join(f"{k}={v}" for k, v in versions.items()), flush=True)
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
