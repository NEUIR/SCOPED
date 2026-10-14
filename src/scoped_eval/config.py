"""Resolve the experiment's batch-script settings for online evaluation."""
import json
from pathlib import Path


def load_configs(names, args):
    configs = json.loads((Path(__file__).parent / "generation.json").read_text())
    for name in names:
        config = configs[name]
        for key in ("max_tokens", "max_model_len", "max_num_seqs", "max_num_batched_tokens", "batch_size"):
            value = getattr(args, key, None)
            if value is not None:
                config[key] = value
        # OCR disables Qwen3 thinking in the original recognition runner.
        is_qwen3 = "qwen3" in args.model.lower() or args.model.lower() in {"xiongyq/scoped-4b", "xiongyq/scoped-8b"}
        if name == "ccocrv2" and is_qwen3:
            config["chat_template_kwargs"] = {"enable_thinking": False}
    return {name: configs[name] for name in names}


def server_settings(config, args):
    return {
        "max_model_len": config["max_model_len"],
        "max_num_seqs": config["max_num_seqs"],
        "max_num_batched_tokens": config["max_num_batched_tokens"],
        "dtype": config["dtype"],
        "limit_images": config["limit_images"],
        "gpu_memory": args.gpu_memory_utilization,
        "seed": config["sampling"]["seed"],
    }
