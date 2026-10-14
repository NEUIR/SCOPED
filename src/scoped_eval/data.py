"""Adapt the shared HF evaluation export to the original baseline prompts."""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import pyarrow.parquet as pq

from .protocol import VANILLA_MLLM, choice_letters

CONFIG_NAMES = {
    "ccocrv2": "CCOCRV2", "chartqa": "ChartQA", "countbenchqa": "CountBenchQA",
    "mmsi-bench": "MMSI-Bench", "muirbench": "MUIRBENCH", "pixmo-count": "pixmo-count",
    "visualpuzzles": "VisualPuzzles", "wemath": "We-Math",
}


def adapt_row(name, raw):
    metadata = json.loads(raw["metadata"])
    row = dict(metadata)
    row.update(id=raw["id"], sample_id=raw["sample_id"], source_split=raw["source_split"])
    if not raw["image_available"]:
        row["_skip"] = raw["skip_reason"] or "missing_local_image"
        return row
    images = []
    for field in raw["images"]:
        with Image.open(BytesIO(field["bytes"])) as image:
            images.append(image.convert("RGB"))
    row["_images"] = images
    question = raw["question"]
    answers = raw["answers"]
    if name == "countbenchqa":
        row.update(_question=question.strip(), _gt_count=int(answers[0]))
        row["_prompt"] = row["_question"] + "\n" + VANILLA_MLLM
    elif name == "pixmo-count":
        row["count"] = int(answers[0])
        row["_prompt"] = f"How many {metadata['label']} are there in the image?\n{VANILLA_MLLM}"
    elif name == "chartqa":
        row.update(query=question, _labels=answers)
        row["_prompt"] = question.strip() + "\n" + VANILLA_MLLM
    elif name == "ccocrv2":
        if raw["task"] != "recognition":
            raise ValueError("SCOPED evaluates only CCOCRV2 recognition")
        row["_prompt"] = question
        row["_gt_text"] = answers[0]
    elif name == "wemath":
        question = question.replace("<image>", "").strip()
        option = str(metadata.get("option") or "").strip()
        row["_prompt"] = f"Question: {question}" + (f"\nOptions: {option}" if option else "")
        row["_prompt"] += "\nPlease reason step by step, and put only the final option letter within <answer></answer>."
        row["answer"] = answers[0]
    elif name == "visualpuzzles":
        options = [str(option).strip() for option in raw["options"]]
        row["_options"] = [option for option in options if option]
        lines = [question.strip()]
        if row["_options"]:
            lines.append("Options:")
            lines.extend(f"{letter}. {option}" for letter, option in zip(choice_letters(len(row["_options"])), row["_options"]))
        lines.append(VANILLA_MLLM)
        row["_prompt"] = "\n".join(line for line in lines if line)
        row["answer"] = answers[0]
    elif name == "mmsi-bench":
        row["_prompt"] = question.strip() + "\nPlease reason step by step, and put only the final option letter within <answer></answer>."
        row["answer"] = answers[0]
    elif name == "muirbench":
        image_index = 0

        def replace_placeholders(text):
            nonlocal image_index
            text = str(text or "").strip()
            while "<image>" in text:
                image_index += 1
                text = text.replace("<image>", f"Image {image_index}", 1)
            return text

        question = replace_placeholders(question)
        options = raw["options"]
        option_lines = [f"{letter}. {replace_placeholders(option)}" for letter, option in zip(choice_letters(len(options)), options)]
        n = len(images)
        row["_prompt"] = (
            f"You are given {n} images in order, indexed as Image 1 to Image {n}. "
            "In the question and options, each image reference points to the corresponding image by index.\n"
            f"Question: {question}\nOptions:\n" + "\n".join(option_lines)
            + "\nPlease reason step by step, and put only the final option letter within <answer></answer>."
        )
        row["answer"] = answers[0]
    else:
        raise ValueError(f"Unknown benchmark: {name}")
    return row


def iter_rows(name, root, manifest, limit=-1):
    spec = manifest["benchmarks"][CONFIG_NAMES[name]]
    count = 0
    for shard in spec["shards"]:
        path = Path(root) / shard["file"]
        if path.stat().st_size != shard["bytes"]:
            raise ValueError(f"Evaluation shard size mismatch: {path}")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
            for raw in batch.to_pylist():
                if limit > 0 and count >= limit:
                    return
                count += 1
                yield adapt_row(name, raw)
