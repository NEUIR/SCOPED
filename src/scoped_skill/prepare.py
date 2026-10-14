"""Read the shared image-embedded Parquet TRAIN split without source folders."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from .common import atomic_text, completed, digest, mark_completed, signature, write_json

# Source markers are training-data provenance, not benchmark-specific rules.
MARKERS = {
    "counting_clevr": "train_counting_clevr", "detection_v3det": "train_detection_v3det",
    "detection_object365": "train_detection_object365", "grounding_dcube": "train_grounding_dcube",
    "pixmo": "pixmo_images", "chart_chartqapro": "train_chart_chartqapro",
    "chart_chartx": "train_chart_chartx", "chart_tablevqa": "train_chart_tablevqa",
    "chart_virl39k": "train_chart_virl39k", "slidevqa": "slidevqa_images",
    "ocr_llavaov": "train_ocr_llavaov", "ocr_estvqa": "train_ocr_estvqa",
    "puzzle_puzzlevqa": "train_puzzle_puzzlevqa", "wemath2.0_pro": "images/wemathpro",
    "math_mmk12": "train_math_mmk12", "m4_ScanQA": "m4_instruct_data/scannet_frames_25k",
    "m4_nuscenes": "m4_instruct_data/nuscenes", "m4_Spot-the-Diff": "m4_instruct_data/spot-the-diff",
    "mantis_spot-the-diff": "mantis_instruct/spot-the-diff", "mantis_nlvr2": "mantis_instruct/nlvr2",
    "mantis_nextqa": "mantis_instruct/nextqa", "mantis_star": "mantis_instruct/star",
    "mantis_multi_vqa": "mantis_instruct/multi_vqa", "mantis_birds-to-words": "mantis_instruct/birds-to-words",
}


def task_type(row):
    if row.get("source"):
        return str(row["source"])
    paths = row.get("image_paths") or [image.get("path", "") for image in row.get("images", [])]
    text = " ".join(str(p).replace("\\", "/") for p in paths)
    matches = [subtype for subtype, marker in MARKERS.items() if marker in text]
    if len(matches) > 1:
        raise ValueError(f"Multiple training subtypes: {matches}")
    return matches[0] if matches else "other"


def section_of(config, capability, subtype):
    for section, subtypes in config["capabilities"][capability]["sections"].items():
        if subtype in subtypes:
            return section
    return "Other tasks"


def materialize_images(row, image_dir):
    paths = []
    for image in row.get("images") or []:
        if not isinstance(image, dict) or not image.get("bytes"):
            raise ValueError("Skill extraction expects image-embedded SCOPED Parquet rows")
        payload = image["bytes"]
        path = image_dir / (hashlib.sha256(payload).hexdigest() + ".image")
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(payload)
            temporary.replace(path)
        paths.append(str(path.resolve()))
    if not paths:
        raise ValueError(f"No images in {row.get('sample_id')}")
    return paths


def prepare(data_dir, work_dir, capability, config, resume=False):
    source_dir = Path(data_dir) / capability / "train"
    sources = sorted(source_dir.glob("*.parquet"))
    if not sources:
        raise ValueError(f"No training Parquet files: {source_dir}")
    output = Path(work_dir) / "data" / f"{capability}.jsonl"
    expected = signature(stage="prepare-v1", capability=capability,
                         sources={p.name: digest(p) for p in sources}, sections=config["capabilities"][capability])
    manifest = output.with_suffix(".manifest.json")
    if resume and completed(output, expected) and manifest.is_file():
        return output
    counts = Counter()
    with atomic_text(output) as handle:
        for source in sources:
            for batch in pq.ParquetFile(source).iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    if row.get("capability") != capability or row.get("extra_info", {}).get("split") != "train":
                        raise ValueError("Extraction accepts only the requested capability's train rows")
                    subtype = task_type(row)
                    query = row.get("problem")
                    if not query or row.get("answer") is None:
                        raise ValueError(f"Missing problem/answer: {row.get('sample_id')}")
                    item = {"sample_id": row["sample_id"], "capability": capability, "subtype": subtype,
                            "section": section_of(config, capability, subtype), "query": query,
                            "answer": row["answer"],
                            "images": materialize_images(row, Path(work_dir) / "images" / capability)}
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                    counts[subtype] += 1
    if not counts:
        raise ValueError(f"Empty training split: {source_dir}")
    write_json(manifest, {"capability": capability, "split": "train", "counts": counts,
                          "input_signature": expected, "rows": sum(counts.values())})
    mark_completed(output, expected)
    print(f"[prepare] {capability}: {sum(counts.values())} train rows", flush=True)
    return output
