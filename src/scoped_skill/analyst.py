"""Analyze correct and incorrect trajectories separately with the same expert."""
from collections import defaultdict
import json
from pathlib import Path
import random

from .common import atomic_text, completed, digest, mark_completed, read_jsonl, signature
from .json_utils import extract_json
from .prompts import ERROR_ANALYST, SUCCESS_ANALYST
from .rollout import split_budget


def select_traces(rows, cap, seed):
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["subtype"], " ".join(row["question"].split()),
               " ".join(row["trajectory"].split()), row["ground_truth"])
        if key not in seen:
            seen.add(key)
            groups[row["subtype"]].append(row)
    selected = []
    rng = random.Random(seed)
    quotas = split_budget(cap, {subtype: len(bucket) for subtype, bucket in groups.items()})
    for subtype in sorted(groups):
        rng.shuffle(groups[subtype])
        selected.extend(groups[subtype][:quotas[subtype]])
    return selected


def analyze(traces, work_dir, capability, config, engine, model_identity, resume=False):
    rows = read_jsonl(traces)
    outputs = []
    for kind in ("success", "failure"):
        output = Path(work_dir) / "patches" / f"{capability}_{kind}.jsonl"
        template = SUCCESS_ANALYST if kind == "success" else ERROR_ANALYST
        expected = signature(stage="analyst-v2-paper", input=digest(traces), config=config,
                             template=template, model=model_identity, kind=kind)
        outputs.append(output)
        if resume and completed(output, expected):
            continue
        selected = select_traces([r for r in rows if r["is_correct"] == (kind == "success")],
                                 config[f"{kind}_analyst_cap"], config["seed"])
        prompts = [template.format(domain=config["capabilities"][capability]["description"],
                                   question=row["question"], trajectory=row["trajectory"],
                                   prediction=row["final_answer"], ground_truth=row["ground_truth"])
                   for row in selected]
        parsed_count = 0
        with atomic_text(output) as handle:
            for start in range(0, len(selected), config["rollout_batch"]):
                batch = selected[start:start + config["rollout_batch"]]
                texts = engine.generate_text(prompts[start:start + len(batch)], {
                    "temperature": config["analyst_temperature"], "top_p": config["analyst_top_p"],
                    "max_tokens": config["analyst_max_tokens"]})
                if len(texts) != len(batch):
                    raise RuntimeError("Incomplete analyst batch")
                for row, text in zip(batch, texts):
                    parsed = extract_json(text)
                    if not isinstance(parsed, dict):
                        parsed = None
                    parsed_count += int(parsed is not None)
                    record = {key: row[key] for key in ("capability", "subtype", "section", "sample_id", "question")}
                    record.update(parsed=parsed, raw=text)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if selected and not parsed_count:
            raise ValueError(f"No valid {kind} analyses; increase analyst_max_tokens or inspect the expert output")
        mark_completed(output, expected)
        print(f"[analyst] {capability}/{kind}: {parsed_count}/{len(selected)} parsed", flush=True)
    return outputs
