"""Sample each expert on its own training questions and judge each completion."""
from collections import defaultdict
import json
from pathlib import Path
import random

from .common import atomic_text, completed, digest, mark_completed, read_jsonl, signature
from .reward import reward_identity, score_trajectory


def split_budget(budget, capacities):
    result = {key: 0 for key in capacities}
    pending = [key for key in sorted(capacities) if capacities[key] > 0]
    while budget > 0 and pending:
        share = max(1, budget // len(pending))
        for key in list(pending):
            take = min(share, capacities[key] - result[key], budget)
            result[key] += take
            budget -= take
            if result[key] == capacities[key]:
                pending.remove(key)
    return result


def select_questions(rows, config):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["section"]][row["subtype"]].append(row)
    budget = len(rows) if config["sample_per_capability"] == -1 else config["sample_per_capability"]
    sections = split_budget(budget, {
        section: sum(len(bucket) for bucket in subtypes.values()) for section, subtypes in groups.items()})
    selected = []
    rng = random.Random(config["seed"])
    for section in sorted(groups):
        subtypes = groups[section]
        quotas = split_budget(sections[section], {key: len(bucket) for key, bucket in subtypes.items()})
        for subtype in sorted(subtypes):
            bucket = list(subtypes[subtype])
            rng.shuffle(bucket)
            k = config["rollout_k_small"] if len(bucket) < config["small_subtype_rows"] else config["rollout_k"]
            selected.extend((row, k) for row in bucket[:quotas[subtype]])
    return selected


def rollout(prepared, work_dir, capability, config, engine, model_identity, resume=False):
    output = Path(work_dir) / "traces" / f"{capability}.jsonl"
    expected = signature(stage="rollout-v2-paper", input=digest(prepared), config=config,
                         model=model_identity, reward=reward_identity(capability))
    if resume and completed(output, expected):
        print(f"[rollout] {capability}: verified cached traces", flush=True)
        return output
    questions = select_questions(read_jsonl(prepared), config)
    by_k = defaultdict(list)
    for row, k in questions:
        by_k[k].append(row)
    correct = written = 0
    with atomic_text(output) as handle:
        for k, rows in sorted(by_k.items()):
            sampling = {"n": k, "temperature": config["rollout_temperature"],
                        "top_p": config["rollout_top_p"], "max_tokens": config["rollout_max_tokens"]}
            for start in range(0, len(rows), config["rollout_batch"]):
                batch = rows[start:start + config["rollout_batch"]]
                generated = engine.generate_vision(batch, sampling)
                if len(generated) != len(batch) or any(len(items) != k for items in generated):
                    raise RuntimeError("Incomplete rollout batch; partial results will not be reused")
                for row, completions in zip(batch, generated):
                    for index, text in enumerate(completions):
                        score, answer = score_trajectory(capability, text, row["answer"])
                        threshold = config.get("reward_threshold", 0.5)
                        ok = score >= threshold
                        record = {"capability": capability, "sample_id": row["sample_id"],
                                  "completion_index": index, "section": row["section"],
                                  "subtype": row["subtype"], "question": row["query"],
                                  "trajectory": text, "final_answer": answer,
                                  "ground_truth": row["answer"], "reward": score,
                                  "reward_threshold": threshold, "is_correct": ok}
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        written += 1
                        correct += int(ok)
                print(f"[rollout] {capability}: {written} traces", flush=True)
    if not written:
        raise ValueError("No rollout traces generated")
    mark_completed(output, expected)
    print(f"[rollout] {capability}: {len(questions)} questions, reward_success_rate={correct / written:.4f}", flush=True)
    return output
