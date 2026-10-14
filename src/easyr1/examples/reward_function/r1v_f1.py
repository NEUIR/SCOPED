from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


REWARD_NAME = "r1v_f1"
REWARD_TYPE = "batch"


def extract_answer(response: str) -> str:
    content_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if content_match:
        return content_match.group(1).strip()
    return response.strip()


def normalize_text(text: str) -> str:
    def _bool_mapping(s: str) -> str:
        return {"true": "yes", "false": "no"}.get(s, s)

    text = text.strip().lower()
    text = _bool_mapping(text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_norm = normalize_text(prediction)
    gt_norm = normalize_text(ground_truth)

    if not pred_norm or not gt_norm:
        return 0.0

    pred_tokens = pred_norm.split()
    gt_tokens = gt_norm.split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        prediction = extract_answer(reward_input["response"])
        score = f1_score(prediction, reward_input["ground_truth"])
        scores.append(
            {
                "overall": score,
                "f1": score,
            }
        )
    return scores
