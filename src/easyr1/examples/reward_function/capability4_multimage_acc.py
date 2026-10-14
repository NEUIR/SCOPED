from __future__ import annotations

import re
import string
from typing import Any


REWARD_NAME = "capability4_multimage_acc"
REWARD_TYPE = "batch"


OPTION_RE = re.compile(r"^\s*([A-Z])\s*[\.\)]?\s*$", re.IGNORECASE)
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


RELATION_PATTERNS: dict[str, tuple[str, ...]] = {
    "left": ("left of", "to the left", "on the left", "left side", "left"),
    "right": ("right of", "to the right", "on the right", "right side", "right"),
    "above": ("above", "higher than", "over", "on top of", "superior to", "elevated"),
    "below": ("below", "lower than", "under", "beneath", "shorter than"),
    "front": ("in front of", "closer", "nearest", "shortest distance", "foreground"),
    "back": ("behind", "further back", "farther back", "more distant", "rear", "greatest distance"),
    "larger": ("larger", "bigger", "greater size", "comes out on top"),
    "smaller": ("smaller", "lesser size"),
    "taller": ("taller", "greater height", "higher height"),
    "shorter": ("shorter", "lesser height"),
    "wider": ("wider", "greater width"),
    "narrower": ("narrower", "smaller width", "lesser width"),
    "touching": ("touch", "touches", "contact", "overlap", "intersect"),
    "not_touching": ("not touch", "no contact", "not in contact", "without contact"),
    "inside": ("inside", "within", "contained"),
    "outside": ("outside", "not inside"),
}

YES_WORDS = {
    "yes",
    "yeah",
    "yep",
    "true",
    "correct",
    "affirmative",
    "indeed",
    "absolutely",
    "definitely",
    "certainly",
}
NO_WORDS = {
    "no",
    "not",
    "false",
    "incorrect",
    "unlikely",
    "improbable",
    "cannot",
    "can't",
}


def extract_answer(response: str) -> str:
    matches = ANSWER_TAG_RE.findall(str(response))
    if matches:
        return matches[-1].strip()
    return str(response).strip()


def has_answer_tag(response: str) -> bool:
    return bool(ANSWER_TAG_RE.search(str(response)))


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace("n't", " not")
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def contains_acc(prediction: str, ground_truth: str) -> float:
    pred_norm = normalize_text(prediction)
    gt_norm = normalize_text(ground_truth)
    if not pred_norm or not gt_norm:
        return 0.0
    return 1.0 if gt_norm in pred_norm else 0.0


def option_letter(text: str) -> str | None:
    answer = extract_answer(text)
    match = OPTION_RE.match(answer)
    if match:
        return match.group(1).upper()

    match = re.match(r"^\s*([A-Z])\s*[\.\)]\s+\S+", answer, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-Z])\b", answer, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def image_indices(text: str) -> set[int]:
    answer = extract_answer(text)
    return {int(idx) for idx in re.findall(r"\bimage\s*(\d+)\b", answer, flags=re.IGNORECASE)}


def relation_labels(text: str) -> set[str]:
    norm = normalize_text(extract_answer(text))
    labels = set()
    for label, patterns in RELATION_PATTERNS.items():
        for pattern in patterns:
            if normalize_text(pattern) in norm:
                labels.add(label)
                break

    if "not_touching" in labels:
        labels.discard("touching")
    if "outside" in labels:
        labels.discard("inside")
    return labels


def yes_no_label(text: str) -> str | None:
    norm = normalize_text(extract_answer(text))
    tokens = set(norm.split())
    has_no = bool(tokens & NO_WORDS)
    has_yes = bool(tokens & YES_WORDS)
    if has_no:
        return "no"
    if has_yes:
        return "yes"
    return None


def subset_acc(pred: set[Any], gt: set[Any]) -> float:
    if not gt:
        return 0.0
    return 1.0 if gt.issubset(pred) else 0.0


def compute_one(response: str, ground_truth: str) -> dict[str, float]:
    format_score = 1.0 if has_answer_tag(response) else 0.0
    prediction = extract_answer(response)
    gt = str(ground_truth).strip()
    exact = 1.0 if normalize_text(prediction) == normalize_text(gt) else 0.0

    gt_option = option_letter(gt)
    if gt_option is not None:
        pred_option = option_letter(prediction)
        acc = 1.0 if pred_option == gt_option else 0.0
        final_acc = acc if format_score > 0 else 0.0
        return {
            "overall": final_acc,
            "acc": final_acc,
            "accuracy": final_acc,
            "format": format_score,
            "option": final_acc,
            "exact": exact if format_score > 0 else 0.0,
            "contains": final_acc,
        }

    gt_images = image_indices(gt)
    if gt_images:
        acc = subset_acc(image_indices(prediction), gt_images)
        final_acc = acc if format_score > 0 else 0.0
        return {
            "overall": final_acc,
            "acc": final_acc,
            "accuracy": final_acc,
            "format": format_score,
            "image_set_acc": final_acc,
            "exact": exact if format_score > 0 else 0.0,
            "contains": final_acc,
        }

    gt_yes_no = yes_no_label(gt)
    if gt_yes_no is not None:
        pred_yes_no = yes_no_label(prediction)
        acc = 1.0 if pred_yes_no == gt_yes_no else 0.0
        final_acc = acc if format_score > 0 else 0.0
        return {
            "overall": final_acc,
            "acc": final_acc,
            "accuracy": final_acc,
            "format": format_score,
            "yes_no": final_acc,
            "exact": exact if format_score > 0 else 0.0,
            "contains": final_acc,
        }

    gt_relations = relation_labels(gt)
    if gt_relations:
        acc = subset_acc(relation_labels(prediction), gt_relations)
        final_acc = acc if format_score > 0 else 0.0
        return {
            "overall": final_acc,
            "acc": final_acc,
            "accuracy": final_acc,
            "format": format_score,
            "relation_acc": final_acc,
            "exact": exact if format_score > 0 else 0.0,
            "contains": final_acc,
        }

    acc = contains_acc(prediction, gt)
    final_acc = acc if format_score > 0 else 0.0
    return {
        "overall": final_acc,
        "acc": final_acc,
        "accuracy": final_acc,
        "format": format_score,
        "exact": exact if format_score > 0 else 0.0,
        "contains": final_acc,
    }


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    return [compute_one(item["response"], item["ground_truth"]) for item in reward_inputs]
