from __future__ import annotations

import ast
import json
import math
import re
import string
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any


REWARD_NAME = "orsta_perception"
REWARD_TYPE = "batch"


def extract_answer(response: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return response.strip()


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = {"true": "yes", "false": "no"}.get(text, text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def decimal_or_none(text: str) -> Decimal | None:
    cleaned = str(text).strip().replace(",", "")
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    try:
        return Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None


def is_numeric_answer(text: str) -> bool:
    return decimal_or_none(text) is not None


def text_accuracy(response: str, ground_truth: str) -> float:
    pred = extract_answer(response)
    gt = str(ground_truth).strip()
    pred_norm = normalize_text(pred)
    gt_norm = normalize_text(gt)
    if not pred_norm or not gt_norm:
        return 0.0

    if is_numeric_answer(gt):
        pred_number_match = re.search(r"-?\d+(?:\.\d+)?%?", pred)
        if pred_number_match is None:
            return 0.0
        pred_num = decimal_or_none(pred_number_match.group(0))
        gt_num = decimal_or_none(gt)
        return 1.0 if pred_num is not None and gt_num is not None and pred_num == gt_num else 0.0

    if gt_norm in {"yes", "no"}:
        return 1.0 if re.search(rf"\b{re.escape(gt_norm)}\b", pred_norm) else 0.0

    if len(gt_norm.split()) == 1:
        return 1.0 if re.search(rf"\b{re.escape(gt_norm)}\b", pred_norm) else 0.0

    return 1.0 if gt_norm in pred_norm else 0.0


def text_f1(response: str, ground_truth: str) -> float:
    pred = extract_answer(response)
    pred_norm = normalize_text(pred)
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


def parse_literal_list(text: str) -> list[Any] | None:
    text = text.strip()
    candidates = [text]
    list_match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if list_match:
        candidates.append(list_match.group(0))

    for candidate in candidates:
        cleaned = re.sub(r"^```(?:json|python)?\s*|\s*```$", "", candidate.strip(), flags=re.IGNORECASE)
        for parser in (ast.literal_eval, json.loads):
            try:
                value = parser(cleaned)
            except Exception:
                continue
            if isinstance(value, list):
                return value
    return None


def normalize_box_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    bbox = item.get("bbox_2d", item.get("bbox", item.get("box")))
    label = item.get("label", item.get("class", item.get("category")))
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or label is None:
        return None

    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None

    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return {"bbox_2d": [x1, y1, x2, y2], "label": str(label)}


def parse_boxes(text: str) -> list[dict[str, Any]] | None:
    value = parse_literal_list(text)
    if value is None:
        return None

    boxes = []
    for item in value:
        box = normalize_box_item(item)
        if box is None:
            return None
        boxes.append(box)
    return boxes


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(box_a: list[float], box_b: list[float]) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = box_area(box_a)
    area_b = box_area(box_b)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def detection_f1_reward(
    pred_boxes: list[dict[str, Any]],
    gt_boxes: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    if not pred_boxes and not gt_boxes:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0, "mean_iou": 1.0, "tp": 0.0}
    if not pred_boxes or not gt_boxes:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "mean_iou": 0.0, "tp": 0.0}

    pairs = []
    for pred_idx, pred in enumerate(pred_boxes):
        pred_label = normalize_text(pred["label"])
        for gt_idx, gt in enumerate(gt_boxes):
            if pred_label != normalize_text(gt["label"]):
                continue
            score = iou(pred["bbox_2d"], gt["bbox_2d"])
            if score >= iou_threshold:
                pairs.append((score, pred_idx, gt_idx))

    pairs.sort(reverse=True)
    used_pred = set()
    used_gt = set()
    matched_iou_sum = 0.0
    for score, pred_idx, gt_idx in pairs:
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matched_iou_sum += score

    tp = len(used_gt)
    precision = tp / len(pred_boxes) if pred_boxes else 0.0
    recall = tp / len(gt_boxes) if gt_boxes else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    mean_iou = matched_iou_sum / tp if tp > 0 else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "tp": float(tp),
    }


def best_match_mean_iou(pred_boxes: list[dict[str, Any]], gt_boxes: list[dict[str, Any]]) -> float:
    if not pred_boxes and not gt_boxes:
        return 1.0
    if not pred_boxes or not gt_boxes:
        return 0.0

    pairs = []
    for pred_idx, pred in enumerate(pred_boxes):
        pred_label = normalize_text(pred["label"])
        for gt_idx, gt in enumerate(gt_boxes):
            if pred_label != normalize_text(gt["label"]):
                continue
            pairs.append((iou(pred["bbox_2d"], gt["bbox_2d"]), pred_idx, gt_idx))

    pairs.sort(reverse=True)
    used_pred = set()
    used_gt = set()
    matched_iou_sum = 0.0
    for score, pred_idx, gt_idx in pairs:
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matched_iou_sum += score

    return matched_iou_sum / max(len(pred_boxes), len(gt_boxes))


def box_reward(
    response: str,
    ground_truth: str,
    format_weight: float,
    f1_50_weight: float,
    f1_75_weight: float,
    mean_iou_weight: float,
) -> dict[str, float]:
    gt_boxes = parse_boxes(ground_truth)
    pred_boxes = parse_boxes(extract_answer(response))

    if gt_boxes is None:
        return {
            "overall": 0.0,
            "format": 0.0,
            "accuracy": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mean_iou": 0.0,
            "iou50_tp": 0.0,
            "f1_50": 0.0,
            "f1_75": 0.0,
        }

    format_score = 1.0 if pred_boxes is not None else 0.0
    det_50 = detection_f1_reward(pred_boxes or [], gt_boxes, iou_threshold=0.5) if pred_boxes is not None else {
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "mean_iou": 0.0,
        "tp": 0.0,
    }
    det_75 = detection_f1_reward(pred_boxes or [], gt_boxes, iou_threshold=0.75) if pred_boxes is not None else {
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "mean_iou": 0.0,
        "tp": 0.0,
    }
    mean_iou_score = best_match_mean_iou(pred_boxes or [], gt_boxes) if pred_boxes is not None else 0.0
    f1_50_score = det_50["f1"]
    f1_75_score = det_75["f1"]
    overall = (
        format_weight * format_score
        + f1_50_weight * f1_50_score
        + f1_75_weight * f1_75_score
        + mean_iou_weight * mean_iou_score
    )
    return {
        "overall": overall,
        "format": format_score,
        "accuracy": f1_50_score,
        "f1": f1_50_score,
        "f1_50": f1_50_score,
        "f1_75": f1_75_score,
        "precision": det_50["precision"],
        "recall": det_50["recall"],
        "mean_iou": mean_iou_score,
        "iou50_tp": det_50["tp"],
    }

def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.1,
    f1_50_weight: float = 0.4,
    f1_75_weight: float = 0.3,
    mean_iou_weight: float = 0.2,
    progress: float | None = None,
    global_step: int | None = None,
    training_steps: int | None = None,
    is_validation: bool = False,
    **_unused_kwargs: Any,
) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        ground_truth = reward_input["ground_truth"]

        if parse_boxes(str(ground_truth)) is not None:
            scores.append(
                box_reward(
                    response,
                    ground_truth,
                    format_weight,
                    f1_50_weight,
                    f1_75_weight,
                    mean_iou_weight,
                )
            )
        else:
            f1 = text_f1(response, ground_truth)
            scores.append({"overall": f1, "format": 0.0, "giou": 0.0, "iou": 0.0, "accuracy": f1, "f1": f1})

    return scores
