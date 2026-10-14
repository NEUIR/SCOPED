from __future__ import annotations

import re
import string
from decimal import Decimal, InvalidOperation
from typing import Any

REWARD_NAME = "format-acc"
REWARD_TYPE = "batch"

def has_answer_tag(response: str) -> bool:
    """检查响应中是否包含 <answer> 标签"""
    return bool(re.search(r"<answer>.*?</answer>", str(response), flags=re.DOTALL | re.IGNORECASE))

def extract_answer(response: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", str(response), flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return str(response).strip()

def normalize_text(text: Any) -> str:
    text = str(text).strip().lower()
    text = {"true": "yes", "false": "no"}.get(text, text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def decimal_or_none(text: Any) -> Decimal | None:
    cleaned = str(text).strip().replace(",", "")
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    try:
        return Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None

def answer_matches(prediction: str, ground_truth: Any) -> bool:
    pred_norm = normalize_text(prediction)
    gt_norm = normalize_text(ground_truth)
    if not pred_norm or not gt_norm:
        return False

    gt_num = decimal_or_none(ground_truth)
    if gt_num is not None:
        pred_number_match = re.search(r"-?\d+(?:\.\d+)?%?", prediction.replace(",", ""))
        if pred_number_match is None:
            return False
        pred_num = decimal_or_none(pred_number_match.group(0))
        return pred_num is not None and pred_num == gt_num

    if gt_norm == pred_norm:
        return True

    if gt_norm in {"yes", "no"} or len(gt_norm.split()) == 1:
        return re.search(rf"\b{re.escape(gt_norm)}\b", pred_norm) is not None

    return gt_norm in pred_norm

def accuracy_reward(response: str, ground_truth: Any) -> float:
    prediction = extract_answer(response)
    if isinstance(ground_truth, (list, tuple, set)):
        return 1.0 if any(answer_matches(prediction, gt) for gt in ground_truth) else 0.0
    return 1.0 if answer_matches(prediction, ground_truth) else 0.0

def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        ground_truth = reward_input["ground_truth"]
        
        # 格式奖励：必须包含 <answer> 标签
        format_score = 1.0 if has_answer_tag(response) else 0.0
        
        # 准确率奖励
        acc_score = accuracy_reward(response, ground_truth)
        
        # 只有格式正确时才计算准确率，否则准确率为 0
        final_acc = acc_score if format_score > 0 else 0.0
        
        scores.append(
            {
                "overall": final_acc,
                "acc": final_acc,
                "accuracy": final_acc,
                "format": format_score,
            }
        )
    return scores
