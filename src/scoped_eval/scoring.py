"""Original accuracy and OCR token-F1 rules, reading ground truths from HF rows."""
from collections import Counter

from .protocol import _has_cjk, _tokenize, judge


def score_record(name, row, response):
    record = judge(name, row, response)
    record.update(sample_id=row["sample_id"], source_split=row["source_split"], skipped=False)
    if name == "ccocrv2":
        gt = row["_gt_text"].strip()
        pred = (record.get("normalized_response") or response or "").strip()
        word_level = not _has_cjk(gt)
        parts = str(row.get("answer_path", "")).split("/")
        multi_scene = any(part == "multi_scene_recognition" or part.startswith("multi_scene_ocr") for part in parts)
        alphanum = bool(multi_scene and word_level)
        gt_tokens, pred_tokens = _tokenize(gt, word_level, alphanum), _tokenize(pred, word_level, alphanum)
        gc, pc = Counter(gt_tokens), Counter(pred_tokens)
        right = sum(min(count, pc.get(token, 0)) for token, count in gc.items())
        recall = right / (len(gt_tokens) + 1e-9)
        precision = right / (len(pred_tokens) + 1e-9)
        record.update(gt_text=gt, gt_tokens=len(gt_tokens), pred_tokens=len(pred_tokens), right_tokens=right,
                      recall=recall, precision=precision, f1=2 * recall * precision / (recall + precision + 1e-9))
    return record


def summarize(name, records):
    records = list(records)
    valid = [row for row in records if not row.get("skipped")]
    skipped = len(records) - len(valid)
    if name == "ccocrv2":
        count = len(valid)
        total_gt = sum(row["gt_tokens"] for row in valid)
        total_pred = sum(row["pred_tokens"] for row in valid)
        total_right = sum(row["right_tokens"] for row in valid)
        recall, precision = total_right / (total_gt + 1e-9), total_right / (total_pred + 1e-9)
        return {"num_samples": count, "skipped": skipped,
                "macro_f1": sum(row["f1"] for row in valid) / (count + 1e-9),
                "micro_f1": 2 * recall * precision / (recall + precision + 1e-9),
                "macro_recall": sum(row["recall"] for row in valid) / (count + 1e-9),
                "macro_precision": sum(row["precision"] for row in valid) / (count + 1e-9)}
    correct = sum(bool(row.get("correct")) for row in valid)
    return {"correct": correct, "total": len(valid), "skipped": skipped, "accuracy": correct / max(1, len(valid))}
