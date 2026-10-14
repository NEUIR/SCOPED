"""Fallback answer parser for capability reward functions."""
import re

def extract_answer(text):
    """提取最后一个 <answer>...</answer> 内容;无标签时退回全文末尾。"""
    if text is None:
        return ""
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return text.strip()
