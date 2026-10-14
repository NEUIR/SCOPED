"""Answer parsing, scoring and online client from the original no-skill eval.py.

The new runner adapts HF data without changing the existing scoring rules.
"""

import argparse
import ast
import asyncio
import base64
import io
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from PIL import Image

SAMPLING = {
    "greedy": {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "presence_penalty": 0.0, "repetition_penalty": 1.0},
    "sample": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "repetition_penalty": 1.0},
}


VANILLA_MLLM = "Please reason step by step, and put your final answer within <answer></answer>."


ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


COUNT_RE = re.compile(r"-?\d+")


JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


NUMBER_WORDS = {
    "zero": 0, "one": 1, "single": 1, "two": 2, "couple": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100,
}


def parse_answer(text: str | None) -> str:
    """提取 <answer>...</answer>(或 \\boxed / 'final answer' 标记 / 最后一行)。"""
    text = text or ""
    matches = ANSWER_TAG_RE.findall(text)
    if matches:
        return matches[-1].strip()
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text, re.DOTALL)
    if boxed:
        return boxed[-1].strip()
    marker_re = re.compile(
        r"(?:final\s+answer|correct\s+answer|answer|答案)\s*(?:is|为|是|:|：|-)?\s*(.+)", re.IGNORECASE)
    for candidate in reversed(marker_re.findall(text)):
        extracted = _clean_candidate(candidate)
        if extracted:
            return extracted
    for line in reversed([x.strip() for x in text.splitlines() if x.strip()]):
        extracted = _clean_candidate(line)
        if extracted:
            return extracted
    return text.strip()


def _clean_candidate(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"^[\s>*#`_\-]+|[\s<*_`。.!！?？]+$", "", text).strip()
    text = re.sub(r"^\$+|\$+$", "", text).strip()
    text = re.sub(r"^\\\((.*)\\\)$", r"\1", text).strip()
    text = re.sub(r"^\\\[(.*)\\\]$", r"\1", text).strip()
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text, re.DOTALL)
    if boxed:
        return boxed[-1].strip()
    bold = re.fullmatch(r"\*\*(.+?)\*\*", text, re.DOTALL)
    if bold:
        text = bold.group(1).strip()
    return text


def parse_count(text: str) -> int | None:
    """计数答案解析:优先 <answer> 标签内数字,支持英文数词。"""
    text = str(text or "").strip()

    def parse_number(candidate: str) -> int | None:
        digits = COUNT_RE.findall(candidate)
        if digits:
            return int(digits[-1])
        lowered = candidate.lower()
        last = None
        for word, value in NUMBER_WORDS.items():
            if re.search(rf"\b{re.escape(word)}\b", lowered):
                last = value
        return last

    matches = ANSWER_TAG_RE.findall(text)
    if matches:
        tagged = parse_number(matches[-1].strip())
        if tagged is not None:
            return tagged
    return parse_number(text)


def normalize_freeform_answer(value: Any) -> str:
    text = parse_answer(str(value or "")).strip()
    text = re.sub(r"^[:：]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^\$+|\$+$", "", text).strip()
    return text.rstrip(".").strip().lower()


def numeric_value(value: Any) -> float | None:
    text = normalize_freeform_answer(value)
    text = text.replace(",", "")
    text = re.sub(r"\\(?:mathrm|text)\{[^{}]*\}", "", text)
    text = re.sub(r"\\(?:degree|circ)\b|\^?°", "", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    matches = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def freeform_answers_match(predicted: Any, ground_truth: Any) -> bool:
    """自由文本匹配:归一化相等,或数值容差 0.1% 内相等。"""
    if normalize_freeform_answer(predicted) == normalize_freeform_answer(ground_truth):
        return True
    p, g = numeric_value(predicted), numeric_value(ground_truth)
    if p is not None and g is not None:
        return abs(p - g) <= max(1e-3, abs(g) * 1e-3)
    return False


def parse_choice_answer(value: Any) -> str | None:
    """从回答中提取选项字母 A-E。"""
    text = str(value or "").strip()
    if not text:
        return None
    tagged = ANSWER_TAG_RE.findall(text)
    if tagged:
        inner = parse_choice_answer(tagged[-1])
        if inner:
            return inner
    boxed = re.findall(r"\\boxed\{\s*([A-E])\s*\}", text, re.IGNORECASE)
    if boxed:
        return boxed[-1].upper()
    marker_re = re.compile(
        r"(?:final\s+answer|correct\s+answer|answer|答案)\s*(?:is|为|是|:|：|-)?\s*"
        r"(?:option|选项)?\s*[\(\[]?\s*([A-E])\b", re.IGNORECASE)
    markers = marker_re.findall(text)
    if markers:
        return markers[-1].upper()
    for line in reversed([x.strip() for x in text.splitlines() if x.strip()]):
        cleaned = re.sub(r"^[\s>*#`_\-]+|[\s<*_`。.!！?？]+$", "", line).strip()
        m = re.fullmatch(r"(?:option\s*)?[\(\[]?([A-E])[\)\].:：]?", cleaned, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    tail = "\n".join([x.strip() for x in text.splitlines() if x.strip()][-5:])
    m = re.search(r"[\(\[]\s*([A-E])\s*[\)\]]", tail, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-E])\b", tail, re.IGNORECASE)
    return m.group(1).upper() if m else None


def is_choice_ground_truth(value: Any) -> bool:
    text = parse_answer(str(value or "")).strip()
    return bool(re.fullmatch(r"(?:option\s*)?[\(\[]?\s*[A-E]\s*[\)\].:：]?", text, re.IGNORECASE))


def choice_letters(count: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(count)]


def normalize_chartqa_answer(value: Any) -> str:
    text = normalize_freeform_answer(value)
    text = text.replace(",", "").strip("[]")
    text = re.sub(r"\s*%\s*$", "%", text)
    try:
        number = float(text.rstrip("%"))
        return f"{number:g}" + ("%" if text.endswith("%") else "")
    except Exception:
        return text


def _try_number_from_text(text: str) -> float | None:
    text = str(text or "").strip().replace(",", "").rstrip("%")
    m = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
    try:
        return float(m[-1]) if m else None
    except Exception:
        return None


def chartqa_match(pred_eval: str, gt_eval: list[str], raw_response: str = "") -> bool:
    """ChartQA 宽松匹配:精确 / 数值容差2% / 词边界包含 / yes-no 等价。"""
    if pred_eval in gt_eval:
        return True
    pred_num = _try_number_from_text(pred_eval)
    if pred_num is not None:
        for gt in gt_eval:
            gt_num = _try_number_from_text(gt)
            if gt_num is not None and abs(pred_num - gt_num) <= max(1e-3, abs(gt_num) * 0.02):
                return True
    for gt in gt_eval:
        gt_clean = gt.strip().lower()
        if gt_clean and re.search(rf"\b{re.escape(gt_clean)}\b", pred_eval.lower()):
            return True
    pred_lower = pred_eval.lower().strip()
    for gt in gt_eval:
        gt_lower = gt.strip().lower()
        if gt_lower in ("yes", "no"):
            equiv = ("yes", "true", "correct") if gt_lower == "yes" else ("no", "false", "incorrect")
            if pred_lower in equiv:
                return True
    return False


def judge(dataset: str, row: dict, raw: str) -> dict:
    if dataset == 'pixmo-count':
        pred = parse_count(raw)
        gt = int(row['count'])
        return {'id': row['id'], 'label': row['label'], 'gt_count': gt, 'pred_count': pred, 'raw_response': raw, 'correct': pred == gt if pred is not None else False}
    if dataset == 'countbenchqa':
        pred = parse_count(raw)
        gt = int(row['_gt_count'])
        return {'id': row['id'], 'question': row['_question'], 'gt_count': gt, 'pred_count': pred, 'raw_response': raw, 'correct': pred == gt if pred is not None else False}
    if dataset == 'chartqa':
        predicted = parse_answer(raw)
        pred_eval = normalize_chartqa_answer(predicted)
        gt_eval = [normalize_chartqa_answer(a) for a in row['_labels']]
        return {'id': row['id'], 'query': row.get('query'), 'gt_answers': row['_labels'], 'predicted_answer': predicted, 'pred_eval': pred_eval, 'gt_eval': gt_eval, 'raw_response': raw, 'correct': chartqa_match(pred_eval, gt_eval, raw)}
    if dataset == 'wemath':
        predicted = parse_answer(raw)
        pred_eval = parse_choice_answer(predicted) or parse_choice_answer(raw)
        gt_eval = parse_choice_answer(row.get('answer'))
        return {'id': row['id'], 'question': row.get('question'), 'gt_answer': row.get('answer'), 'predicted_answer': predicted, 'pred_eval': pred_eval, 'gt_eval': gt_eval, 'raw_response': raw, 'correct': pred_eval == gt_eval if pred_eval and gt_eval else False}
    if dataset == 'muirbench':
        predicted = parse_answer(raw)
        pred_eval = parse_choice_answer(predicted)
        gt_eval = parse_choice_answer(row.get('answer'))
        return {'id': row['id'], 'task': row.get('task'), 'gt_answer': row.get('answer'), 'predicted_answer': predicted, 'pred_eval': pred_eval, 'gt_eval': gt_eval, 'raw_response': raw, 'correct': pred_eval == gt_eval if pred_eval and gt_eval else False}
    if dataset == 'visualpuzzles':
        predicted = parse_answer(raw)
        pred_eval = parse_choice_answer(predicted) or parse_choice_answer(raw)
        gt_eval = parse_choice_answer(row.get('answer'))
        return {'id': row['id'], 'question': row.get('question'), 'options': row.get('_options'), 'gt_answer': row.get('answer'), 'pred_eval': pred_eval, 'gt_eval': gt_eval, 'raw_response': raw, 'correct': pred_eval == gt_eval if pred_eval and gt_eval else False}
    if dataset == 'mmsi-bench':
        predicted = parse_answer(raw)
        gt = row.get('answer')
        if is_choice_ground_truth(gt):
            pred_eval = parse_choice_answer(predicted) or parse_choice_answer(raw)
            gt_eval = parse_choice_answer(gt)
            correct = pred_eval == gt_eval if pred_eval and gt_eval else False
        else:
            pred_eval, gt_eval = (normalize_freeform_answer(predicted), normalize_freeform_answer(gt))
            correct = freeform_answers_match(predicted, gt)
        return {'id': row['id'], 'question': row.get('question'), 'gt_answer': gt, 'pred_eval': pred_eval, 'gt_eval': gt_eval, 'raw_response': raw, 'correct': correct}
    if dataset == 'ccocrv2':
        matches = ANSWER_TAG_RE.findall(raw)
        normalized = matches[-1].strip() if matches and matches[-1].strip() else raw
        return {'id': row['id'], 'task_rel_path': row.get('task_rel_path'), 'subset': row.get('subset'), 'stem': row.get('stem'), 'raw_response': raw, 'normalized_response': normalized}
    raise ValueError(f'未知数据集: {dataset}')


def _token_normalize(token: str, is_lower: bool = True, alphanum_only: bool = False) -> str:
    if is_lower:
        token = token.lower()
    if alphanum_only:
        token = re.sub("[^A-Za-z0-9]+", "", token)
    return token


def _tokenize(text: str, word_level: bool, alphanum_only: bool = False) -> list[str]:
    text = text.replace("\t", " ").replace("\n", " ").replace("###", "").replace("***", "")
    text = re.sub(r"\s+", " ", text)
    if not word_level:
        text = text.replace(" ", "")
    tokens = text.split(" ") if word_level else list(text)
    return [t for t in (_token_normalize(x, True, alphanum_only) for x in tokens) if t]


def _has_cjk(s: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in s)


def _pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class OnlineEngine:
    """OpenAI 兼容 API 异步并发客户端, 接口与 Engine.generate 一致。"""

    def __init__(self, cfg: dict, base_url: str, concurrency: int = 64, dp_size: int = 1, model_name: str = "eval-model"):
        import aiohttp
        self._aiohttp = aiohttp
        self.cfg = cfg
        self.base_url = base_url.rstrip("/")
        self.concurrency = concurrency
        self.dp_size = dp_size
        self.model_name = model_name
        self._req_counter = 0
        self.timeout = aiohttp.ClientTimeout(total=7200)

    def _payload(self, images: list, prompt: str) -> dict:
        sp = self.cfg.get("sampling") or SAMPLING[self.cfg["mode"]]
        content = [{"type": "image_url", "image_url": {"url": _pil_to_data_url(img)}} for img in images]
        content.append({"type": "text", "text": prompt})
        body = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.cfg["max_tokens"],
            "temperature": sp["temperature"],
            "top_p": sp["top_p"],
            "presence_penalty": sp["presence_penalty"],
            "repetition_penalty": sp["repetition_penalty"],
            "seed": sp.get("seed", 0),
            "top_k": sp["top_k"],
        }
        if self.cfg.get("chat_template_kwargs"):
            body["chat_template_kwargs"] = self.cfg["chat_template_kwargs"]
        return body

    async def _one(self, session, sem, payload: dict) -> str:
        async with sem:
            headers = {}
            if self.dp_size > 1:
                rank = self._req_counter % self.dp_size
                self._req_counter += 1
                headers["x-vllm-dp-rank"] = str(rank)
            for attempt in range(3):
                try:
                    async with session.post(self.base_url + "/v1/chat/completions",
                                            json=payload,
                                            headers=headers) as resp:
                        data = await resp.json()
                        if resp.status != 200:
                            raise RuntimeError(f"HTTP {resp.status}: {str(data)[:300]}")
                        return (data["choices"][0]["message"]["content"] or "").strip()
                except Exception:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2 ** attempt)

    def generate(self, batch: list[tuple[list, str]]) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=16) as ex:
            payloads = list(ex.map(lambda bp: self._payload(bp[0], bp[1]), batch))

        async def run():
            sem = asyncio.Semaphore(self.concurrency)
            async with self._aiohttp.ClientSession(timeout=self.timeout) as session:
                return list(await asyncio.gather(
                    *[self._one(session, sem, p) for p in payloads]))
        return asyncio.run(run())

    def generate_stream(self, batch: list[tuple[list, str]], on_done) -> None:
        """全量提交(仅受并发信号量限制), 每条完成立即回调 on_done(index, response)。"""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=16) as ex:
            payloads = list(ex.map(lambda bp: self._payload(bp[0], bp[1]), batch))

        async def run():
            sem = asyncio.Semaphore(self.concurrency)
            async with self._aiohttp.ClientSession(timeout=self.timeout) as session:
                async def wrapped(idx: int, payload: dict):
                    resp = await self._one(session, sem, payload)
                    on_done(idx, resp)
                await asyncio.gather(*[wrapped(i, p) for i, p in enumerate(payloads)])
        asyncio.run(run())
