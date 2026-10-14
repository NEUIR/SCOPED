"""Verify generated skill provenance and available teacher context before MOPD."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs=4, required=True)
    parser.add_argument("--skills", nargs=4, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--student-context", type=int, default=12288 + 8096)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from scoped_skill.__main__ import model_identity
    for index, (model, skill) in enumerate(zip(args.models, args.skills), 1):
        path = Path(skill)
        card_path = path.with_suffix(".card.json")
        if card_path.is_file():
            card = json.loads(card_path.read_text())
            if card.get("capability") != f"capability{index}" or card.get("model") != model_identity(model):
                raise ValueError(f"Skill does not match capability{index}'s current teacher: {path}. Rerun stage 2.")
        text = path.read_text().strip()
        if not text:
            raise ValueError(f"Empty teacher skill: {path}")
        tokenizer = AutoTokenizer.from_pretrained(model)
        count = len(tokenizer.encode("\n\n" + text, add_special_tokens=False))
        required = args.student_context + count + 1
        if required > args.max_model_len:
            raise ValueError(f"{path}: requires TEACHER_MAX_MODEL_LEN>={required}, configured {args.max_model_len}")
        print(f"capability{index}: teacher skill verified ({count} tokens)", flush=True)


if __name__ == "__main__":
    main()
