"""Request each capability expert through its OpenAI-compatible vLLM server."""
import base64
from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path

from openai import OpenAI
from PIL import Image


class Engine:
    def __init__(self, model, base_url, config, api_key="EMPTY"):
        self.model = model
        self.config = config
        self.tokenizer = None
        self.client = OpenAI(base_url=base_url, api_key=api_key,
                             timeout=config["request_timeout"], max_retries=2)

    def check_model(self, expected_model=None):
        models = self.client.with_options(timeout=3, max_retries=0).models.list().data
        entry = next((item for item in models if item.id == self.model), None)
        if entry is None:
            raise ValueError(f"Endpoint does not serve {self.model}; available: {sorted(item.id for item in models)}")
        actual = getattr(entry, "root", None)
        if expected_model and actual:
            matches = (Path(actual).resolve() == Path(expected_model).resolve()
                       if Path(expected_model).is_dir() else actual == expected_model)
            if not matches:
                raise ValueError(f"Endpoint serves {actual}, not this capability's expert {expected_model}")

    def _chat(self, content, sampling):
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": content}], **sampling)
        if len(response.choices) != sampling.get("n", 1):
            raise RuntimeError("Expert returned fewer completions than requested")
        outputs = []
        for choice in response.choices:
            if not choice.message.content or choice.finish_reason not in ("stop", "length"):
                raise RuntimeError(f"Invalid expert completion: {choice.finish_reason}")
            outputs.append(choice.message.content.strip())
        return outputs

    def _vision(self, item, sampling):
        images = item["images"]
        if len(images) > self.config["server_max_images"]:
            raise ValueError("Increase server_max_images; images must not be silently truncated")
        content = []
        for path in images:
            with Image.open(path) as image:
                image = image.convert("RGB")
                longest = max(image.size)
                if longest > self.config["image_max_side"]:
                    scale = self.config["image_max_side"] / longest
                    image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
            url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": item["query"].replace("<image>", "").strip()})
        return self._chat(content, sampling)

    def generate_vision(self, items, sampling):
        with ThreadPoolExecutor(max_workers=self.config["online_max_workers"]) as pool:
            return list(pool.map(lambda item: self._vision(item, sampling), items))

    def generate_text(self, prompts, sampling):
        for prompt in prompts:
            if self.count_text_tokens(prompt) + sampling["max_tokens"] > self.config["server_max_model_len"]:
                raise ValueError("Text request exceeds the expert context; reduce evidence or increase server_max_model_len")
        with ThreadPoolExecutor(max_workers=self.config["online_max_workers"]) as pool:
            return list(pool.map(lambda prompt: self._chat([{"type": "text", "text": prompt}], sampling)[0], prompts))

    def count_text_tokens(self, prompt):
        if self.tokenizer is None:
            raise ValueError("The expert tokenizer is required for extraction context budgeting")
        return len(self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                      tokenize=True, add_generation_prompt=True))
