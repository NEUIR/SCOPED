# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fixed-probe dataset for policy-drift analysis.

Each probe sample is a prompt paired with a FROZEN chosen response (token IDs generated
once from the shared initial model, see scripts/make_probe_set.py). The dataset builds
teacher-forcing sequences (prompt + chosen response) in exactly the same way RLHFDataset
builds prompts, so probe logprobs are comparable with training-time logprobs.
"""

import json
import os
from typing import Any, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF
from .dataset import process_image


class ProbeDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "problem",
        image_key: str = "images",
        image_dir: Optional[str] = None,
        max_prompt_length: int = 8192,
        max_response_length: int = 4096,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.image_key = image_key
        self.image_dir = image_dir
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        with open(data_path, encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.samples)

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        content_list = []
        for i, content in enumerate(prompt_str.split("<image>")):
            if i != 0:
                content_list.append({"type": "image"})

            if content:
                content_list.append({"type": "text", "text": content})

        return [{"role": "user", "content": content_list}]

    def __getitem__(self, index):
        example = self.samples[index]
        messages = self._build_messages(example)

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        images = example[self.image_key]
        if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):
            images = [os.path.join(self.image_dir, image) for image in images]

        processed_images = [] if len(images) != 0 else None
        for image in images:
            processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

        model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="right",
        )

        # frozen chosen response, right-padded
        chosen = example["chosen_response_ids"][: self.max_response_length]
        response_length = self.max_response_length
        responses = torch.full((response_length,), self.tokenizer.pad_token_id, dtype=input_ids.dtype)
        response_mask = torch.zeros((response_length,), dtype=attention_mask.dtype)
        responses[: len(chosen)] = torch.tensor(chosen, dtype=input_ids.dtype)
        response_mask[: len(chosen)] = 1

        # response position ids continue from the last prompt position (mirrors vLLM rollout)
        increments = torch.arange(1, response_length + 1)
        if position_ids.dim() == 2:  # mrope: (4, P) -> (4, T)
            response_position_ids = position_ids[..., -1:] + increments.unsqueeze(0)
        else:  # (P,) -> (T,)
            response_position_ids = position_ids[..., -1:] + increments

        sample = {
            "input_ids": torch.cat((input_ids, responses), dim=0),
            "attention_mask": torch.cat((attention_mask, response_mask), dim=0),
            "position_ids": torch.cat((position_ids, response_position_ids), dim=-1),
            "responses": responses,
            "response_mask": response_mask,
            "multi_modal_data": {"images": images},
            "uid": f"probe_{example['sample_id']}",
            "sample_id": example["sample_id"],
            "subtype": example.get("subtype", "unknown"),
            "ground_truth": example.get("answer", ""),
        }
        return sample
