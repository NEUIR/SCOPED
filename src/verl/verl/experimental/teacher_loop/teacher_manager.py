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
import logging
import os
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.tokenizer.tokenizer import normalize_token_ids
from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _get_teacher_sampling_params(
    teacher_model_config: DistillationTeacherModelConfig,
    distillation_loss_config: DistillationLossConfig,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    # Temperature has no effect on prompt_logprobs: the teacher performs a forward pass over
    # existing tokens (no sampling). Always use temperature=1.0 regardless of the config value.
    # The default distillation.yaml copies the student rollout temperature via Hydra interpolation
    # (temperature: ${oc.select:actor_rollout_ref.rollout.temperature}), which causes a spurious
    # crash when rollout.temperature != 1.0.
    if teacher_model_config.inference.temperature != 1.0:
        logger.warning(
            "Teacher inference temperature is set to %.1f, but temperature has no effect "
            "on prompt_logprobs (forward pass only). Using temperature=1.0.",
            teacher_model_config.inference.temperature,
        )
    num_logprobs = distillation_loss_config.topk if distillation_loss_config.loss_settings.use_topk else 0
    return {
        "max_tokens": 1,
        "temperature": 1.0,
        "prompt_logprobs": num_logprobs,
    }


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO(wuxibin): remove padding and use tensordict.
    left_pad_size = prompt_width - prompt_length
    right_pad_size = response_width - response_length
    padding = (0, 0, left_pad_size, right_pad_size)
    return (
        F.pad(teacher_ids, padding, value=pad_token_id).unsqueeze(0),
        F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0),
    )


class AsyncTeacherLLMServerManager:
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        teacher_client: dict[str, LLMServerClient],
        tokenizer: Any,
    ):
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.teacher_key: str = self.distillation_config.teacher_key

        self.teacher_model_configs: dict[str, DistillationTeacherModelConfig] = self.distillation_config.teacher_models
        expected = set(self.teacher_model_configs)
        if set(teacher_client.keys()) != expected:
            raise ValueError(
                f"teacher client keys {sorted(teacher_client.keys())} "
                f"do not match teacher routing keys {sorted(expected)}."
            )
        self.teacher_client: dict[str, LLMServerClient] = teacher_client
        self.tokenizer = tokenizer
        self._chat_template_kwargs = dict(config.data.get("apply_chat_template_kwargs", {}))
        self._skill_token_ids: dict[tuple[str, str], list[int]] = {}
        self._skill_path_aliases: dict[str, str] = {}
        self._user_suffix_token_ids: Optional[list[int]] = None

    def _get_skill_token_ids(self, teacher_key: str, skill_path: Optional[str] = None) -> list[int]:
        """Return teacher-only skill-text tokens, cached per teacher/path."""
        skill_path = str(skill_path) if skill_path is not None else None
        # An explicitly empty path disables the skill for this row (e.g. OCR
        # rows trained without skill) instead of falling back to the
        # domain-level default.
        if skill_path is not None and not skill_path.strip():
            return []
        skill_path = skill_path or self.teacher_model_configs[teacher_key].skill_prompt_path
        if skill_path is None:
            return []
        skill_path = str(skill_path)
        if not skill_path.strip():
            return []
        requested_skill_path = skill_path

        # Public exports contain no private skill paths. User-provided paths
        # resolve against the launch directory or an explicit absolute location.
        aliased_path = self._skill_path_aliases.get(skill_path)
        resolved_skill_path = Path(aliased_path if aliased_path is not None else skill_path).expanduser()
        if not resolved_skill_path.is_file():
            raise FileNotFoundError(f"Teacher skill prompt does not exist: {skill_path}")

        skill_path = str(resolved_skill_path)
        self._skill_path_aliases[requested_skill_path] = skill_path
        cache_key = (teacher_key, skill_path)
        if cache_key in self._skill_token_ids:
            return self._skill_token_ids[cache_key]

        with resolved_skill_path.open(encoding="utf-8") as file:
            skill_prompt = file.read().strip()
        if not skill_prompt:
            raise ValueError(f"Teacher skill prompt is empty: {skill_path}")

        # Keep the skill as plain content, rather than rendering it as a
        # ``system`` turn.  It is inserted just before the original user's
        # closing turn marker by ``_inject_skill_as_user_suffix`` below.
        # A leading blank line separates it from the question while preserving
        # the JSON skill text itself unchanged.
        token_ids = self.tokenizer("\n\n" + skill_prompt, add_special_tokens=False)
        self._skill_token_ids[cache_key] = normalize_token_ids(token_ids)
        return self._skill_token_ids[cache_key]

    def _get_user_suffix_token_ids(self) -> list[int]:
        """Return ``user-close + assistant-generation`` tokens for this template.

        The student prompt is rendered as a completed user turn followed by
        the assistant generation header.  Deriving this boundary from the
        tokenizer's own chat template keeps suffix injection correct for Qwen
        and other templates without depending on hard-coded special token IDs.
        """
        if self._user_suffix_token_ids is not None:
            return self._user_suffix_token_ids

        user_empty = normalize_token_ids(
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}],
                tokenize=True,
                add_generation_prompt=False,
                **self._chat_template_kwargs,
            )
        )
        user_probe = normalize_token_ids(
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "teacher-suffix-probe"}],
                tokenize=True,
                add_generation_prompt=False,
                **self._chat_template_kwargs,
            )
        )
        common_tail_size = 0
        while (
            common_tail_size < len(user_empty)
            and common_tail_size < len(user_probe)
            and user_empty[-1 - common_tail_size] == user_probe[-1 - common_tail_size]
        ):
            common_tail_size += 1
        if not common_tail_size:
            raise ValueError("Could not derive the user-turn closing tokens for teacher skill suffix injection")
        user_close = user_empty[-common_tail_size:]

        with_generation = normalize_token_ids(
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}],
                tokenize=True,
                add_generation_prompt=True,
                **self._chat_template_kwargs,
            )
        )
        if with_generation[: len(user_empty)] != user_empty:
            raise ValueError(
                "Chat template's generation prompt does not extend its completed user turn; "
                "cannot safely inject teacher skill as a user suffix"
            )
        assistant_generation = with_generation[len(user_empty) :]
        if not assistant_generation:
            raise ValueError("Chat template has no assistant generation prompt for teacher skill suffix injection")

        self._user_suffix_token_ids = user_close + assistant_generation
        return self._user_suffix_token_ids

    def _inject_skill_as_user_suffix(
        self,
        sequence_ids: list[int],
        student_prompt_length: int,
        skill_token_ids: list[int],
    ) -> list[int]:
        """Insert skill text after the question, inside the final user turn."""
        if not skill_token_ids:
            return sequence_ids
        user_suffix = self._get_user_suffix_token_ids()
        if student_prompt_length < len(user_suffix):
            raise ValueError(
                f"Student prompt length {student_prompt_length} is shorter than the chat-template suffix "
                f"({len(user_suffix)} tokens)"
            )
        suffix_start = student_prompt_length - len(user_suffix)
        if sequence_ids[suffix_start:student_prompt_length] != user_suffix:
            raise ValueError(
                "Student prompt does not end in the expected user-close plus assistant-generation tokens; "
                "cannot safely inject teacher skill as a user suffix"
            )
        return sequence_ids[:suffix_start] + skill_token_ids + sequence_ids[suffix_start:]

    def _resolve_teacher_key(self, routing_key: Optional[str]) -> str:
        if len(self.teacher_model_configs) == 1:
            # Single-teacher path: route everything to the one teacher regardless of the sample's key.
            return next(iter(self.teacher_model_configs))
        if routing_key is None:
            raise ValueError(
                f"Routing key is required for multi-teacher distillation "
                f"(configured via distillation.teacher_key={self.teacher_key!r})."
            )
        if routing_key not in self.teacher_model_configs:
            raise ValueError(
                f"No teacher configured for routing key {routing_key!r}. "
                f"Configured teachers: {sorted(self.teacher_model_configs)}."
            )
        return routing_key

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        student_prompt_length: int,
        pad_token_id: int,
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
        skill_prompt_path: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher log probabilities for a single unpadded sequence."""
        multi_modal_data = multi_modal_data or {}
        teacher_key = self._resolve_teacher_key(routing_key)
        teacher_model_config = self.teacher_model_configs[teacher_key]
        client = self.teacher_client[teacher_key]
        skill_token_ids = self._get_skill_token_ids(teacher_key, skill_prompt_path)
        teacher_sequence_ids = self._inject_skill_as_user_suffix(
            sequence_ids, student_prompt_length, skill_token_ids
        )
        max_model_len = teacher_model_config.inference.max_model_len
        if max_model_len is not None and len(teacher_sequence_ids) + 1 > max_model_len:
            raise ValueError(
                f"Teacher sequence ({len(teacher_sequence_ids)} tokens including its skill) plus one generation "
                f"token exceeds max_model_len={max_model_len} for teacher {teacher_key!r}."
            )
        teacher_output = await client.generate(
            request_id=uuid4().hex,
            prompt_ids=teacher_sequence_ids,
            sampling_params=_get_teacher_sampling_params(teacher_model_config, self.distillation_loss_config),
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_kwargs=mm_processor_kwargs,
        )
        # Shapes: # S, (1 or K), where S is the response length, K is either 1 or topk depending on
        # the distillation loss settings.
        teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
        teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
        assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == len(teacher_sequence_ids)

        # The teacher-only suffix changes the teacher prompt length, but the
        # loss is defined only on student response tokens.  Drop the modified
        # teacher prompt logprobs and restore a dummy student-prompt prefix so
        # downstream padding and response masks remain aligned with the
        # student's unmodified prompt.
        if not 0 < student_prompt_length <= len(sequence_ids):
            raise ValueError(
                "student_prompt_length must be positive and no greater than the full sequence length; "
                f"got {student_prompt_length=} and sequence_length={len(sequence_ids)}"
            )
        response_length = len(sequence_ids) - student_prompt_length

        # vLLM drops the first prompt logprob and appends a dummy row.  The
        # first response-token logprob therefore lives at the predecessor
        # position: suffix_skill_len + student_prompt_length - 1.  Keep the
        # final dummy row so the returned tensor has exactly the student's
        # sequence length and remains compatible with no_padding_2_padding().
        response_start = len(skill_token_ids) + student_prompt_length - 1
        teacher_ids = teacher_ids[response_start:]
        teacher_logprobs = teacher_logprobs[response_start:]
        assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == response_length + 1
        prefix_shape = (student_prompt_length - 1, *teacher_ids.shape[1:])
        teacher_ids = torch.cat((torch.full(prefix_shape, pad_token_id, dtype=teacher_ids.dtype), teacher_ids), dim=0)
        teacher_logprobs = torch.cat(
            (
                torch.zeros((student_prompt_length - 1, *teacher_logprobs.shape[1:]), dtype=teacher_logprobs.dtype),
                teacher_logprobs,
            ),
            dim=0,
        )
        assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == len(sequence_ids)
        return teacher_ids, teacher_logprobs
