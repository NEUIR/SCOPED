# SCOPED: Skill-Decoupled On-Policy Expert Distillation for Unified Vision-Language Models

<a name="paper"></a>

Source code for **SCOPED: Skill-Decoupled On-Policy Expert Distillation for Unified Vision-Language Models**.

Paper: [arXiv:2610.22217](https://arxiv.org/abs/2610.22217).

<div align="center">

[![Paper](https://img.shields.io/badge/arXiv-2610.22217-red)](https://arxiv.org/abs/2610.22217)
[![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/NEUIR/SCOPED)
[![Model 4B](https://img.shields.io/badge/🤗%20Model-SCOPED--4B-blue)](https://huggingface.co/xiongyq/SCOPED-4B)
[![Model 8B](https://img.shields.io/badge/🤗%20Model-SCOPED--8B-blue)](https://huggingface.co/xiongyq/SCOPED-8B)
[![Training Data](https://img.shields.io/badge/🤗%20Hugging%20Face-SCOPED--Train-orange)](https://huggingface.co/datasets/xiongyq/SCOPED-Train)
[![Evaluation Data](https://img.shields.io/badge/🤗%20Hugging%20Face-SCOPED--Eval-orange)](https://huggingface.co/datasets/xiongyq/SCOPED-Eval)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[English](README.md) | [中文](README_zh.md)

</div>

<div align="center">

[📖 Overview](#overview) • [⚙️ Setup](#setup) • [⚡ Training](#training) • [🔧 Inference](#inference)<br>
[📃 Evaluation](#evaluation) • [🚀 Quick Start](#quick-start) • [📧 Contact](#contact)

</div>

<a name="overview"></a>

## 📖 Overview

SCOPED is a framework for unifying visual understanding capabilities through skill-decoupled on-policy expert distillation. It first trains capability-specialized experts with reinforcement learning, then abstracts reusable skills from their successful and failed trajectories. These skills provide teacher-only guidance during on-policy distillation, allowing a single student model to learn from skill-conditioned experts.

![SCOPED overview](figs/overeview.png)

The framework covers four complementary capabilities: **Perception**, **OCR & Chart**, **Symbolic Reasoning**, and **Multi-Image Reasoning**. The resulting student performs inference directly, without additional skill prompts.

<a name="setup"></a>

## ⚙️ Setup

Use Python 3.12 on Linux with NVIDIA GPUs and a compatible CUDA runtime. Clone the repository first:

```bash
git clone https://github.com/NEUIR/SCOPED.git
cd SCOPED
```

Install two environments from the repository root: `.venv-rl` for expert reinforcement learning, and `.venv-scoped` for OPD, skill abstraction, inference and evaluation. The bundled EasyR1 and veRL backends require different dependency versions and must be installed separately.

```bash
# Expert reinforcement learning.
python3.12 -m venv .venv-rl
.venv-rl/bin/python -m pip install -r requirements-rl.txt

# OPD, skill abstraction, inference and evaluation.
python3.12 -m venv .venv-scoped
.venv-scoped/bin/python -m pip install -r requirements.txt
```

Launch scripts automatically select the environment for each stage; activation is not required. If the required environment is missing, the script stops with an installation reminder.

Models are downloaded from Hugging Face when a model ID is specified, and cached files are reused when available. Training and evaluation scripts automatically download the corresponding datasets. Adjust GPU settings to match your hardware; the launch scripts default to four GPUs for expert training, skill abstraction and evaluation, and eight for distillation.

<a name="training"></a>

## ⚡ Training

> To use the released checkpoints, skip training and proceed to [Inference](#inference) or [Evaluation](#evaluation).

Our training pipeline builds on [EasyR1](https://github.com/hiyouga/EasyR1) and [veRL](https://github.com/verl-project/verl) and consists of three stages. Training data is available on [Hugging Face](https://huggingface.co/datasets/xiongyq/SCOPED-Train). Set the base and student model paths in [scripts/train.sh](scripts/train.sh); stage-specific settings are provided in [configs/](configs/).

### Stage 1: Capability-Specialized Expert Learning

In the first stage, four experts are independently optimized on capability-specific data to learn perception, OCR and chart understanding, symbolic reasoning, and multi-image reasoning. All experts are initialized from the same base model, which defaults to `Qwen/Qwen3-VL-8B-Instruct`.

```bash
bash scripts/train.sh rl
# Train one expert:
bash scripts/train.sh capability1
```

The full RL stage automatically exports the expert checkpoints for subsequent stages. To use existing experts, set `CAPABILITY1_TEACHER` through `CAPABILITY4_TEACHER` in [scripts/train.sh](scripts/train.sh).

### Stage 2: Capability Skill Abstraction from Expert Trajectories

In the second stage, each expert reflects on successful and failed trajectories generated from its own training data. The resulting evidence is consolidated into a capability-level skill containing reusable verification principles and failure-avoidance guidance.

```bash
bash scripts/train.sh skill
# Extract one expert's skill:
bash scripts/train.sh skill capability1
```

This stage produces one `SKILL.md` for each capability, which is loaded by the corresponding teacher during distillation.

### Stage 3: Skill-Decoupled On-Policy Expert Distillation

In the final stage, the student generates on-policy responses and learns from the corresponding capability expert conditioned on its extracted skill. Skills serve as privileged guidance for teachers, while the student learns a unified policy from their supervision.

```bash
bash scripts/train.sh opd
```

Expert checkpoints and skills from the preceding stages are loaded automatically. Set `STUDENT_MODEL` in [scripts/train.sh](scripts/train.sh) to select the student initialization.

<a name="inference"></a>

## 🔧 Inference

To run inference with a released SCOPED checkpoint, start a vLLM server:

```bash
.venv-scoped/bin/vllm serve xiongyq/SCOPED-8B \
    --served-model-name scoped \
    --tensor-parallel-size 4 \
    --port 8000
```

Replace the model ID with `xiongyq/SCOPED-4B` or a local checkpoint as needed, and adjust `--tensor-parallel-size` for your hardware. The [Quick Start](#quick-start) example below sends an image question to this server.

<a name="evaluation"></a>

## 📃 Evaluation

Evaluate SCOPED on all supported benchmarks or a selected subset. The script automatically downloads [SCOPED-Eval](https://huggingface.co/datasets/xiongyq/SCOPED-Eval), generates predictions and reports benchmark-specific metrics.

```bash
bash scripts/eval.sh all
# Selected benchmarks:
bash scripts/eval.sh chartqa,ccocrv2
# Another checkpoint:
MODEL=xiongyq/SCOPED-4B bash scripts/eval.sh all
```

The default model is `xiongyq/SCOPED-8B`. To evaluate through the server started above:

```bash
BASE_URL=http://localhost:8000 bash scripts/eval.sh chartqa --served-model-name scoped
```

Set `EVAL_OUT_DIR` to choose the result directory, or pass `--resume` to continue a previous run.

<a name="quick-start"></a>

## 🚀 Quick Start

After starting the server in [Inference](#inference), replace `image.png` with your own PNG image and run the following example. It asks the model to reason step by step and return its final answer within `<answer>` tags.

```python
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

image = base64.b64encode(Path("image.png").read_bytes()).decode()
payload = {
    "model": "scoped",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{image}"
            }},
            {"type": "text", "text": (
                "What is shown in this image? "
                "Please reason step by step, and put your final answer "
                "within <answer></answer>."
            )},
        ],
    }],
    "temperature": 0.0,
    "max_tokens": 2048,
}
request = Request(
    "http://localhost:8000/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urlopen(request, timeout=600) as response:
    result = json.load(response)
print(result["choices"][0]["message"]["content"])
```

## Acknowledgements

Our work builds on [EasyR1](https://github.com/hiyouga/EasyR1), [veRL](https://github.com/verl-project/verl), [vLLM](https://github.com/vllm-project/vllm) and [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL). We thank their contributors for making these projects available.

## License

SCOPED original code is released under the [MIT License](LICENSE). Bundled EasyR1 and veRL code retains its Apache 2.0 licenses and notices; see [NOTICE](NOTICE).

<a name="contact"></a>

## 📧 Contact

If you have questions, suggestions, and bug reports, please give us an issue or email:

```
yqxiong54@gmail.com
```
