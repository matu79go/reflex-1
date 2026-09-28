---
license: apache-2.0
base_model: google/gemma-4-E4B-it
library_name: peft
pipeline_tag: text-classification
language:
- en
- ja
tags:
- decision-model
- classification
- latent-reasoning
- lora
- gemma4
---

# Reflex-1 (4B)

**Frontier-LLM accuracy at Jev speed. A 4B open model.**

Reflex-1 is a decision model for routing, guardrails and classification, built on Google Gemma 4 E4B. It returns a choice with probabilities instead of generated text.

- **instant** — classification in a single forward pass; the probability of each option's first token is read from the LM head. No training needed; text and image inputs. Same accuracy band as GPT-5.6 and Gemini 3.8 on five public benchmarks, at 0.10 s median latency.
- **deep (latent reasoning)** — for trained judgments. The last hidden state is fed back as the next input, one latent step per reasoning hop, without writing reasoning out as text. During training each latent step is supervised through the model's shared LM head as a one-line description of the state at that hop, so the steps can be decoded into words (`trace`).

Code, endpoint and API: [github.com/matu79go/reflex-1](https://github.com/matu79go/reflex-1)

## Variants

| Repository | What it is | Download | GPU memory |
|---|---|---|---|
| **matu79go/Reflex-1-4B** (this page) | The Reflex-1 skills (LoRA + latent modules) and registry. Needed for every variant | 1.8 GB | — |
| [matu79go/Reflex-1-4B-bnb-4bit](https://huggingface.co/matu79go/Reflex-1-4B-bnb-4bit) | Base model, pre-quantized 4-bit NF4. **Recommended** | 9.3 GB | ~10 GB |
| [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | Base model, original BF16 (quantize at load with `--quant nf4`, or run in BF16 with `--quant none`) | ~16 GB | ~10 GB / ~16 GB |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matu79go/reflex-1/blob/main/notebooks/quickstart.ipynb)

## What is in this repository

| Path | Contents |
|---|---|
| `skills/web_of_lies/` | LoRA adapter + latent modules for chains of truth-tellers and liars (BIG-Bench Hard format, trained up to 15 people) |
| `skills/boolean_expressions/` | LoRA adapter + latent modules for not/and/or expressions |
| `skills/navigate/` | LoRA adapter + latent modules for navigation (return to start?) |
| `skills.json` | Skill registry: prompt, options and latent-step rule per skill |

Each skill folder holds `adapter_config.json`, `adapter_model.safetensors` (PEFT LoRA on Gemma 4 E4B) and `latent_parts.safetensors` (the projection from the last hidden state back to the input, and the initial latent seats). `instant` mode uses the base model without adapters.

## Usage

```bash
git clone https://github.com/matu79go/reflex-1.git && cd reflex-1
pip install -r requirements.txt
python -m reflex.server --base matu79go/Reflex-1-4B-bnb-4bit --skills skills.json --port 8097
```

```json
POST /v1/decisions
{"state": "Question: Ryan tells the truth. Michael says Ryan lies. ... Does Millicent tell the truth?",
 "mode": "deep", "skill": "web_of_lies", "trace": true,
 "questions": {"answer": {"type": "choice", "instructions": "Answer Yes or No.", "criteria": {"Yes": "", "No": ""}}}}
```

Base model: use the pre-quantized [matu79go/Reflex-1-4B-bnb-4bit](https://huggingface.co/matu79go/Reflex-1-4B-bnb-4bit) (9.3 GB download) or the original `google/gemma-4-E4B-it` (about 16 GB, quantized at load time with `--quant nf4`). The skills work with both.

## Evaluation

| Task | GPT-5.6 Sol | Gemini 3.8 Flash | Jev | **Reflex-1** |
|---|---|---|---|---|
| Intent (CLINC150) | 97 | 96 | 96.5 | 93.5 |
| Japanese intent (MASSIVE ja-JP) | 96 | 96 | 92.5 | 90.0 |
| Emotion (tweet_eval) | 84 | 79 | 81.0 | **82.0** |
| Sentiment (tweet_eval) | 78 | 77 | 74.5 | 74.5 |
| Moderation: insult (Civil Comments) | 73 | 76 | 73.5 | **80.0** |
| Median latency (classification) | 1.1–1.3 s | 2.1–2.6 s | 0.30 s | **0.09–0.21 s** |
| Web of Lies, 10 people | 46 (reasoning off) | 100 (reasoning on, 2.7 s) | 59.0 | **100 (0.8 s)** |
| Web of Lies, 20 people | 50 | 100 (3.1 s) | 58.0 | **72 (1.5 s)** |
| Object tracking, 7 people / 14 swaps | 10 | 100 (3.5 s) | 26.2 | **70 (1.2 s)** |
| Boolean expressions (BBH) | 92 | 100 | 98.9 | **100 (0.4 s)** |
| Navigation (BBH) | 78 | 100 | 98.2 | 80.8 (0.5 s) |

Jev and Reflex-1: 200 questions per classification task; frontier LLMs: 100 (50 for reasoning). Reasoning tasks are BIG-Bench Hard tasks deepened with the same rules. Reflex-1 measured sequentially on one NVIDIA GB10; others via OpenRouter, network included (September 2026). The object-tracking result uses an earlier task-specific model and is not included as a skill here.

At 4-bit (NF4) the deep skills keep their accuracy and run 13–17% faster than BF16; the 4-bit weights take 9.8 GB of GPU memory.

## Training data

Skills were trained on demonstrations generated from the public rules of each task (the state after every step is computed by a program). No outputs of Jev, GPT or Gemini were used for training. Training code is not released.

## Limitations

- `deep` works on the kinds and depths of judgment each skill was trained on and does not transfer as-is to other kinds of reasoning. Beyond the trained depth accuracy drops (Web of Lies with 40 people: 46%).
- `deep` currently requires an NVIDIA GPU with CUDA.
- The navigate trace is readable only for the first steps; the web_of_lies trace is readable throughout.

## License

Apache License 2.0. Derived from Google Gemma 4 E4B (Apache 2.0). Not affiliated with or endorsed by Google. Jev is a product of TypeSafe AI; comparisons were measured by the author and are not endorsed by TypeSafe AI.

## Citation

```bibtex
@misc{reflex1_2026,
  title  = {Reflex-1: frontier-LLM accuracy at Jev speed with a 4B open model},
  author = {suzuki_shoten},
  year   = {2026},
  url    = {https://huggingface.co/matu79go/Reflex-1-4B}
}
```
