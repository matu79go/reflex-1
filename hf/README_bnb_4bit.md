---
license: apache-2.0
base_model: google/gemma-4-E4B-it
library_name: transformers
tags:
- bitsandbytes
- 4-bit
- nf4
- gemma4
- reflex-1
---

# Reflex-1-4B-bnb-4bit

Pre-quantized 4-bit (bitsandbytes NF4, double quantization) copy of **Google Gemma 4 E4B** for use as the base model of [Reflex-1](https://huggingface.co/matu79go/Reflex-1-4B). Download is 9.3 GB instead of about 16 GB, and no quantization is needed at load time.

The vision/audio towers, embeddings and LM head are kept in bfloat16; the transformer blocks are NF4.

## Variants

All Reflex-1 repositories: [Reflex-1 collection](https://huggingface.co/collections/matu79go/reflex-1-6ab9c9c2a8842e5c1f599ce5)

| Repository | What it is | Download | GPU memory |
|---|---|---|---|
| [matu79go/Reflex-1-4B](https://huggingface.co/matu79go/Reflex-1-4B) | The Reflex-1 skills (LoRA + latent modules) and registry. Needed for every variant | 1.8 GB | — |
| **matu79go/Reflex-1-4B-bnb-4bit** (this page) | Base model, pre-quantized 4-bit NF4. **Recommended** | 9.3 GB | ~10 GB |
| [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | Base model, original BF16 (quantize at load with `--quant nf4`, or run in BF16 with `--quant none`) | ~16 GB | ~10 GB / ~16 GB |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matu79go/reflex-1/blob/main/notebooks/quickstart.ipynb)

## Usage

```bash
git clone https://github.com/matu79go/reflex-1.git && cd reflex-1
pip install -r requirements.txt
python -m reflex.server --base matu79go/Reflex-1-4B-bnb-4bit --skills skills.json --port 8097
```

Results with this base are the same as quantizing `google/gemma-4-E4B-it` at load time (checked on intent classification, image classification and the three reasoning skills).

## License

Apache License 2.0. Derived from Google Gemma 4 E4B (Apache 2.0); only quantized, no other changes. Not affiliated with or endorsed by Google.
