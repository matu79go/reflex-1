"""Reflex-1 inference: instant decisions (text / image) and latent-reasoning skills.

instant: one forward pass; each option's first-token probability is read from the LM head.
deep:    the last hidden state is fed back as the next input for K latent steps (one step per
         reasoning hop), then the answer is read out. Each latent step can be decoded into one
         line of text through the LM head (trace).
"""
from __future__ import annotations

import contextlib
import json
import os
import re

import torch
import torch.nn as nn

BOT = "<|channel>thought\n"  # Gemma 4 thinking channel: latent steps follow this marker
EOT = "<channel|>"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class LatentProj(nn.Module):
    """Last hidden state -> next input embedding + per-layer embeddings (Gemma 4 PLE)."""

    def __init__(self, hidden: int, n_layers: int, ple_dim: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.ple = nn.Linear(hidden, n_layers * ple_dim)
        self.n_layers, self.ple_dim = n_layers, ple_dim

    def forward(self, h):
        e = self.body(h)
        return e, self.ple(e).view(*h.shape[:-1], self.n_layers, self.ple_dim)


def _messages(question: str, options: dict, state: str, image=None):
    opts = "\n".join(f"- {k}: {v}" for k, v in options.items())
    text = f"{state}\n\nQuestion: {question}\nOptions:\n{opts}\n\nAnswer with exactly one option name and nothing else."
    content = ([{"type": "image", "image": image}] if image is not None else []) + [{"type": "text", "text": text}]
    return [{"role": "user", "content": content if image is not None else text}]


class Reflex:
    def __init__(self, base: str = "google/gemma-4-E4B-it", skills: dict | None = None, quant: str = "nf4", seats: int = 40):
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        self.proc = AutoProcessor.from_pretrained(base)
        self.tok = self.proc.tokenizer
        kw = {}
        from transformers import AutoConfig
        prequant = getattr(AutoConfig.from_pretrained(base), "quantization_config", None) is not None
        if prequant:
            quant = "none"  # the checkpoint is already quantized (e.g. matu79go/Reflex-1-4B-bnb-4bit)
        if quant != "none":
            skip = ["lm_head", "vision_tower", "audio_tower", "embed_vision", "embed_audio"]
            kw["quantization_config"] = (BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip) if quant == "int8" else
                                         BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                                                            bnb_4bit_use_double_quant=True, llm_int8_skip_modules=skip))
        model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda", **kw)
        self.text_model = next(m for m in model.modules() if m.__class__.__name__ == "Gemma4TextModel")
        cfg = self.text_model.config
        self.H, self.softcap = cfg.hidden_size, getattr(cfg, "final_logit_softcapping", None)
        self.prj = LatentProj(self.H, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input).to("cuda", torch.bfloat16)
        self.seats = seats
        self.slot = torch.zeros(seats, self.H, device="cuda", dtype=torch.bfloat16)
        self.skills, self.parts, self.cur = skills or {}, {}, None
        if self.skills:
            from peft import PeftModel
            for i, (name, sk) in enumerate(self.skills.items()):
                path = sk["path"]
                if i == 0:
                    model = PeftModel.from_pretrained(model, path, adapter_name=name)
                else:
                    model.load_adapter(path, adapter_name=name)
                from safetensors.torch import load_file
                self.parts[name] = load_file(os.path.join(path, "latent_parts.safetensors"), device="cuda")
            self.text_model = next(m for m in model.modules() if m.__class__.__name__ == "Gemma4TextModel")
        self.model = model.eval()
        self.embed_w = model.get_output_embeddings().weight

    # ---------------------------------------------------------------- helpers
    def _off(self):
        return self.model.disable_adapter() if hasattr(self.model, "disable_adapter") else contextlib.nullcontext()

    def _use(self, name):
        if name == self.cur:
            return
        self.model.set_adapter(name)
        p = self.parts[name]
        self.prj.load_state_dict({k[4:]: v for k, v in p.items() if k.startswith("prj.")})
        self.slot = p["slot"].to(torch.bfloat16)
        self.cur = name

    def _logits(self, h):
        lg = (h @ self.embed_w.T).float()
        return torch.tanh(lg / self.softcap) * self.softcap if self.softcap else lg

    def _options(self, question, criteria):
        """Map options to first tokens; if first tokens collide, ask with letters and map back."""
        names = list(criteria)
        first = [self.tok.encode(n, add_special_tokens=False)[0] for n in names]
        if len(set(first)) == len(first):
            return question, dict(criteria), names, first
        opts = {LETTERS[i]: (f"{n}: {d}" if d else n) for i, (n, d) in enumerate(criteria.items())}
        return question + " Reply with the letter only.", opts, names, [self.tok.encode(k, add_special_tokens=False)[0] for k in opts]

    @staticmethod
    def _probs(lg, names):
        p = torch.softmax(lg.float(), -1).tolist()
        probs = {n: round(x, 4) for n, x in zip(names, p)}
        best = max(probs, key=probs.get)
        return {"choice": best, "probs": probs, "confidence": probs[best]}

    # ---------------------------------------------------------------- instant
    @torch.no_grad()
    def instant(self, state: str, instructions: str, criteria: dict, image=None) -> dict:
        q, opts, names, ids = self._options(instructions, criteria)
        msgs = _messages(q, opts, state, image)
        inputs = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt", enable_thinking=False).to("cuda")
        with self._off():
            out = self.model(**inputs, use_cache=False)
        return self._probs(out.logits[0, -1, ids], names)

    # ---------------------------------------------------------------- deep (latent reasoning)
    def trace_line(self, h):
        words = self.tok.decode(self._logits(h).argmax(-1).tolist(), skip_special_tokens=True).split("\n")[0].split()
        out = []
        for w in words:  # seats beyond the line length were not supervised; stop at the first repeat
            if out and w == out[-1]:
                break
            out.append(w)
        return " ".join(out)

    def steps_for(self, skill, state):
        rule = self.skills[skill]["steps"]
        return len(re.findall(rule["regex"], state)) + rule.get("plus", 0)

    @torch.no_grad()
    def deep(self, state: str, skill: str, steps: int | None = None, trace: bool = False) -> dict:
        sk = self.skills[skill]
        self._use(skill)
        task = sk["task"]
        k = int(steps or self.steps_for(skill, state))
        prompt = self.tok.apply_chat_template(_messages(task["question"], task["options"], state), add_generation_prompt=True,
                                              tokenize=False, enable_thinking=True) + BOT
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        M = enc["attention_mask"]
        P = M.shape[1]
        out = self.text_model(input_ids=enc["input_ids"], attention_mask=M, use_cache=True)
        cache, h = out.past_key_values, out.last_hidden_state[:, -1:].expand(1, self.seats, self.H)
        pos = (P + torch.arange(self.seats, device="cuda"))[None]
        lines = []
        for _ in range(k):
            e, p = self.prj(h)
            M = torch.cat([M, torch.ones(1, self.seats, dtype=M.dtype, device="cuda")], 1)
            out = self.text_model(inputs_embeds=e + self.slot[None], per_layer_inputs=p, attention_mask=M, position_ids=pos,
                                  past_key_values=cache, use_cache=True)
            cache, h = out.past_key_values, out.last_hidden_state
            if trace:
                lines.append(self.trace_line(h[0]))
        eot = torch.tensor([self.tok.encode(EOT, add_special_tokens=False)], device="cuda")
        out = self.text_model(input_ids=eot, attention_mask=torch.cat([M, torch.ones_like(eot)], 1),
                              position_ids=(P + self.seats + torch.arange(eot.shape[1], device="cuda"))[None],
                              past_key_values=cache, use_cache=True)
        names = list(task["options"])
        ids = [self.tok.encode(n, add_special_tokens=False)[0] for n in names]
        res = self._probs(self._logits(out.last_hidden_state[0, -1])[ids], names)
        if trace:
            res["trace"] = lines
        return res

    # ---------------------------------------------------------------- free text
    @torch.no_grad()
    def write(self, state: str, instructions: str, max_tokens: int = 256) -> str:
        txt = self.tok.apply_chat_template([{"role": "user", "content": f"{state}\n\n{instructions}"}], add_generation_prompt=True,
                                           tokenize=False, enable_thinking=False)
        ids = self.tok(txt, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
        with self._off():
            g = self.model.generate(input_ids=ids, max_new_tokens=max_tokens, do_sample=False)
        return self.tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()


def load_skills(path: str) -> dict:
    """skills.json: {name: {path, task: {question, options}, steps: {regex, plus}}}.

    path is either a local folder (relative paths resolve next to skills.json) or "hf://<user>/<repo>/<folder>",
    which is downloaded from the Hugging Face Hub on first use.
    """
    sk = json.load(open(path, encoding="utf-8"))
    base = os.path.dirname(os.path.abspath(path))
    for v in sk.values():
        p = v["path"]
        if p.startswith("hf://"):
            from huggingface_hub import snapshot_download
            user, repo, sub = p[5:].split("/", 2)
            v["path"] = os.path.join(snapshot_download(f"{user}/{repo}", allow_patterns=[f"{sub}/*"]), sub)
        elif not p.startswith("/"):
            v["path"] = os.path.join(base, p)
    return sk
