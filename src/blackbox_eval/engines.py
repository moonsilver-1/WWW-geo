# -*- coding: utf-8 -*-
"""Engine wrappers for P0-1: Qwen2.5-VL-3B (cached) and InternVL3-2B-hf (D: cache).
Two operations per state:
  generate(blocks, query) -> answer text (T=0 greedy, max_new_tokens)
  score(blocks, query, ref_answer) -> mean token log-prob of ref answer (teacher-forced)
"""
import os, sys, time
import torch
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

SYSTEM = ("You are a precise research assistant. You are given a web page as a sequence of "
          "tagged evidence blocks. Answer the user's question using ONLY the information in "
          "these blocks. After each factual claim, cite the tag(s) of the evidence blocks "
          "that support it, in the form [EV-2] or [EV-1][EV-4]. If the blocks do not contain "
          "the information needed, reply exactly: INSUFFICIENT EVIDENCE")

def blocks_to_messages(blocks, query, ref_answer=None, system=None, suffix=None):
    content = []
    for b in blocks:
        if b["type"] in ("I", "C"):
            content.append({"type": "text", "text": f"[EV-{b['tag']}] ({'image' if b['type']=='I' else 'chart'})"})
            content.append({"type": "image", "image": b["image"]})
        else:
            kind = "caption" if b.get("caption") else ("table" if b["type"] == "B" else "text")
            content.append({"type": "text", "text": f"[EV-{b['tag']}] ({kind})\n{b['text']}"})
    content.append({"type": "text", "text": f"Question: {query}"})
    if suffix:
        content.append({"type": "text", "text": suffix})
    messages = [{"role": "system", "content": system or SYSTEM},
                {"role": "user", "content": content}]
    if ref_answer is not None:
        messages.append({"role": "assistant", "content": ref_answer})
    return messages

class QwenEngine:
    name = "qwen2.5-vl-3b-instruct"
    repo = "Qwen/Qwen2.5-VL-3B-Instruct"

    def load(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(self.repo, min_pixels=256*28*28, max_pixels=768*28*28)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.repo, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()

    def _inputs(self, messages, add_gen):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_gen)
        images = [Image.open(c["image"]).convert("RGB")
                  for m in messages for c in (m["content"] if isinstance(m["content"], list) else [])
                  if isinstance(c, dict) and c.get("type") == "image"]
        enc = self.processor(text=[text], images=images if images else None,
                             padding=True, return_tensors="pt").to("cuda")
        return enc

    @torch.no_grad()
    def generate(self, blocks, query, max_new_tokens=140, system=None, suffix=None,
                 do_sample=False, temperature=0.0, top_p=1.0):
        enc = self._inputs(blocks_to_messages(blocks, query, system=system, suffix=suffix), add_gen=True)
        t0 = time.time()
        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample,
                      "temperature": (temperature if do_sample else None),
                      "top_p": (top_p if do_sample else None)}
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        out = self.model.generate(**enc, **gen_kwargs)
        gen = out[0][enc.input_ids.shape[1]:]
        txt = self.processor.tokenizer.decode(gen, skip_special_tokens=True)
        return txt.strip(), time.time() - t0

    @torch.no_grad()
    def score(self, blocks, query, ref_answer):
        enc = self._inputs(blocks_to_messages(blocks, query, ref_answer), add_gen=False)
        with_prefix = self._inputs(blocks_to_messages(blocks, query), add_gen=True)
        n_prefix = with_prefix.input_ids.shape[1]
        logits = self.model(**enc).logits[0].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        ids = enc.input_ids[0]
        ans_ids = ids[n_prefix:]
        if len(ans_ids) == 0:
            return float("nan")
        lp = logprobs[n_prefix - 1: len(ids) - 1].gather(1, ans_ids.unsqueeze(1)).squeeze(1)
        return float(lp.mean().item())

class InternVLEngine:
    name = "internvl3-2b-hf"
    repo = "OpenGVLab/InternVL3-2B-hf"

    def load(self):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.processor = AutoProcessor.from_pretrained(
            self.repo, cache_dir="D:/hf_cache")
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.repo, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            cache_dir="D:/hf_cache").to("cuda").eval()

    def _inputs(self, messages, add_gen):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_gen)
        images = [Image.open(c["image"]).convert("RGB")
                  for m in messages for c in (m["content"] if isinstance(m["content"], list) else [])
                  if isinstance(c, dict) and c.get("type") == "image"]
        enc = self.processor(text=[text], images=images if images else None,
                             return_tensors="pt").to("cuda")
        return enc

    @torch.no_grad()
    def generate(self, blocks, query, max_new_tokens=140, system=None, suffix=None,
                 do_sample=False, temperature=0.0, top_p=1.0):
        enc = self._inputs(blocks_to_messages(blocks, query, system=system, suffix=suffix), add_gen=True)
        t0 = time.time()
        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample,
                      "temperature": (temperature if do_sample else None),
                      "top_p": (top_p if do_sample else None)}
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        out = self.model.generate(**enc, **gen_kwargs)
        gen = out[0][enc.input_ids.shape[1]:]
        txt = self.processor.tokenizer.decode(gen, skip_special_tokens=True)
        return txt.strip(), time.time() - t0

    @torch.no_grad()
    def score(self, blocks, query, ref_answer):
        enc = self._inputs(blocks_to_messages(blocks, query, ref_answer), add_gen=False)
        with_prefix = self._inputs(blocks_to_messages(blocks, query), add_gen=True)
        n_prefix = with_prefix.input_ids.shape[1]
        logits = self.model(**enc).logits[0].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        ids = enc.input_ids[0]
        ans_ids = ids[n_prefix:]
        if len(ans_ids) == 0:
            return float("nan")
        lp = logprobs[n_prefix - 1: len(ids) - 1].gather(1, ans_ids.unsqueeze(1)).squeeze(1)
        return float(lp.mean().item())

def load_engine(key):
    eng = {"qwen3b": QwenEngine, "internvl3-2b": InternVLEngine}[key]()
    eng.load()
    return eng
