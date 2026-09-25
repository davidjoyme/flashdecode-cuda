"""
Ground-truth oracle.

Runs HuggingFace's own Llama and captures (a) per-layer hidden states on the
prefill pass and (b) the greedy token sequence. Everything the engine
produces is diffed against this.

Run once; regenerate only when MODEL_ID or PARITY_PROMPT changes.

    python -m reference.oracle
"""

from __future__ import annotations

import os
from typing import Dict

import torch

import config

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oracle.pt")


@torch.inference_mode()
def build(model_id: str = config.MODEL_ID,
          prompt: str = config.PARITY_PROMPT,
          max_new: int = config.PARITY_MAX_NEW_TOKENS,
          device: str = config.DEVICE,
          dtype=config.DTYPE) -> Dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()

    ids = tok(prompt, return_tensors="pt").input_ids.to(device)

    acts: Dict[str, torch.Tensor] = {}
    handles = []

    def hook(name):
        def fn(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            acts[name] = t.detach().float().cpu().clone()
        return fn

    # Names must match what engine/model.py records via _rec().
    handles.append(model.model.embed_tokens.register_forward_hook(hook("embed")))
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(hook(f"layer.{i}")))
    handles.append(model.model.layers[0].input_layernorm
                   .register_forward_hook(hook("L0.input_layernorm")))
    handles.append(model.model.norm.register_forward_hook(hook("final_norm")))

    out = model(ids)
    acts["logits"] = out.logits[:, -1].detach().float().cpu().clone()
    for h in handles:
        h.remove()

    gen = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         temperature=None, top_p=None, top_k=None,
                         pad_token_id=tok.eos_token_id)
    new_tokens = gen[0, ids.shape[1]:].tolist()

    payload = {
        "model_id": model_id,
        "prompt": prompt,
        "input_ids": ids.cpu(),
        "activations": acts,
        "greedy_tokens": new_tokens,
        "greedy_text": tok.decode(new_tokens),
        "dtype": str(dtype),
    }
    torch.save(payload, OUT)
    print(f"wrote {OUT}")
    print(f"  prompt : {prompt!r}")
    print(f"  greedy : {payload['greedy_text']!r}")
    print(f"  traced : {len(acts)} tensors")
    return payload


def load() -> Dict:
    if not os.path.exists(OUT):
        raise FileNotFoundError(f"{OUT} missing -- run `python -m reference.oracle` first.")
    return torch.load(OUT, weights_only=False)


if __name__ == "__main__":
    build()