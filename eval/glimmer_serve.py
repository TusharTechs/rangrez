"""Serve the published Glimmer (Muse Glimmer 30B + the Rangrez LoRA) on Modal for the live demo.

An OpenAI-style /v1/chat/completions endpoint, so the lab calls it exactly as it would call
Fireworks. Base weights and adapter come from the Modal volume the training run filled; nothing
is downloaded from Hugging Face. It scales to zero 10 minutes after the last request.

    uv run --with modal modal run eval/glimmer_serve.py       # smoke test: load, answer 3 targets, exit
    uv run --with modal modal deploy eval/glimmer_serve.py    # serve; prints the URL
    # then in .env: GLIMMER_MODEL=rangrez-glimmer  GLIMMER_BASE_URL=<url>/v1  (GLIMMER_API_KEY is set for you)
    uv run --with modal modal app stop rangrez-glimmer-serve  # stop billing when the recording is done
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

import modal

ROOT = Path(__file__).parents[1]
BASE = "meta-models/Muse-Glimmer-30B"
REVISION = "cec94fcfd6512f2cfe708138c32533b346ea4f69"


def api_key() -> str:
    """A bearer key for the endpoint, generated once and kept in .env (gitignored)."""
    env = ROOT / ".env"
    text = env.read_text() if env.exists() else ""
    for line in text.splitlines():
        if line.startswith("GLIMMER_API_KEY=") and line.split("=", 1)[1].strip():
            return line.split("=", 1)[1].strip()
    key = "rgz-" + secrets.token_urlsafe(24)
    with open(env, "a") as f:
        f.write(f"\n# Bearer key for the Modal Glimmer endpoint (eval/glimmer_serve.py)\nGLIMMER_API_KEY={key}\n")
    return key


app = modal.App("rangrez-glimmer-serve")
volume = modal.Volume.from_name("rangrez-glimmer", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.15.0", "transformers==5.22.1", "peft==0.21.2", "accelerate==1.15.0", "safetensors>=0.8.0",
    "huggingface_hub>=1.31.0", "fastapi[standard]",
)
KEY = api_key() if modal.is_local() else os.environ.get("GLIMMER_API_KEY", "")


@app.cls(gpu="A100-80GB", image=image, volumes={"/vol": volume}, scaledown_window=600, timeout=900,
         secrets=[modal.Secret.from_dict({"GLIMMER_API_KEY": KEY})])
class Glimmer:
    @modal.enter()
    def load(self) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        path = snapshot_download(BASE, revision=REVISION, cache_dir="/vol/hf", local_files_only=True)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        base = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
        self.model = PeftModel.from_pretrained(base, "/vol/out/adapter").eval()

    def complete(self, messages: list[dict], max_tokens: int = 96) -> str:
        import torch

        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).replace("to=user", "", 1).strip()

    @modal.method()
    def answer(self, messages: list[dict]) -> str:
        return self.complete(messages)

    @modal.asgi_app()
    def web(self):
        import time
        import uuid

        from fastapi import FastAPI, Header, HTTPException

        api = FastAPI()

        @api.post("/v1/chat/completions")
        def chat(body: dict, authorization: str = Header("")) -> dict:
            if not secrets.compare_digest(authorization.encode(), f"Bearer {os.environ['GLIMMER_API_KEY']}".encode()):
                raise HTTPException(401, "bad key")
            text = self.complete(body["messages"], int(body.get("max_tokens") or 96))
            return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()),
                    "model": body.get("model", "rangrez-glimmer"),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

        return api


@app.local_entrypoint()
def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from rangrez.glimmer import SYSTEM, parse, prompt

    g = Glimmer()
    for lab in ([61.4, -5.6, -15.0], [55.0, 35.0, 4.0], [72.0, 4.0, 52.0]):
        text = g.answer.remote([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(lab)}])
        print(lab, "->", text, "->", parse(text))
