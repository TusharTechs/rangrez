"""Fine-tune Muse Glimmer 30B on Modal (one H100): weights we own and can publish.

Same data and split as the Fireworks Serverless Training run (eval/glimmer/run/): 2,900 of the
lab's dips for training, the same 100 held out. LoRA rank 8 on the text model's attention and MLP
projections, bf16, 2 epochs, batch 16, learning rate 1e-4. After training it answers the held-out
targets greedily and returns the adapter, so `eval/glimmer.py score` can judge it by the colour
its recipes really dye.

    uv run --with modal modal setup                          # once: log in to Modal in the browser
    uv run --with modal modal run eval/glimmer_modal.py      # about an hour on one H100

Writes eval/glimmer/adapter/ (the LoRA) and eval/glimmer/modal_run/ (predictions, training log).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import modal

BASE = "meta-models/Muse-Glimmer-30B"
REVISION = "cec94fcfd6512f2cfe708138c32533b346ea4f69"
ROOT = Path(__file__).parent / "glimmer"

app = modal.App("rangrez-glimmer")
volume = modal.Volume.from_name("rangrez-glimmer", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.15.0", "transformers==5.22.1", "peft==0.21.2", "accelerate==1.15.0", "safetensors>=0.8.0", "huggingface_hub>=1.31.0",
)


@app.function(gpu="H100", image=image, volumes={"/vol": volume}, timeout=4 * 3600)
def train(rows: list[str], holdout: list[str], epochs: int = 2, batch: int = 16, micro: int = 8,
          lr: float = 1e-4, rank: int = 8, alpha: int = 32) -> dict:
    import math
    import random
    import time

    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    started = time.time()
    path = snapshot_download(BASE, revision=REVISION, cache_dir="/vol/hf")
    volume.commit()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False

    # LoRA on the language model only; the vision tower is untouched (text-only task).
    names = sorted({n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "vision" not in n
                    and n.rsplit(".", 1)[-1] in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")})
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=names))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def encode(example: dict) -> tuple[list[int], list[int]]:
        msgs = example["messages"]
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs, tokenize=False)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        f_ids = tok(full, add_special_tokens=False).input_ids
        k = 0  # train only on the reply: everything the prompt rendered is context
        while k < min(len(p_ids), len(f_ids)) and p_ids[k] == f_ids[k]:
            k += 1
        return f_ids, [-100] * k + f_ids[k:]

    data = [encode(json.loads(r)) for r in rows]
    sample_prompt = tok.apply_chat_template(json.loads(rows[0])["messages"], tokenize=False)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    steps = math.ceil(len(data) / batch) * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.1, 1 - s / steps))
    log, step, rng = [], 0, random.Random(0)
    model.train()
    for epoch in range(epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for b0 in range(0, len(order), batch):
            idx = order[b0:b0 + batch]
            n_tok = sum(sum(1 for t in data[i][1] if t != -100) for i in idx)
            total = 0.0
            for m0 in range(0, len(idx), micro):
                chunk = [data[i] for i in idx[m0:m0 + micro]]
                width = max(len(ids) for ids, _ in chunk)
                ids = torch.tensor([x + [pad] * (width - len(x)) for x, _ in chunk], device="cuda")
                lab = torch.tensor([y + [-100] * (width - len(y)) for _, y in chunk], device="cuda")
                att = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in chunk], device="cuda")
                logits = model(input_ids=ids, attention_mask=att).logits.float()
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)), lab[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
                (loss / n_tok).backward()
                total += loss.item()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            log.append({"step": step, "epoch": epoch + 1, "loss": total / n_tok, "lr": sched.get_last_lr()[0]})
            if step % 10 == 0 or step == 1:
                print(json.dumps(log[-1]), flush=True)

    model.save_pretrained("/vol/out/adapter")
    volume.commit()

    # Answer the held-out targets, greedily, exactly as the lab would ask.
    model.eval()
    model.config.use_cache = True
    tok.padding_side = "left"
    preds = []
    for h0 in range(0, len(holdout), 16):
        exs = [json.loads(h) for h in holdout[h0:h0 + 16]]
        prompts = [tok.apply_chat_template(e["messages"][:-1], tokenize=False, add_generation_prompt=True) for e in exs]
        enc = tok(prompts, add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=96, do_sample=False, pad_token_id=pad)
        for i, e in enumerate(exs):
            text = tok.decode(out[i, enc.input_ids.shape[1]:], skip_special_tokens=True)
            preds.append({"index": h0 + i, "gold": e["messages"][-1]["content"], "answer_text": text})
    files = {p.name: p.read_bytes() for p in Path("/vol/out/adapter").iterdir() if p.is_file()}
    return {"log": log, "predictions": preds, "adapter": files, "trainable_params": trainable, "lora_modules": len(names),
            "minutes": round((time.time() - started) / 60, 1), "rendered_example": sample_prompt[-400:]}


@app.local_entrypoint()
def main(epochs: int = 2) -> None:
    run = ROOT / "run"  # the Fireworks run's split, for a like-for-like comparison
    rows = (run / "train.jsonl").read_text().splitlines()
    holdout = (run / "holdout.jsonl").read_text().splitlines()
    print(f"training on {len(rows)} dips, {len(holdout)} held out, on Modal (H100)")
    res = train.remote(rows, holdout, epochs=epochs)
    out = ROOT / "modal_run"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(run / "holdout.jsonl", out / "holdout.jsonl")
    shutil.copy(run / "predictions_base.jsonl", out / "predictions_base.jsonl")  # base Glimmer, same holdout
    (out / "predictions_tuned.jsonl").write_text("".join(json.dumps(p) + "\n" for p in res["predictions"]))
    (out / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in res["log"]))
    adapter = ROOT / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    for name, data in res["adapter"].items():
        (adapter / name).write_bytes(data)
    info = {"trained_on": "Modal, 1x H100", "base_model": f"{BASE}@{REVISION}", "method": "LoRA (PEFT), bf16",
            "rank": 8, "alpha": 32, "epochs": epochs, "batch": 16, "learning_rate": 1e-4, "train_rows": len(rows),
            "trainable_params": res["trainable_params"], "lora_modules": res["lora_modules"], "minutes": res["minutes"],
            "final_loss": res["log"][-1]["loss"]}
    (adapter / "training.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print("rendered example tail:", res["rendered_example"])
    print("next: uv run python eval/glimmer.py score eval/glimmer/modal_run")
