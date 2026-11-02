"""Glimmer: Muse Glimmer 30B fine-tuned on this lab's own dips to propose the first recipe.

The lab's records are what it learns from: every dip is a recipe and the colour SAM 3.1
measured on the dried cloth (a real-looking photo, card-calibrated). Glimmer learns the
inverse for this dye lot and this camera: target colour -> the recipe that dyes it.
Training is supervised fine-tuning (LoRA) on Fireworks; the weights are published.

    uv run python eval/glimmer.py data          # the lab's records -> eval/glimmer/{train,test}.jsonl
    uv run python eval/glimmer.py train         # upload, start the SFT job, wait
    uv run python eval/glimmer.py deploy        # on-demand deployment of the tuned model (billed hourly)
    uv run python eval/glimmer.py score <run>   # a Serverless Training run's base vs tuned answers, by the colour they dye
    uv run python eval/glimmer.py eval          # same, live against a deployed Glimmer
    uv run python eval/glimmer.py download      # LoRA adapter -> eval/glimmer/adapter/
    uv run --with huggingface_hub python eval/glimmer.py publish   # adapter + dip records -> Hugging Face (private)
    uv run python eval/glimmer.py undeploy      # delete the deployment (stop billing)
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).parent / "glimmer"
load_dotenv(Path(__file__).parents[1] / ".env")

from rangrez.colour import delta_e_2000
from rangrez.glimmer import SYSTEM, answer, first_guess, parse, prompt
from rangrez.km import fit_gains, solve_recipe
from rangrez.photoreal import scene_names
from rangrez.profiles import PROFILE
from rangrez.rig import Recipe
from rangrez.sim import VirtualRig
from rangrez.vision import measure

BASE_MODEL = "accounts/fireworks/models/muse-glimmer-30b"
API = "https://api.fireworks.ai/v1"
JOB_ID = os.environ.get("GLIMMER_JOB", "rangrez-glimmer-v1")  # a Managed SFT job, if dataset uploads are enabled
# The tuned model from Serverless Training (run with the Fireworks cookbook's Muse Glimmer starter;
# its base vs tuned answers and summary are in eval/glimmer/run/).
MODEL_ID = os.environ.get("GLIMMER_MODEL_ID", "muse-starter-sl-rangrez-glimmer-v1")


# -- the lab's records ---------------------------------------------------------------
def random_recipe(rng: np.random.Generator) -> np.ndarray:
    """Recipes a dye house actually runs: mostly one or two main dyes, some trace shading."""
    conc = rng.uniform(0, 1, 3) ** 2 * PROFILE.max_conc * 0.7
    conc[rng.random(3) < 0.25] = 0.0
    return np.round(conc, 2)


def one_dip(i: int) -> dict | None:
    rng = np.random.default_rng(10_000 + i)
    scenes = [s for s in scene_names() if s not in {"phone_flash", "no_paper"}]
    rig = VirtualRig(seed=i, scene=scenes[i % len(scenes)])
    conc = random_recipe(rng)
    rig.last_swatch_lab = rig.true_swatch_lab(Recipe(tuple(conc)))
    m = measure(rig.render(), rig.segmenter())
    if not m.ok:
        return None
    return {"conc": conc.tolist(), "measured_lab": m.lab.tolist(), "true_lab": rig.last_swatch_lab.tolist(), "scene": rig.scene}


def cmd_data(n_train: int = 3000, n_test: int = 200) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor() as pool:
        dips = [d for d in pool.map(one_dip, range(n_train + n_test), chunksize=20) if d]
    train, test = dips[:n_train], dips[n_train:]
    with open(ROOT / "train.jsonl", "w") as f:
        f.writelines(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt(d["measured_lab"])},
                {"role": "assistant", "content": answer(d["conc"])},
            ]}) + "\n" for d in train)
    (ROOT / "records.json").write_text(json.dumps({"train": train, "test": test}))
    print(f"{len(train)} training dips, {len(test)} held-out targets -> {ROOT}")


# -- Fireworks -----------------------------------------------------------------------
def account() -> str:
    acct = os.environ.get("FIREWORKS_ACCOUNT_ID", "").strip()
    if not acct or not os.environ.get("FIREWORKS_API_KEY", "").strip():
        sys.exit("set FIREWORKS_API_KEY and FIREWORKS_ACCOUNT_ID in .env")
    return acct


def call(method: str, path: str, body: dict | None = None, data: bytes | None = None, ctype: str = "application/json") -> dict:
    req = urllib.request.Request(API + path, method=method, headers={
        "Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY'].strip()}", "Content-Type": ctype,
    }, data=data if data is not None else (json.dumps(body).encode() if body is not None else None))
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        sys.exit(f"{method} {path} -> {exc.code}: {exc.read().decode()[:600]}")


def cmd_train() -> None:
    acct = account()
    path = ROOT / "train.jsonl"
    n = len(path.read_text().splitlines())
    ds = f"{JOB_ID}-data"
    existing = {d["name"].rsplit("/", 1)[-1] for d in call("GET", f"/accounts/{acct}/datasets?pageSize=200").get("datasets", [])}
    if ds not in existing:
        call("POST", f"/accounts/{acct}/datasets", {"datasetId": ds, "dataset": {
            "displayName": "Rangrez lab dips (recipe <- measured colour)", "exampleCount": str(n), "userUploaded": {}, "format": "CHAT"}})
        boundary = "rangrez" + str(int(time.time()))
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"train.jsonl\"\r\n"
                "Content-Type: application/jsonl\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        call("POST", f"/accounts/{acct}/datasets/{ds}:upload", data=body, ctype=f"multipart/form-data; boundary={boundary}")
    while (state := call("GET", f"/accounts/{acct}/datasets/{ds}").get("state")) != "READY":
        print("dataset", state)
        time.sleep(5)
    jobs = {j["name"].rsplit("/", 1)[-1] for j in call("GET", f"/accounts/{acct}/supervisedFineTuningJobs?pageSize=200").get("supervisedFineTuningJobs", [])}
    if JOB_ID not in jobs:
        call("POST", f"/accounts/{acct}/supervisedFineTuningJobs?supervisedFineTuningJobId={JOB_ID}", {
            "dataset": f"accounts/{acct}/datasets/{ds}", "baseModel": BASE_MODEL, "outputModel": f"accounts/{acct}/models/{JOB_ID}",
            "displayName": "Glimmer: Rangrez first-dip colourist", "epochs": 2, "loraRank": 16, "learningRate": 1e-4,
        })
    while True:
        job = call("GET", f"/accounts/{acct}/supervisedFineTuningJobs/{JOB_ID}")
        print(time.strftime("%H:%M:%S"), job.get("state"), json.dumps(job.get("jobProgress", {}))[:160])
        if job.get("state") in ("JOB_STATE_COMPLETED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
            print(json.dumps(job.get("status", {}))[:400])
            return
        time.sleep(30)


def cmd_deploy() -> None:
    acct = account()
    dep = call("POST", f"/accounts/{acct}/deployments?deploymentId={MODEL_ID}", {
        "baseModel": f"accounts/{acct}/models/{MODEL_ID}", "deploymentShape": "default", "minReplicaCount": 0, "maxReplicaCount": 1,
        "autoscalingPolicy": {"scaleToZeroWindow": "900s"}, "displayName": "Glimmer (Rangrez)",
    })
    while (state := call("GET", f"/accounts/{acct}/deployments/{MODEL_ID}").get("state")) != "READY":
        print(time.strftime("%H:%M:%S"), "deployment", state)
        if state == "FAILED":
            sys.exit(json.dumps(call("GET", f"/accounts/{acct}/deployments/{MODEL_ID}"))[:600])
        time.sleep(20)
    print("ready:", dep.get("name", f"accounts/{acct}/deployments/{MODEL_ID}"))


def cmd_undeploy() -> None:
    acct = account()
    call("DELETE", f"/accounts/{acct}/deployments/{MODEL_ID}?hard=true")
    print("deleted deployment", MODEL_ID)


def cmd_download() -> None:
    acct = account()
    urls = call("GET", f"/accounts/{acct}/models/{MODEL_ID}:getDownloadEndpoint").get("filenameToSignedUrls", {})
    out = ROOT / "adapter"
    out.mkdir(parents=True, exist_ok=True)
    for name, url in urls.items():
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        print(name, dest.stat().st_size)


# -- does it help? ----------------------------------------------------------------------
def cmd_eval() -> None:
    acct = account()
    recs = json.loads((ROOT / "records.json").read_text())
    test = recs["test"][: int(os.environ.get("N", "100"))]
    world = VirtualRig()
    supplier = PROFILE.nominal()
    refit = fit_gains(supplier, [(np.array(d["conc"]), np.array(d["measured_lab"])) for d in recs["train"][:400]])
    tuned = f"accounts/{acct}/models/{MODEL_ID}#accounts/{acct}/deployments/{MODEL_ID}"
    models = {"supplier model": lambda lab: solve_recipe(supplier, lab, PROFILE.max_conc)[0],
              "model refitted to the lab's dips": lambda lab: solve_recipe(refit, lab, PROFILE.max_conc)[0],
              "Glimmer (fine-tuned)": lambda lab: first_guess(lab, model=tuned)}
    if os.environ.get("BASE_GLIMMER"):
        models["Glimmer (base, zero-shot)"] = lambda lab: first_guess(lab, model=BASE_MODEL)
    results = {}
    for name, fn in models.items():
        errs, fails = [], 0
        for d in test:
            conc = fn(np.array(d["measured_lab"]))
            if conc is None:
                fails += 1
                continue
            errs.append(float(delta_e_2000(world.true_swatch_lab(Recipe(tuple(conc))), np.array(d["true_lab"]))))
        results[name] = {"median": float(np.median(errs)), "p90": float(np.percentile(errs, 90)), "under_3": float(np.mean(np.array(errs) < 3)), "unparsed": fails, "n": len(test)}
        print(f"{name:34s} first-dip dE00 median {results[name]['median']:.2f}  p90 {results[name]['p90']:.2f}  "
              f"under 3: {results[name]['under_3']:.0%}  unparsed {fails}")
    (ROOT / "eval.json").write_text(json.dumps(results, indent=2))


def cmd_score() -> None:
    """Score a Serverless Training run's base and tuned answers by the colour they really dye."""
    run = Path(sys.argv[2])
    recs = json.loads((ROOT / "records.json").read_text())
    by_prompt = {prompt(d["measured_lab"]): d for d in recs["train"]}
    holdout = [json.loads(line) for line in (run / "holdout.jsonl").read_text().splitlines()]
    targets = [by_prompt[next(m["content"] for m in h["messages"] if m["role"] == "user")] for h in holdout]
    held = {id(d) for d in targets}
    world = VirtualRig()
    supplier = PROFILE.nominal()
    seen = [d for d in recs["train"] if id(d) not in held][:400]
    refit = fit_gains(supplier, [(np.array(d["conc"]), np.array(d["measured_lab"])) for d in seen])

    def score(recipes: list) -> dict:
        errs = [float(delta_e_2000(world.true_swatch_lab(Recipe(tuple(c))), np.array(d["true_lab"])))
                for c, d in zip(recipes, targets) if c is not None]
        e = np.array(errs)
        return {"median": round(float(np.median(e)), 2), "mean": round(float(e.mean()), 2), "p90": round(float(np.percentile(e, 90)), 2),
                "under_3": round(float((e < 3).mean()), 2), "answered": len(errs), "n": len(targets)}

    out = {
        "supplier model (today's first dip)": score([solve_recipe(supplier, np.array(d["measured_lab"]), PROFILE.max_conc)[0] for d in targets]),
        "model refitted to 400 of the lab's dips": score([solve_recipe(refit, np.array(d["measured_lab"]), PROFILE.max_conc)[0] for d in targets]),
    }
    for label in ("base", "tuned"):
        preds = [json.loads(line) for line in (run / f"predictions_{label}.jsonl").read_text().splitlines()]
        recipes = [None] * len(targets)
        for p in preds:
            recipes[p["index"]] = parse(p["answer_text"])
        out[f"Muse Glimmer 30B, {label}"] = score(recipes)
    for name, r in out.items():
        print(f"{name:42s} first-dip dE00 median {r['median']:5.2f}  mean {r['mean']:5.2f}  p90 {r['p90']:5.2f}  "
              f"under 3: {r['under_3']:.0%}  answered {r['answered']}/{r['n']}")
    (ROOT / "eval.json").write_text(json.dumps(out, indent=2))


def cmd_publish() -> None:
    """Upload the adapter and the lab's dip records to Hugging Face (private; flip public at submission)."""
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"].strip())
    user = api.whoami()["name"]
    adapter = ROOT / "adapter"
    if not any(adapter.glob("*")):
        sys.exit("no adapter yet: run `eval/glimmer.py download` first")
    results = json.loads((ROOT / "eval.json").read_text()) if (ROOT / "eval.json").exists() else {}
    table = "\n".join(f"| {k} | {v['median']} | {v['p90']} | {v['under_3']:.0%} |" for k, v in results.items())
    card = f"""---
base_model: meta-models/Muse-Glimmer-30B
library_name: peft
license: apache-2.0
tags: [lora, colour, textile, dyeing, rangrez]
---

# Glimmer for Rangrez: first-dip colourist (LoRA on Muse Glimmer 30B)

Given a target colour in CIELAB, proposes the first dye recipe (turquoise, fuchsia and lemon yellow
reactive dyes, g/L) for the Rangrez self-driving dye lab. Fine-tuned with LoRA (rank 8, 2 epochs,
364 steps) on Fireworks Serverless Training, on 2,900 of the lab's recorded dips: each the recipe dipped and the
colour SAM 3.1 measured on the dried cloth in a real-looking phone photo. Dataset:
[{user}/rangrez-lab-dips](https://huggingface.co/datasets/{user}/rangrez-lab-dips).

The dips come from Rangrez's simulator (a hidden dye lot plus realistic photos), not from a
physical dye house, and the adapter knows that lot.

## First-dip error on 100 held-out targets (CIEDE2000 of the colour the recipe really dyes)

| Method | Median | p90 | Under 3 |
| --- | --- | --- | --- |
{table}

Code: https://github.com/TusharTechs/rangrez
"""
    model_repo, data_repo = f"{user}/rangrez-glimmer-lora", f"{user}/rangrez-lab-dips"
    api.create_repo(model_repo, private=True, exist_ok=True)
    (adapter / "README.md").write_text(card)
    api.upload_folder(folder_path=str(adapter), repo_id=model_repo, commit_message="Glimmer LoRA for Rangrez")
    api.create_repo(data_repo, repo_type="dataset", private=True, exist_ok=True)
    for f in ("train.jsonl", "records.json"):
        api.upload_file(path_or_fileobj=str(ROOT / f), path_in_repo=f, repo_id=data_repo, repo_type="dataset")
    print(f"https://huggingface.co/{model_repo}\nhttps://huggingface.co/datasets/{data_repo}")


if __name__ == "__main__":
    {"data": cmd_data, "train": cmd_train, "deploy": cmd_deploy, "undeploy": cmd_undeploy,
     "download": cmd_download, "eval": cmd_eval, "score": cmd_score, "publish": cmd_publish}[sys.argv[1]]()
