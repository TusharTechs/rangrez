"""Glimmer: Muse Glimmer 30B fine-tuned on this lab's own dips to propose the first recipe.

The lab's records are what it learns from: every dip is a recipe and the colour SAM 3.1
measured on the dried cloth (a real-looking photo, card-calibrated). Glimmer learns the
inverse for this dye lot and this camera: target colour -> the recipe that dyes it.
Training is supervised fine-tuning (LoRA) on Fireworks; the weights are published.

    uv run python eval/glimmer.py data          # the lab's records -> eval/glimmer/{train,test}.jsonl
    uv run python eval/glimmer.py train         # upload, start the SFT job, wait
    uv run python eval/glimmer.py deploy        # on-demand deployment of the tuned model (billed hourly)
    uv run python eval/glimmer.py eval          # first-dip error: supplier model, refitted model, Glimmer
    uv run python eval/glimmer.py download      # LoRA adapter -> eval/glimmer/adapter/
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

from rangrez.colour import delta_e_2000, lab_to_hex  # noqa: E402
from rangrez.km import fit_gains, solve_recipe  # noqa: E402
from rangrez.photoreal import scene_names  # noqa: E402
from rangrez.profiles import PROFILE  # noqa: E402
from rangrez.rig import DYES, Recipe  # noqa: E402
from rangrez.sim import VirtualRig  # noqa: E402
from rangrez.vision import measure  # noqa: E402

BASE_MODEL = "accounts/fireworks/models/muse-glimmer-30b"
API = "https://api.fireworks.ai/v1"
JOB_ID = os.environ.get("GLIMMER_JOB", "rangrez-glimmer-v1")
SYSTEM = (
    f"You are Glimmer, the colourist of a small cotton dye lab. Dyes: {', '.join(DYES)} "
    f"({PROFILE.description}). Given a target colour measured in CIELAB (D65), reply with only the "
    f"recipe for the first dip as JSON with one key per dye, in {PROFILE.unit}, each between 0 and {PROFILE.max_conc:g}."
)


def prompt(lab) -> str:
    L, a, b = (float(v) for v in lab)
    return f"Target L* {L:.1f}, a* {a:.1f}, b* {b:.1f} ({lab_to_hex(np.array([L, a, b]))}). First dip recipe?"


def answer(conc) -> str:
    return json.dumps({d: round(float(c), 2) for d, c in zip(DYES, conc)})


def parse(text: str) -> np.ndarray | None:
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        obj = json.loads(text[start:end])
        return np.clip([float(obj.get(d, 0.0)) for d in DYES], 0, PROFILE.max_conc)
    except (ValueError, json.JSONDecodeError, TypeError):
        return None


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
        for d in train:
            f.write(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt(d["measured_lab"])},
                {"role": "assistant", "content": answer(d["conc"])},
            ]}) + "\n")
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
    n = sum(1 for _ in open(path))
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
    dep = call("POST", f"/accounts/{acct}/deployments?deploymentId={JOB_ID}", {
        "baseModel": f"accounts/{acct}/models/{JOB_ID}", "deploymentShape": "default", "minReplicaCount": 0, "maxReplicaCount": 1,
        "autoscalingPolicy": {"scaleToZeroWindow": "900s"}, "displayName": "Glimmer (Rangrez)",
    })
    while (state := call("GET", f"/accounts/{acct}/deployments/{JOB_ID}").get("state")) != "READY":
        print(time.strftime("%H:%M:%S"), "deployment", state)
        if state == "FAILED":
            sys.exit(json.dumps(call("GET", f"/accounts/{acct}/deployments/{JOB_ID}"))[:600])
        time.sleep(20)
    print("ready:", dep.get("name", f"accounts/{acct}/deployments/{JOB_ID}"))


def cmd_undeploy() -> None:
    acct = account()
    call("DELETE", f"/accounts/{acct}/deployments/{JOB_ID}?hard=true")
    print("deleted deployment", JOB_ID)


def cmd_download() -> None:
    acct = account()
    urls = call("GET", f"/accounts/{acct}/models/{JOB_ID}:getDownloadEndpoint").get("filenameToSignedUrls", {})
    out = ROOT / "adapter"
    out.mkdir(parents=True, exist_ok=True)
    for name, url in urls.items():
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        print(name, dest.stat().st_size)


# -- does it help? ----------------------------------------------------------------------
def glimmer_recipe(lab, model: str) -> np.ndarray | None:
    from openai import OpenAI

    client = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=os.environ["FIREWORKS_API_KEY"].strip())
    r = client.chat.completions.create(model=model, temperature=0, max_tokens=200, messages=[
        {"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(lab)}])
    return parse(r.choices[0].message.content or "")


def cmd_eval() -> None:
    acct = account()
    recs = json.loads((ROOT / "records.json").read_text())
    test = recs["test"][: int(os.environ.get("N", "100"))]
    world = VirtualRig()
    supplier = PROFILE.nominal()
    refit = fit_gains(supplier, [(np.array(d["conc"]), np.array(d["measured_lab"])) for d in recs["train"][:400]])
    tuned = f"accounts/{acct}/models/{JOB_ID}#accounts/{acct}/deployments/{JOB_ID}"
    models = {"supplier model": lambda lab: solve_recipe(supplier, lab, PROFILE.max_conc)[0],
              "model refitted to the lab's dips": lambda lab: solve_recipe(refit, lab, PROFILE.max_conc)[0],
              "Glimmer (fine-tuned)": lambda lab: glimmer_recipe(lab, tuned)}
    if os.environ.get("BASE_GLIMMER"):
        models["Glimmer (base, zero-shot)"] = lambda lab: glimmer_recipe(lab, BASE_MODEL)
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


if __name__ == "__main__":
    {"data": cmd_data, "train": cmd_train, "deploy": cmd_deploy, "undeploy": cmd_undeploy,
     "download": cmd_download, "eval": cmd_eval}[sys.argv[1]]()
