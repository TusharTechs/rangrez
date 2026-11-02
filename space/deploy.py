"""Publish the dashboard to a Hugging Face Space (private until you flip it).

    uv run --with huggingface_hub python space/deploy.py [TechTush/rangrez] [--public] [--sleep SECONDS]

Uploads only src/, pyproject.toml, uv.lock and this folder's Dockerfile and README.
No .env, recordings or keys. After it is running, add the Space secrets in its settings:
MODEL_API_KEY (your Meta key) and RANGREZ_LIVE_PASSCODE (any phrase you choose).
It sets two public variables: 200 free colour reads a day for visitors, at most 5 each.

The Space stays awake by default (--sleep -1), so a judge never waits for it to wake;
that costs about $0.03 an hour on cpu-upgrade. --sleep 3600 sends it to sleep after an
idle hour. --public makes it visible to everyone.
"""
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).parents[1]
load_dotenv(ROOT / ".env")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
repo = args[0] if args and "/" in args[0] else "TechTush/rangrez"
sleep = int(sys.argv[sys.argv.index("--sleep") + 1]) if "--sleep" in sys.argv else -1
api = HfApi(token=os.environ["HF_TOKEN"].strip())
# Docker Spaces on the free tier need PRO; the paid "cpu-upgrade" tier (about $0.03/hour,
# billed to HF credits) does not.
api.create_repo(repo, repo_type="space", space_sdk="docker", space_hardware="cpu-upgrade",
                space_sleep_time=sleep, private="--public" not in sys.argv, exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    shutil.copytree(ROOT / "src", Path(tmp) / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("pyproject.toml", "uv.lock"):
        shutil.copy(ROOT / f, tmp)
    for f in ("Dockerfile", "README.md"):
        shutil.copy(ROOT / "space" / f, tmp)
    api.upload_folder(folder_path=tmp, repo_id=repo, repo_type="space", commit_message="Rangrez dashboard")
# Public free colour reads (SAM 3.1 + the home recipe): capped so the demo can't run up a bill.
api.add_space_variable(repo, "RANGREZ_PUBLIC_READS_PER_DAY", "200")
api.add_space_variable(repo, "RANGREZ_PUBLIC_READS_PER_VISITOR", "5")
api.set_space_sleep_time(repo, sleep)  # -1: never sleep
if "--public" in sys.argv:
    api.update_repo_settings(repo, private=False, repo_type="space")
print(f"https://huggingface.co/spaces/{repo}")
for _ in range(90):
    stage = api.get_space_runtime(repo).stage
    print(time.strftime("%H:%M:%S"), stage, flush=True)
    if stage in ("RUNNING", "BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"):
        break
    time.sleep(20)
