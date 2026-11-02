"""Publish the dashboard to a Hugging Face Space (private until you flip it).

    uv run --with huggingface_hub python space/deploy.py [TechTush/rangrez]

Uploads only src/, pyproject.toml, uv.lock and this folder's Dockerfile and README.
No .env, recordings or keys. After it is running, add the Space secrets in its settings:
MODEL_API_KEY (your Meta key) and RANGREZ_LIVE_PASSCODE (any phrase you choose).
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
repo = sys.argv[1] if len(sys.argv) > 1 else "TechTush/rangrez"
api = HfApi(token=os.environ["HF_TOKEN"].strip())
# Docker Spaces on the free tier need PRO; the paid "cpu-upgrade" tier (about $0.03/hour,
# billed to HF credits) does not. It sleeps after an hour without visitors, so it bills only while watched.
api.create_repo(repo, repo_type="space", space_sdk="docker", space_hardware="cpu-upgrade",
                space_sleep_time=3600, private=True, exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    shutil.copytree(ROOT / "src", Path(tmp) / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("pyproject.toml", "uv.lock"):
        shutil.copy(ROOT / f, tmp)
    for f in ("Dockerfile", "README.md"):
        shutil.copy(ROOT / "space" / f, tmp)
    api.upload_folder(folder_path=tmp, repo_id=repo, repo_type="space", commit_message="Rangrez dashboard")
print(f"https://huggingface.co/spaces/{repo}")
for _ in range(90):
    stage = api.get_space_runtime(repo).stage
    print(time.strftime("%H:%M:%S"), stage, flush=True)
    if stage in ("RUNNING", "BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"):
        break
    time.sleep(20)
