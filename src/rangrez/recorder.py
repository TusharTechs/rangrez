"""Every campaign, recorded to disk for review and for the demo video.

    recordings/2026-11-02_101500_SerialRig/
        meta.json          rig, dyes, agent, scene, tolerance
        target.jpg         the customer's photo, if the target came from one (+ target.json)
        events.jsonl       every event: Meta API calls with timings, Wonder's notes and tool calls,
                           dispenses, gate refusals, measurements, status
        dip01_photo.jpg    each dip exactly as the camera or phone sent it
        dip01_sam.jpg      the same photo with SAM 3.1's card and strip boxes drawn on
        state.json         the final campaign state

Turn one into a video with `uv run python eval/demo_reel.py recordings/<run>`.
Set RANGREZ_RECORD=0 to switch it off.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path


class Recorder:
    def __init__(self, root: str | None = None) -> None:
        self.root = Path(root or os.environ.get("RANGREZ_RECORD_DIR", "recordings"))
        self.enabled = os.environ.get("RANGREZ_RECORD", "1") != "0"
        self.dir: Path | None = None
        self.pending: list[tuple[str, bytes]] = []  # made before the campaign starts (the target photo)
        self.lock = threading.Lock()

    def stash(self, name: str, data: bytes | str | dict) -> None:
        """Keep something for the next campaign's folder."""
        self.pending.append((name, self._bytes(data)))

    def begin(self, label: str, meta: dict) -> Path | None:
        if not self.enabled:
            return None
        with self.lock:
            self.dir = self.root / f"{time.strftime('%Y-%m-%d_%H%M%S')}_{label}"
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
            for name, data in self.pending:
                (self.dir / name).write_bytes(data)
            self.pending = []
        return self.dir

    def event(self, event: dict) -> None:
        if self.dir is None:
            return
        with self.lock, open(self.dir / "events.jsonl", "a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def save(self, name: str, data: bytes | str | dict) -> None:
        if self.dir is not None:
            (self.dir / name).write_bytes(self._bytes(data))

    @staticmethod
    def _bytes(data: bytes | str | dict) -> bytes:
        if isinstance(data, bytes):
            return data
        if isinstance(data, dict):
            data = json.dumps(data, indent=2, default=str)
        return data.encode()
