"""Replay a recorded live run on the dashboard, for visitors without the live passcode.

Every event in a recording happened for real: each SAM 3.1, Muse Wonder, Muse Spark, Muse Image
or Muse Voice Transcribe call with its timing, each dip photo and SAM's masks, the notebook, the
report. A replay sends those events again in order (a little faster), and the dashboard labels
it as a recording with its date. Recordings shipped with the app live in samples/runs/.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

SAMPLES = Path(__file__).parent / "samples"


def runs() -> list[dict]:
    out = []
    for d in sorted((SAMPLES / "runs").glob("*")):
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
        out.append({"name": d.name, "title": meta.get("title", d.name), "started": meta.get("started", ""),
                    "agent": meta.get("agent", ""), "rig": meta.get("rig", "")})
    return out


class Replay:
    def __init__(self, name: str, publish, speed: float = 3.0, max_gap: float = 2.5) -> None:
        self.dir = SAMPLES / "runs" / Path(name).name  # no path tricks: a run name only
        if not (self.dir / "events.jsonl").exists():
            raise FileNotFoundError(name)
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8")) if (self.dir / "meta.json").exists() else {}
        self.events = [json.loads(line) for line in (self.dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.publish, self.speed, self.max_gap = publish, speed, max_gap
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stopped = True

    @property
    def running(self) -> bool:
        return self.thread.is_alive()

    def file(self, name: str) -> bytes | None:
        p = self.dir / Path(name).name
        return p.read_bytes() if p.exists() else None

    def _run(self) -> None:
        recorded = self.meta.get("started", "")
        last = None
        for e in self.events:
            if self.stopped:
                return
            at = e.get("at")
            if last is not None and at is not None:
                time.sleep(min(max(at - last, 0) / self.speed, self.max_gap))
            last = at if at is not None else last
            e = {**e, "at": time.time(), "replay": True}
            if e.get("kind") == "start":
                e["recorded"] = recorded
            self.publish(e)
