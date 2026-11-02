"""The real rig: ESP32 over USB serial, plus a USB / Continuity camera.

Protocol: one JSON object per line each way. The firmware refuses anything
outside its own hard limits (see firmware/rangrez_rig/rangrez_rig.ino), so a
bug here still cannot overfill the cup or cook a strip.

Pumps: 0 turquoise, 1 fuchsia, 2 lemon yellow, 3 water, 4 drain.
Cotton tape is pre-soaked in soda ash and dried; a stepper advances a fresh
segment over the cup for each dip, so the finished tape is a ribbon of every
attempt in order.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import serial

from .camera import grab
from .camera import source as camera_source
from .rig import DYES, Inventory, PumpPlan, Recipe, RigLimits, ShotResult, now

WATER, DRAIN = 3, 4


class RigError(RuntimeError):
    pass


@dataclass
class SerialRig:
    port: str
    limits: RigLimits = field(default_factory=RigLimits)
    inventory: Inventory = field(default_factory=Inventory)
    camera: int | str = field(default_factory=camera_source)
    stir_seconds: float = 8.0
    dry_seconds: float = 90.0
    settle_seconds: float = 2.0
    segment_mm: float = 70.0
    boot_seconds: float = 2.0  # the ESP32 resets when the port opens

    def __post_init__(self) -> None:
        self.link = serial.Serial(self.port, 115200, timeout=5)
        time.sleep(self.boot_seconds)
        hello = self.cmd({"cmd": "ping"})
        if not str(hello.get("fw", "")).startswith("rangrez"):
            raise RigError(f"unexpected firmware reply: {hello}")

    def cmd(self, payload: dict, timeout: float = 30.0) -> dict:
        self.link.write((json.dumps(payload) + "\n").encode())
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self.link.readline().decode(errors="replace").strip()
            if not line or not line.startswith("{"):
                continue
            reply = json.loads(line)
            if reply.get("error"):
                raise RigError(f"firmware refused {payload['cmd']}: {reply['error']}")
            if reply.get("done") == payload["cmd"] or payload["cmd"] == "ping":
                return reply
        raise RigError(f"timed out waiting for {payload['cmd']}")

    def _pump(self, pump_id: int, ml: float) -> None:
        if ml > 0.05:
            self.cmd({"cmd": "pump", "id": pump_id, "ml": round(ml, 2)}, timeout=60)

    def run_shot(self, recipe: Recipe) -> ShotResult:
        started = now()
        dye_ml = recipe.as_array() / self.limits.stock_g_per_l * self.limits.bath_ml
        water_ml = self.limits.bath_ml - dye_ml.sum()
        # Rinse the cup: drain, water, stir, drain.
        self.cmd({"cmd": "drain", "seconds": 8}, timeout=20)
        self._pump(WATER, 10)
        self.cmd({"cmd": "stir", "seconds": 3}, timeout=10)
        self.cmd({"cmd": "drain", "seconds": 8}, timeout=20)
        # Dispense, mix, dip a fresh tape segment, dry it under the camera.
        for i, ml in enumerate(dye_ml):
            self._pump(i, float(ml))
        self._pump(WATER, float(water_ml))
        self.cmd({"cmd": "stir", "seconds": self.stir_seconds}, timeout=30)
        self.cmd({"cmd": "advance", "mm": self.segment_mm}, timeout=30)
        self.cmd({"cmd": "dip", "seconds": recipe.dip_seconds}, timeout=recipe.dip_seconds + 30)
        self.cmd({"cmd": "advance", "mm": self.segment_mm}, timeout=30)  # dyed segment into the light box
        self.cmd({"cmd": "dry", "seconds": self.dry_seconds}, timeout=self.dry_seconds + 30)
        self.cmd({"cmd": "light", "on": True})
        time.sleep(self.settle_seconds)
        frame = self.capture()
        plan = PumpPlan(tuple(float(v) for v in dye_ml), float(water_ml))
        return ShotResult(frame, plan, started, now(), {"dyes": DYES})

    def capture(self) -> bytes:
        try:
            return grab(self.camera)
        except RuntimeError as exc:
            raise RigError(str(exc)) from exc
