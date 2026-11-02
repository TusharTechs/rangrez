"""A virtual Rangrez rig: the ESP32 firmware's protocol and hard limits on a pseudo serial port,
with the simulator's hidden dye chemistry and a light-box camera that records video.

The limits are read from firmware/rangrez_rig/rangrez_rig.ino itself, so the emulator
refuses exactly what the board refuses. The real SerialRig driver talks to it over a
real (pseudo) serial port and grabs frames from its camera video with OpenCV, so the
whole robot path runs end to end without a single part bought.

    RANGREZ_RIG=serial RANGREZ_PORT=virtual uv run rangrez-lab
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import termios
import threading
import time
import tty
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .km import reflectance_to_lab
from .sim import VirtualRig

FIRMWARE = Path(__file__).parents[2] / "firmware" / "rangrez_rig" / "rangrez_rig.ino"


def firmware_limits(path: Path = FIRMWARE) -> dict[str, float]:
    """The `const float MAX_*` limits and pump calibration, as compiled into the board."""
    src = path.read_text()
    limits = {k: float(v) for k, v in re.findall(r"const float (MAX_\w+) = ([\d.]+);", src)}
    limits["msPerMl"] = [float(v) for v in re.search(r"msPerMl\[5\] = \{([^}]*)\}", src).group(1).split(",")]
    return limits


@dataclass
class VirtualESP32:
    """Speaks the firmware's line protocol. time_scale=0 skips the real waiting."""

    camera_path: str = field(default_factory=lambda: tempfile.mkstemp(suffix=".mp4", prefix="rangrez-cam-")[1])
    scene: str = "light_box"
    card: bool = True
    seed: int = 21
    time_scale: float = 0.0
    pump_rate_error: tuple[float, ...] = (0.97, 1.04, 1.01, 1.0)  # real pumps vs their calibration
    pump_cv: float = 0.02

    def __post_init__(self) -> None:
        self.limits = firmware_limits()
        self.rig = VirtualRig(seed=self.seed, scene=self.scene, card=self.card)
        self.rng = np.random.default_rng(self.seed)
        self.bath_ml = 0.0
        self.dye_ml = np.zeros(3)
        self.estop = False
        self.light = False
        self.log: list[dict] = []
        self._write_video()  # the camera shows the empty light box until the first dip

    # -- e-stop button ----------------------------------------------------------
    def press_estop(self) -> None:
        self.estop = True

    def _wait(self, seconds: float) -> bool:
        end = time.monotonic() + seconds * self.time_scale
        while time.monotonic() < end:
            if self.estop:
                return False
            time.sleep(0.005)
        return not self.estop

    # -- protocol ---------------------------------------------------------------
    def handle(self, line: str) -> dict:
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            return self._reply("?", "bad json")
        cmd = str(doc.get("cmd", ""))
        self.log.append(doc)
        if self.estop and cmd not in ("reset", "ping"):
            return self._reply(cmd, "e-stop latched; send reset")
        L = self.limits
        if cmd == "ping":
            return {"fw": "rangrez-1", "estop": self.estop, "bath_ml": self.bath_ml}
        if cmd == "reset":
            self.estop = False
            return self._reply(cmd)
        if cmd == "pump":
            pid, ml = int(doc.get("id", -1)), float(doc.get("ml", 0))
            if not 0 <= pid <= 3:
                return self._reply(cmd, "pump id must be 0-3 (use drain for 4)")
            if ml <= 0 or ml > L["MAX_PUMP_ML"]:
                return self._reply(cmd, "ml out of range")
            if self.bath_ml + ml > L["MAX_BATH_ML"]:
                return self._reply(cmd, "cup would overflow")
            ok = self._wait(ml * L["msPerMl"][pid] / 1000)
            delivered = ml * self.pump_rate_error[pid] * self.rng.normal(1, self.pump_cv)
            if pid < 3:
                self.dye_ml[pid] += delivered
            self.bath_ml += ml  # worst case, for the overflow check
            return self._reply(cmd) if ok else self._reply(cmd, "e-stop during pump")
        if cmd == "drain":
            s = float(doc.get("seconds", 0))
            if s <= 0 or s > L["MAX_DRAIN_S"]:
                return self._reply(cmd, "drain seconds out of range")
            if not self._wait(s):
                return self._reply(cmd, "e-stop during drain")
            self.bath_ml, self.dye_ml = 0.0, np.zeros(3)
            return self._reply(cmd)
        if cmd == "stir":
            s = float(doc.get("seconds", 0))
            if s <= 0 or s > L["MAX_STIR_S"]:
                return self._reply(cmd, "stir seconds out of range")
            return self._reply(cmd) if self._wait(s) else self._reply(cmd, "e-stop during stir")
        if cmd == "dip":
            s = float(doc.get("seconds", 0))
            if s < 10 or s > L["MAX_DIP_S"]:
                return self._reply(cmd, "dip seconds out of range")
            if self.bath_ml < 10:
                return self._reply(cmd, "cup is nearly empty")
            if not self._wait(s):
                return self._reply(cmd, "e-stop during dip")
            self._dye_segment(s)
            return self._reply(cmd)
        if cmd == "advance":
            mm = float(doc.get("mm", 0))
            if mm <= 0 or mm > L["MAX_ADVANCE_MM"]:
                return self._reply(cmd, "advance mm out of range")
            return self._reply(cmd) if self._wait(mm / 30) else self._reply(cmd, "e-stop")
        if cmd == "dry":
            s = float(doc.get("seconds", 0))
            if s <= 0 or s > L["MAX_DRY_S"]:
                return self._reply(cmd, "dry seconds out of range")
            return self._reply(cmd) if self._wait(s) else self._reply(cmd, "e-stop during dry")
        if cmd == "light":
            self.light = bool(doc.get("on", False))
            if self.light:
                self._write_video()
            return self._reply(cmd)
        if cmd == "calibrate":
            pid, ms = int(doc.get("id", -1)), float(doc.get("ms_per_ml", 0))
            if not 0 <= pid <= 3 or not 100 <= ms <= 5000:
                return self._reply(cmd, "bad calibration")
            L["msPerMl"][pid] = ms
            return self._reply(cmd)
        return self._reply(cmd, "unknown command")

    def _reply(self, cmd: str, error: str | None = None) -> dict:
        out = {"error": error} if error else {"done": cmd}
        return {**out, "bath_ml": self.bath_ml}

    # -- the physical world -------------------------------------------------------
    def _dye_segment(self, dip_seconds: float) -> None:
        r = self.rig
        dye_ml = self.dye_ml * r.limits.bath_ml / max(self.bath_ml, 1e-6)  # as if in the nominal bath
        conc = dye_ml / r.limits.bath_ml * r.limits.stock_g_per_l
        saturated = conc / (1 + conc / r.saturation_g_per_l)
        time_factor = (1 - np.exp(-dip_seconds / 45)) / (1 - np.exp(-90 / 45))
        r.last_swatch_lab = reflectance_to_lab(r.world.reflectance(saturated * time_factor))

    def _write_video(self, frames: int = 14) -> None:
        """What the light-box webcam sees: a short clip with a little sensor noise per frame."""
        jpeg = self.rig.render()
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        tmp = self.camera_path + ".tmp.mp4"
        out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
        for _ in range(frames):
            noisy = np.clip(img.astype(np.int16) + self.rng.normal(0, 1.5, img.shape).astype(np.int16), 0, 255)
            out.write(noisy.astype(np.uint8))
        out.release()
        os.replace(tmp, self.camera_path)

    @property
    def true_lab(self) -> np.ndarray | None:
        return self.rig.last_swatch_lab

    # -- serial port -------------------------------------------------------------
    def serve(self) -> str:
        """Open a pseudo serial port, answer on it from a thread, return its device path."""
        master, slave = os.openpty()
        tty.setraw(slave)
        attrs = termios.tcgetattr(slave)
        attrs[3] &= ~termios.ECHO
        termios.tcsetattr(slave, termios.TCSANOW, attrs)
        self._slave_fd = slave  # keep it open so the port stays alive between connections
        threading.Thread(target=self._loop, args=(master,), daemon=True).start()
        return os.ttyname(slave)

    def _loop(self, fd: int) -> None:
        buf = b""
        while True:
            try:
                chunk = os.read(fd, 1024)
            except OSError:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    os.write(fd, (json.dumps(self.handle(line.decode(errors="replace"))) + "\n").encode())


def start(**kw) -> tuple[VirtualESP32, str]:
    """A running virtual rig: (board, serial port path). Point the camera at board.camera_path."""
    board = VirtualESP32(**kw)
    return board, board.serve()

