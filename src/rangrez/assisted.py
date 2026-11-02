"""Assisted rig: a person is the pumps, the phone is the camera.

The agent and the safety gate work exactly as with the robot rig. Instead of
driving pumps, a dip becomes a short set of instructions on the dashboard
(how many mL of each colour, how long to dip), and the lab waits for a photo
of the dried strip lying next to the colour card.
"""

from __future__ import annotations

import io
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps

from .profiles import PROFILE
from .rig import DYES, UNIT, Inventory, PumpPlan, Recipe, RigLimits, ShotResult, now

STEP_ML = PROFILE.step_ml
MAX_SIDE_PX = 1920  # SAM 3.1 accepts up to about 4 MP; phones shoot 12+


class Cancelled(RuntimeError):
    pass


def normalise_photo(data: bytes) -> bytes:
    """Any phone photo -> upright JPEG no larger than MAX_SIDE_PX."""
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as exc:
        raise ValueError("could not read that image; please send a JPEG or PNG") from exc
    img.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=93)
    return buf.getvalue()


@dataclass
class AssistedRig:
    limits: RigLimits = field(default_factory=RigLimits)
    inventory: Inventory = field(default_factory=Inventory)
    on_instructions: Callable[[dict], None] = lambda _: None
    timeout_s: float = 45 * 60
    camera_index: int = 0

    def __post_init__(self) -> None:
        self._photo: bytes | None = None
        self._ready = threading.Event()
        self._cancelled = False
        self.waiting: dict | None = None

    # -- what a person can actually measure ------------------------------
    def quantize(self, recipe: Recipe) -> Recipe:
        ml = recipe.as_array() / self.limits.stock_g_per_l * self.limits.bath_ml
        ml = np.round(ml / STEP_ML) * STEP_ML
        conc = ml / self.limits.bath_ml * self.limits.stock_g_per_l
        return Recipe(tuple(float(round(v, 4)) for v in conc), recipe.dip_seconds)

    def instructions(self, recipe: Recipe, index: int) -> dict:
        ml = recipe.as_array() / self.limits.stock_g_per_l * self.limits.bath_ml
        water = self.limits.bath_ml - ml.sum()
        return {
            "index": index,
            "dyes_ml": {d: round(float(v), 2) for d, v in zip(DYES, ml)},
            "amounts": {d: (round(c) if UNIT == "drops" else round(float(v), 1)) for d, v, c in zip(DYES, ml, recipe.conc)},
            "unit": "drops" if UNIT == "drops" else "mL",
            "water_ml": round(float(water), 1),
            "dip_seconds": int(recipe.dip_seconds),
            "steps": [
                "Rinse the glass and the dropper or syringes.",
                *[self._amount(d, v, c) for d, v, c in zip(DYES, ml, recipe.conc) if v >= STEP_ML / 2],
                (f"Add 4 teaspoons (20 mL) of water and stir." if UNIT == "drops" else f"Add {water:.1f} mL of water and stir for 10 seconds."),
                f"Dip a fresh white cotton strip for {int(recipe.dip_seconds)} seconds.",
                "Squeeze it gently and dry it fully with the hair dryer (about 2 minutes).",
                "Lay it on a sheet of white paper (or next to the colour card) in the same spot and light as before, and send the photo.",
            ],
        }

    @staticmethod
    def _amount(dye: str, ml: float, conc: float) -> str:
        if UNIT == "drops":
            return f"Add {round(conc)} drops of {dye}."
        return f"Add {ml:.1f} mL of {dye}."

    # -- rig protocol ----------------------------------------------------------
    def run_shot(self, recipe: Recipe) -> ShotResult:
        started = now()
        index = getattr(self, "_count", 0) + 1
        self._count = index
        self.waiting = self.instructions(recipe, index)
        self._photo, self._cancelled = None, False
        self._ready.clear()
        self.on_instructions(self.waiting)
        if not self._ready.wait(self.timeout_s):
            self.waiting = None
            raise Cancelled("no photo arrived in time")
        self.waiting = None
        if self._cancelled or self._photo is None:
            raise Cancelled("campaign stopped")
        ml = recipe.as_array() / self.limits.stock_g_per_l * self.limits.bath_ml
        plan = PumpPlan(tuple(float(v) for v in ml), float(self.limits.bath_ml - ml.sum()))
        return ShotResult(self._photo, plan, started, now(), {"assisted": True})

    def provide(self, photo: bytes) -> None:
        """Called when the person uploads a photo or presses 'use webcam'."""
        if self.waiting is None:
            raise RuntimeError("the lab is not waiting for a photo right now")
        self._photo = normalise_photo(photo)
        self._ready.set()

    def cancel(self) -> None:
        self._cancelled = True
        self._ready.set()

    def capture(self) -> bytes:
        import cv2

        cam = cv2.VideoCapture(self.camera_index)
        try:
            for _ in range(10):
                cam.read()
            ok, frame = cam.read()
            if not ok:
                raise RuntimeError("webcam returned no frame")
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
            return jpeg.tobytes()
        finally:
            cam.release()
