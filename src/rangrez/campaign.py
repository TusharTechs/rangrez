"""The colourist's toolbox: deterministic tools the agent calls.

A shade-matching campaign is what a dye-house colourist does by hand: an
initial recipe from the dye model, a lab dip, a measurement, a correction,
again. Here every step is a tool with a checkable result. The agent decides
which tool to call and why; it never computes a colour or a volume itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import glimmer
from .colour import delta_e_2000, lab_to_hex
from .km import DyeModel, fit_gains, solve_recipe
from .memory import LabMemory
from .profiles import PROFILE
from .rig import DYES, UNIT, GateError, Recipe, Rig, check_recipe, execute
from .vision import Measurement, Segmenter, measure, overlay

SIDE_BY_SIDE_MAX_DE00 = 12.0  # a cloth further than this from the target is not the customer's


@dataclass
class Shot:
    index: int
    recipe: Recipe
    measurement: Measurement
    delta_e: float
    rationale: str
    image_jpeg: bytes
    overlay_jpeg: bytes
    at: float = field(default_factory=time.time)


@dataclass
class Campaign:
    target_lab: np.ndarray
    rig: Rig
    segmenter: Segmenter
    tolerance: float = 1.5  # CIEDE2000 pass threshold
    max_shots: int = 10
    model: DyeModel = field(default_factory=PROFILE.nominal)
    shots: list[Shot] = field(default_factory=list)
    notebook: list[dict] = field(default_factory=list)
    listeners: list = field(default_factory=list)
    status: str = "running"
    side_readings: list = field(default_factory=list)  # the customer's cloth, read beside each strip
    memory: LabMemory | None = None  # every dip this lab has made with this dye lot
    campaign_id: str = field(default_factory=lambda: f"c{int(time.time() * 1000)}")
    memory_source_de: float | None = None  # how far the remembered dip offered first was from the target

    def __post_init__(self) -> None:
        if self.memory is not None and self.memory.dips:
            self.model = self.memory.warm_model(self.model)
            self.note(f"Lab memory: {len(self.memory.dips)} dips from {len(self.memory.campaigns)} earlier campaigns "
                      f"with this dye lot. Starting from the uptake they showed: "
                      f"{dict(zip(DYES, np.round(self.model.gains, 2).tolist()))}.", who="lab")

    # -- events ------------------------------------------------------------
    def emit(self, kind: str, **data) -> None:
        event = {"kind": kind, "at": time.time(), **data}
        for fn in self.listeners:
            fn(event)

    def note(self, text: str, who: str = "agent") -> None:
        entry = {"who": who, "text": text, "at": time.time()}
        self.notebook.append(entry)
        self.emit("note", **entry)

    # -- state ---------------------------------------------------------------
    @property
    def best(self) -> Shot | None:
        ok = [s for s in self.shots if s.measurement.ok]
        return min(ok, key=lambda s: s.delta_e) if ok else None

    def state(self) -> dict:
        best = self.best
        return {
            "target_lab": [round(v, 2) for v in self.target_lab],
            "target_hex": lab_to_hex(self.target_lab),
            "tolerance_de00": self.tolerance,
            "shots_used": len(self.shots),
            "shots_left": self.max_shots - len(self.shots),
            "best": None if best is None else self._shot_summary(best),
            "history": [self._shot_summary(s) for s in self.shots],
            "model_gains": dict(zip(DYES, np.round(self.model.gains, 3).tolist())),
            "stock_ml": dict(zip(DYES, np.round(self.rig.inventory.stock_ml, 1).tolist())),
            "dyer_notes": [n["text"] for n in self.notebook if n["who"] == "dyer"][-5:],
            "status": self.status,
        }

    @staticmethod
    def _shot_summary(s: Shot) -> dict:
        m = s.measurement
        return {
            "shot": s.index,
            "recipe": dict(zip(DYES, np.round(s.recipe.conc, 3).tolist())),
            "measured_lab": [round(v, 2) for v in m.lab],
            "lab_sd": [round(v, 2) for v in m.lab_sd],
            "delta_e00": round(s.delta_e, 2),
            "flags": m.flags,
        }

    # -- tools -----------------------------------------------------------------
    def fit_model(self) -> dict:
        """Refit per-dye uptake to every valid shot so far (and every remembered dip of this lot)."""
        data = [(s.recipe.as_array(), s.measurement.lab) for s in self.shots if s.measurement.ok]
        if self.memory is not None:  # it already holds this campaign's dips
            data = self.memory.recent()
        before = self.model.gains.copy()
        self.model = fit_gains(self.model, data)
        errs = [float(delta_e_2000(self.model.predict_lab(c), lab)) for c, lab in data]
        out = {
            "gains_before": dict(zip(DYES, np.round(before, 3).tolist())),
            "gains_after": dict(zip(DYES, np.round(self.model.gains, 3).tolist())),
            "model_fit_error_de00": round(float(np.mean(errs)), 2) if errs else None,
            "shots_used": len(data),
        }
        self.emit("model", **out)
        return out

    def suggest_recipes(self) -> dict:
        """Candidate recipes with predicted CIEDE2000, from two independent methods."""
        cands = []
        conc, de = solve_recipe(self.model, self.target_lab, max_conc=self.rig.limits.max_conc_g_per_l)
        cands.append(("model_inverse", conc, de, "Invert the refitted Kubelka-Munk model."))
        if not self.shots and glimmer.configured():
            g = glimmer.first_guess(self.target_lab, log=getattr(self.segmenter, "log", None))
            if g is not None:
                d = float(delta_e_2000(self.model.predict_lab(g), self.target_lab))
                cands.append(("glimmer", g, d, ("Muse Glimmer fine-tuned on this lab's own dips: its first recipes "
                              "land closer than the supplier model's (the predicted dE00 shown is the supplier model's opinion).")))
        if not self.shots and self.memory is not None and (near := self.memory.nearest(self.target_lab)):
            c0, lab0, d0 = near
            step = self._correct(c0, lab0)
            pred = float(delta_e_2000(lab0 + self._jac(c0) @ (step - c0), self.target_lab))
            self.memory_source_de = d0
            cands.append(("lab_memory", step, pred, (f"The most similar dip this lab has made (dE00 {d0:.1f} from the target), "
                          "corrected towards it with the refitted model's sensitivity.")))
        best = self.best
        if best is not None:
            step = self._correction_step(best)
            if step is not None:
                pred = float(delta_e_2000(best.measurement.lab + self._jac(best.recipe.as_array()) @ (step - best.recipe.as_array()), self.target_lab))
                cands.append(("colourist_correction", step, pred, "Correct the best real dip by its measured error, using the refitted model's sensitivity."))
        out = []
        for name, c, d, why in cands:
            c = np.clip(c, 0, self.rig.limits.max_conc_g_per_l)
            recipe = Recipe(tuple(float(round(v, 3)) for v in c))
            try:
                check_recipe(recipe, self.rig.limits, self.rig.inventory)
                gate = "pass"
            except GateError as exc:
                gate = f"refused: {exc}"
            out.append({
                "method": name,
                "recipe": dict(zip(DYES, recipe.conc)),
                "predicted_delta_e00": round(float(d), 2),
                "gate": gate,
                "why": why,
            })
        return {"candidates": out}

    def run_shot(self, recipe: dict, dip_seconds: float = 90.0, rationale: str = "") -> dict:
        """Gate, dispense, dip, dry, photograph, measure."""
        conc = tuple(float(recipe.get(d, 0.0)) for d in DYES)
        recipe = Recipe(conc, dip_seconds)
        # A rig that can only measure coarsely (a person with a syringe) reports what it will really dispense.
        quantize = getattr(self.rig, "quantize", None)
        if quantize:
            recipe = quantize(recipe)
        if len(self.shots) >= self.max_shots:
            return {"refused": "shot budget exhausted; escalate or stop"}
        try:
            check_recipe(recipe, self.rig.limits, self.rig.inventory)
        except GateError as exc:
            self.emit("gate", recipe=recipe.describe(), refused=str(exc))
            return {"refused": str(exc)}
        self.emit("dispense", recipe=recipe.describe(), index=len(self.shots) + 1, rationale=rationale)
        result = execute(self.rig, recipe)
        m = measure(result.image_jpeg, self.segmenter, with_target=True, seed=len(self.shots))
        if m.ok and m.target_lab is not None:
            self._side_by_side(m)
        de = float(delta_e_2000(m.lab, self.target_lab)) if m.ok else float("nan")
        if m.ok and self.memory is not None:
            self.memory.record_dip(recipe.as_array(), m.lab, self.campaign_id)
        shot = Shot(len(self.shots) + 1, recipe, m, de, rationale, result.image_jpeg, overlay(result.image_jpeg, m))
        self.shots.append(shot)
        self.emit("shot", index=shot.index, recipe=recipe.describe(), lab=m.lab.tolist(),
                  hex=lab_to_hex(m.lab), delta_e=de, flags=m.flags, rationale=rationale)
        matched = m.ok and de <= self.tolerance and self._confirmed(shot)
        if matched:
            self.status = "matched"
            self.emit("status", status=self.status, shot=shot.index, delta_e=shot.delta_e)
        return self._shot_summary(shot) | {"matched": matched}

    def _confirmed(self, shot: Shot) -> bool:
        """No single photo decides a pass: an automated rig photographs the strip again and both must agree."""
        if hasattr(self.rig, "provide") or not hasattr(self.rig, "capture"):
            return True  # a person sent this photo; asking for another would only slow them down
        try:
            m2 = measure(self.rig.capture(), self.segmenter, seed=100 + shot.index)
        except RuntimeError as exc:
            self.note(f"Confirmation photo failed ({exc}); not calling it a match on one photo.", who="lab")
            return False
        if not m2.ok:
            self.note("Confirmation photo was rejected; not calling it a match on one photo.", who="lab")
            return False
        de2 = float(delta_e_2000(m2.lab, self.target_lab))
        if de2 > self.tolerance:
            self.note(f"Dip {shot.index} read dE00 {shot.delta_e:.2f}, but a second photo reads {de2:.2f}: "
                      "not a match yet.", who="lab")
            return False
        shot.delta_e = (shot.delta_e + de2) / 2
        self.note(f"Confirmed with a second photo: dE00 {shot.delta_e * 2 - de2:.2f} and {de2:.2f}.", who="lab")
        return True

    def _side_by_side(self, m) -> None:
        """The customer's cloth lies beside the strip: compare them under the same light.

        A target read from a photo taken elsewhere carries that photo's light and camera
        error; the cloth in the dip photo shares the strip's. Re-anchor to the running mean
        of those side-by-side readings, unless the "cloth" is clearly something else.
        """
        seen = np.asarray(m.target_lab, dtype=float)
        if float(delta_e_2000(seen, self.target_lab if not self.side_readings else np.mean(self.side_readings, axis=0))) > SIDE_BY_SIDE_MAX_DE00:
            m.flags.append("WARN: another cloth is in the photo but it is not the target colour; ignored it")
            return
        first = not self.side_readings
        self.side_readings.append(seen)
        before = self.target_lab
        self.target_lab = np.mean(self.side_readings, axis=0)
        m.flags.append("NOTE: compared side by side with the customer's cloth in the same photo")
        if first:
            self.note(f"The customer's cloth is in the photo, so I compare the strip with it under the same light. "
                      f"Target re-read as {lab_to_hex(self.target_lab)} (was {lab_to_hex(before)}, "
                      f"dE00 {float(delta_e_2000(before, self.target_lab)):.1f} apart).", who="agent")
            for s in self.shots:  # earlier dips are judged against the better target too
                s.delta_e = float(delta_e_2000(s.measurement.lab, self.target_lab)) if s.measurement.ok else float("nan")

    def close(self) -> None:
        """Remember how this campaign went (the night shift's chart reads these)."""
        if self.memory is None:
            return
        ok = [s for s in self.shots if s.measurement.ok]
        self.memory.record_campaign({
            "campaign": self.campaign_id, "target_lab": [round(float(v), 2) for v in self.target_lab],
            "status": self.status, "dips": len(self.shots),
            "first_de00": round(ok[0].delta_e, 2) if ok else None,
            "best_de00": round(self.best.delta_e, 2) if self.best else None,
        })

    def declare_out_of_gamut(self) -> dict:
        best = self.best
        self.status = "out_of_gamut"
        text = (
            f"Target is outside what {', '.join(DYES)} can reach on cotton within the "
            f"{self.rig.limits.max_conc_g_per_l:.0f} {UNIT} safety cap. Closest achievable: dip {best.index}, "
            f"{lab_to_hex(best.measurement.lab)}, dE00 {best.delta_e:.2f}. A darker or different dye is needed."
        )
        self.note(text, who="agent")
        self.emit("status", status=self.status, reason=text, shot=best.index, delta_e=best.delta_e)
        return {"out_of_gamut": True, "closest_shot": best.index, "delta_e00": round(best.delta_e, 2)}

    def escalate(self, reason: str) -> dict:
        self.status = "escalated"
        self.note(f"Escalated to the colourist: {reason}", who="agent")
        self.emit("status", status=self.status, reason=reason)
        return {"escalated": True}

    # -- numerics ----------------------------------------------------------------
    def _jac(self, conc: np.ndarray) -> np.ndarray:
        # Sensitivity from the model refitted to every real dip. Learning it from the dips
        # themselves (Broyden, or ridge regression on nearby dips) chases measurement noise
        # when steps are whole drops: tried, and it matched fewer shades.
        return self._model_jacobian(conc)

    def _model_jacobian(self, conc: np.ndarray) -> np.ndarray:
        h = 0.05
        base = self.model.predict_lab(conc)
        cols = []
        for i in range(len(conc)):
            c = conc.copy()
            c[i] += h
            cols.append((self.model.predict_lab(c) - base) / h)
        return np.stack(cols, axis=1)

    def _correction_step(self, best: Shot) -> np.ndarray | None:
        return self._correct(best.recipe.as_array(), best.measurement.lab)

    def _correct(self, c0: np.ndarray, lab: np.ndarray) -> np.ndarray:
        """Move recipe c0, which dyed `lab`, towards the target."""
        J = self._jac(c0)
        err = self.target_lab - lab
        # Damped least squares; weight lightness less, as CIEDE2000 does.
        Wt = np.diag([0.6, 1.0, 1.0])
        A = Wt @ J
        lam = 0.05
        dc = np.linalg.solve(A.T @ A + lam * np.eye(len(c0)), A.T @ (Wt @ err))
        return np.clip(c0 + dc, 0, self.rig.limits.max_conc_g_per_l)
