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

from .colour import delta_e_2000, lab_to_hex
from .km import DyeModel, fit_gains, nominal_procion_cmy, solve_recipe
from .rig import DYES, GateError, Recipe, Rig, check_recipe, execute
from .vision import Measurement, Segmenter, measure, overlay


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
    model: DyeModel = field(default_factory=nominal_procion_cmy)
    shots: list[Shot] = field(default_factory=list)
    notebook: list[dict] = field(default_factory=list)
    listeners: list = field(default_factory=list)
    jacobian: np.ndarray | None = None  # d(Lab)/d(conc) learned from real shots
    status: str = "running"

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
            "status": self.status,
        }

    @staticmethod
    def _shot_summary(s: Shot) -> dict:
        m = s.measurement
        return {
            "shot": s.index,
            "recipe_g_per_l": dict(zip(DYES, np.round(s.recipe.conc, 3).tolist())),
            "measured_lab": [round(v, 2) for v in m.lab],
            "lab_sd": [round(v, 2) for v in m.lab_sd],
            "delta_e00": round(s.delta_e, 2),
            "flags": m.flags,
        }

    # -- tools -----------------------------------------------------------------
    def fit_model(self) -> dict:
        """Refit per-dye uptake to every valid shot so far."""
        data = [(s.recipe.as_array(), s.measurement.lab) for s in self.shots if s.measurement.ok]
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
        best = self.best
        if best is not None:
            step = self._correction_step(best)
            if step is not None:
                pred = float(delta_e_2000(best.measurement.lab + self._jac(best.recipe.as_array()) @ (step - best.recipe.as_array()), self.target_lab))
                cands.append(("colourist_correction", step, pred, "Correct the best real dip using the Jacobian learned from real shots."))
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
                "recipe_g_per_l": dict(zip(DYES, recipe.conc)),
                "predicted_delta_e00": round(float(d), 2),
                "gate": gate,
                "why": why,
            })
        return {"candidates": out}

    def run_shot(self, recipe_g_per_l: dict, dip_seconds: float = 90.0, rationale: str = "") -> dict:
        """Gate, dispense, dip, dry, photograph, measure."""
        conc = tuple(float(recipe_g_per_l.get(d, 0.0)) for d in DYES)
        recipe = Recipe(conc, dip_seconds)
        if len(self.shots) >= self.max_shots:
            return {"refused": "shot budget exhausted; escalate or stop"}
        try:
            check_recipe(recipe, self.rig.limits, self.rig.inventory)
        except GateError as exc:
            self.emit("gate", recipe=recipe.describe(), refused=str(exc))
            return {"refused": str(exc)}
        self.emit("dispense", recipe=recipe.describe(), index=len(self.shots) + 1, rationale=rationale)
        result = execute(self.rig, recipe)
        m = measure(result.image_jpeg, self.segmenter, seed=len(self.shots))
        de = float(delta_e_2000(m.lab, self.target_lab)) if m.ok else float("nan")
        shot = Shot(len(self.shots) + 1, recipe, m, de, rationale, result.image_jpeg, overlay(result.image_jpeg, m))
        prev_best = self.best
        self.shots.append(shot)
        if m.ok and prev_best is not None:
            self._update_jacobian(prev_best, shot)
        self.emit("shot", index=shot.index, recipe=recipe.describe(), lab=m.lab.tolist(),
                  hex=lab_to_hex(m.lab), delta_e=de, flags=m.flags, rationale=rationale)
        if m.ok and de <= self.tolerance:
            self.status = "matched"
            self.emit("status", status=self.status, shot=shot.index, delta_e=de)
        return self._shot_summary(shot) | {"matched": m.ok and de <= self.tolerance}

    def escalate(self, reason: str) -> dict:
        self.status = "escalated"
        self.note(f"Escalated to the colourist: {reason}", who="agent")
        self.emit("status", status=self.status, reason=reason)
        return {"escalated": True}

    # -- numerics ----------------------------------------------------------------
    def _jac(self, conc: np.ndarray) -> np.ndarray:
        if self.jacobian is not None:
            return self.jacobian
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

    def _update_jacobian(self, a: Shot, b: Shot) -> None:
        """Broyden rank-1 update from two real dips: reality, not the model."""
        dc = b.recipe.as_array() - a.recipe.as_array()
        dlab = b.measurement.lab - a.measurement.lab
        if np.linalg.norm(dc) < 1e-6:
            return
        J = self._jac(a.recipe.as_array())
        self.jacobian = J + np.outer(dlab - J @ dc, dc) / (dc @ dc)

    def _correction_step(self, best: Shot) -> np.ndarray | None:
        c0 = best.recipe.as_array()
        J = self._jac(c0)
        err = self.target_lab - best.measurement.lab
        # Damped least squares; weight lightness less, as CIEDE2000 does.
        Wt = np.diag([0.6, 1.0, 1.0])
        A = Wt @ J
        lam = 0.05
        dc = np.linalg.solve(A.T @ A + lam * np.eye(len(c0)), A.T @ (Wt @ err))
        return np.clip(c0 + dc, 0, self.rig.limits.max_conc_g_per_l)
