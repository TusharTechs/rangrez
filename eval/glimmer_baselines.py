"""Is Glimmer better than a fair data-driven baseline on the same data?

Same 2,900 dips, same 100 held-out targets, same scoring (the true colour each recipe dyes):
  * k nearest neighbours: average the recipes of the 5 dips whose measured colour is closest
  * learned forward model: an MLP (and a Gaussian process) from recipe to measured colour,
    inverted by optimisation, the way a colourist's software inverts its model
  * the Kubelka-Munk model with its 3 uptake gains refitted to all 2,900 dips

    uv run --with scikit-learn python eval/glimmer_baselines.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from rangrez.colour import delta_e_2000
from rangrez.glimmer import prompt
from rangrez.km import fit_gains, solve_recipe
from rangrez.profiles import PROFILE
from rangrez.rig import Recipe
from rangrez.sim import VirtualRig

ROOT = Path(__file__).parent / "glimmer"


def main() -> None:
    recs = json.loads((ROOT / "records.json").read_text())
    by_prompt = {prompt(d["measured_lab"]): d for d in recs["train"]}
    holdout = [json.loads(line) for line in (ROOT / "run" / "holdout.jsonl").read_text().splitlines()]
    targets = [by_prompt[next(m["content"] for m in h["messages"] if m["role"] == "user")] for h in holdout]
    held = {id(d) for d in targets}
    train = [d for d in recs["train"] if id(d) not in held]
    X_conc = np.array([d["conc"] for d in train])
    Y_lab = np.array([d["measured_lab"] for d in train])
    world = VirtualRig()
    cap = PROFILE.max_conc

    def score(recipes) -> dict:
        e = np.array([float(delta_e_2000(world.true_swatch_lab(Recipe(tuple(np.clip(c, 0, cap)))), np.array(d["true_lab"])))
                      for c, d in zip(recipes, targets)])
        return {"median": round(float(np.median(e)), 2), "p90": round(float(np.percentile(e, 90)), 2),
                "under_3": round(float((e < 3).mean()), 2)}

    labs = [np.array(d["measured_lab"]) for d in targets]
    out = {}

    knn = KNeighborsRegressor(n_neighbors=5, weights="distance").fit(Y_lab, X_conc)
    out["k nearest neighbours (k=5)"] = score(knn.predict(np.array(labs)))

    def inverse(predict) -> list:
        res = []
        for lab in labs:
            start = knn.predict(lab[None])[0]  # a good starting point from the data
            r = least_squares(lambda c, lab=lab: predict(np.clip(c, 0, cap)[None])[0] - lab, start, bounds=(0, cap))
            res.append(r.x)
        return res

    mlp = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=3000, random_state=0).fit(np.sqrt(X_conc), Y_lab)
    out["learned forward model (MLP), inverted"] = score(inverse(lambda c: mlp.predict(np.sqrt(c))))

    kernel = ConstantKernel(100.0) * RBF(length_scale=[1.0, 1.0, 1.0]) + WhiteKernel(0.3)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0).fit(np.sqrt(X_conc[:1500]), Y_lab[:1500])
    out["Gaussian process forward model, inverted"] = score(inverse(lambda c: gp.predict(np.sqrt(c))))

    km = fit_gains(PROFILE.nominal(), list(zip(X_conc, Y_lab)))
    out["Kubelka-Munk, 3 gains refitted to all dips"] = score([solve_recipe(km, lab, cap)[0] for lab in labs])

    glim = json.loads((ROOT / "modal_run" / "eval.json").read_text())
    out["Muse Glimmer 30B, LoRA (Modal)"] = {k: glim["Muse Glimmer 30B, tuned"][k] for k in ("median", "p90", "under_3")}
    for k, v in out.items():
        print(f"{k:45s} median {v['median']:5.2f}  p90 {v['p90']:5.2f}  under 3: {v['under_3']:.0%}")
    (ROOT / "baselines.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
