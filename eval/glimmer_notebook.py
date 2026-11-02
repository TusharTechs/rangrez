"""Glimmer v2 data: each target comes with the lab's most similar past dips, as the lab would show them.

A Gaussian process on the same dips beats a Glimmer that sees only the target (0.77 vs 0.92 dE00,
eval/glimmer/baselines.json): a 30B model learning a 3-to-3 mapping is not the point. Here Glimmer
reads the lab's notebook instead: the k nearest remembered dips (recipe and measured colour, k from
0 to 6 so it copes with a young memory), and reasons from them to the first recipe. Same 2,900
training dips and the same 100 held-out targets; neighbours always come from the training dips only.

    uv run python eval/glimmer_notebook.py          # -> eval/glimmer/notebook/{train,holdout}.jsonl
    uv run --with modal modal run eval/glimmer_modal.py --data notebook --out notebook_run --adapter adapter_notebook
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rangrez.colour import delta_e_2000
from rangrez.glimmer import SYSTEM, answer, notebook_prompt, prompt

ROOT = Path(__file__).parent / "glimmer"
K_MAX = 6


def main() -> None:
    recs = json.loads((ROOT / "records.json").read_text())
    by_prompt = {prompt(d["measured_lab"]): d for d in recs["train"]}
    holdout = [json.loads(line) for line in (ROOT / "run" / "holdout.jsonl").read_text().splitlines()]
    targets = [by_prompt[next(m["content"] for m in h["messages"] if m["role"] == "user")] for h in holdout]
    held = {id(d) for d in targets}
    train = [d for d in recs["train"] if id(d) not in held]
    labs = np.array([d["measured_lab"] for d in train])
    rng = np.random.default_rng(0)

    def neighbours(lab, exclude: int | None, k: int) -> list:
        des = delta_e_2000(labs, np.broadcast_to(np.asarray(lab, dtype=float), labs.shape))
        if exclude is not None:
            des[exclude] = np.inf
        return [(train[i]["conc"], train[i]["measured_lab"]) for i in np.argsort(des)[:k]]

    def row(d, nb) -> str:
        return json.dumps({"messages": [{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": notebook_prompt(d["measured_lab"], nb)},
                                        {"role": "assistant", "content": answer(d["conc"])}]})

    out = ROOT / "notebook"
    out.mkdir(parents=True, exist_ok=True)
    # Training: a random notebook size per row, so Glimmer works from an empty memory to a full one.
    ks = rng.integers(0, K_MAX + 1, len(train))
    (out / "train.jsonl").write_text("".join(row(d, neighbours(d["measured_lab"], i, int(k))) + "\n"
                                             for i, (d, k) in enumerate(zip(train, ks))))
    # Held out: a lab with a full notebook (the 6 nearest of the 2,900 training dips).
    (out / "holdout.jsonl").write_text("".join(row(d, neighbours(d["measured_lab"], None, K_MAX)) + "\n" for d in targets))
    print(f"{len(train)} training rows, {len(targets)} held out -> {out}")
    print(json.loads((out / "holdout.jsonl").read_text().splitlines()[0])["messages"][1]["content"])


if __name__ == "__main__":
    main()
