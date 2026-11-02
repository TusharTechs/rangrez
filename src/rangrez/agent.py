"""Muse Wonder runs the campaign; deterministic rules keep it honest.

The model chooses which tool to call and explains why in the lab notebook.
Code enforces the rest: the shot budget, the safety gate, and escalation to a
human after three dips in a row that fail to improve on the best match.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable

import numpy as np

from .campaign import Campaign
from .profiles import PROFILE
from .rig import DYES
from .vision import BASE_URL, ApiCall, CallLog

WONDER_MODEL = os.environ.get("RANGREZ_AGENT_MODEL", "muse-wonder-1.0")
NO_PROGRESS_LIMIT = 3
MIN_IMPROVEMENT_DE00 = 0.3  # smaller gains are within dip-to-dip noise
CORRECTION_ZONE_DE00 = 6.0
OUT_OF_GAMUT_STREAK = 2  # dips at the concentration cap that still miss

SYSTEM_PROMPT = """You are the colourist running Rangrez, a small self-driving dye lab for cotton.
Goal: match the target shade on a real cotton strip to CIEDE2000 <= the tolerance, in as few dips as possible.

You control the lab only through tools. Rules:
- Never invent a colour, concentration or measurement. Every number you state must come from a tool result.
- Before every run_shot, call suggest_recipes. You may pick a candidate or blend candidates, but explain why.
- After each shot with a usable measurement, call fit_model so the dye model learns this dye lot.
- For the first dip, if a glimmer candidate is offered, use it: Glimmer learned this dye lot from the lab's own dips.
  Otherwise, if lab_memory is offered and its remembered dip was within about dE00 6 of the target, use it:
  it corrects a real dip of this very dye lot.
- Far from the target, trust model_inverse. Once the best dip is within about dE00 6, prefer colourist_correction:
  the model always predicts a perfect match but cannot capture everything about this dye lot (saturation, lot strength),
  while the correction is learned from real dips.
- If a shot is flagged REJECT, do not trust its colour; re-run or change something and say what.
- Write one short notebook entry (write_note) per shot: hypothesis, what you changed, what you expect.
- If dips at the concentration cap still miss, the shade is outside this dye set's gamut: say so and report the closest dip.
- Stop with finish when the target is matched or the budget is spent. Use escalate if something looks physically wrong.
Dyes: {dyes}. {description}."""

TOOLS = [
    {"type": "function", "function": {"name": "get_state", "description": "Target, shots so far, best match, model gains, stock levels.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "fit_model", "description": "Refit the Kubelka-Munk dye model to all valid shots.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "suggest_recipes", "description": "Candidate recipes from model inversion and colourist correction, each with predicted dE00 and gate status.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "run_shot", "description": "Dispense, dip a cotton strip, dry, photograph and measure. Gated by hard safety limits.", "parameters": {"type": "object", "properties": {
        "recipe": {"type": "object", "properties": {d: {"type": "number"} for d in DYES}, "required": list(DYES)},
        "dip_seconds": {"type": "number", "description": "30-300, default 90"},
        "rationale": {"type": "string"}}, "required": ["recipe", "rationale"]}}},
    {"type": "function", "function": {"name": "write_note", "description": "Append to the lab notebook.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "escalate", "description": "Stop and hand over to the human colourist.", "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}},
    {"type": "function", "function": {"name": "finish", "description": "End the campaign with a short summary.", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
]


def _no_progress(c: Campaign) -> int:
    """Dips since the best match last improved by more than measurement noise."""
    best_de, since = float("inf"), 0
    for s in c.shots:
        if s.measurement.ok and s.delta_e < best_de - MIN_IMPROVEMENT_DE00:
            best_de, since = s.delta_e, 0
        else:
            since += 1
    return since


def _out_of_gamut(c: Campaign) -> bool:
    """The last dips pushed a dye to its safety cap and still missed: unreachable."""
    recent = [s for s in c.shots if s.measurement.ok][-OUT_OF_GAMUT_STREAK:]
    if len(recent) < OUT_OF_GAMUT_STREAK:
        return False
    cap = 0.98 * c.rig.limits.max_conc_g_per_l
    return all(max(s.recipe.conc) >= cap and s.delta_e > c.tolerance for s in recent)


def dispatch(c: Campaign, name: str, args: dict) -> dict:
    if name == "get_state":
        return c.state()
    if name == "fit_model":
        return c.fit_model()
    if name == "suggest_recipes":
        return c.suggest_recipes()
    if name == "run_shot":
        out = c.run_shot(args["recipe"], float(args.get("dip_seconds", 90.0)), args.get("rationale", ""))
        if c.status == "running" and _out_of_gamut(c):
            c.declare_out_of_gamut()
            out["out_of_gamut"] = True
        elif c.status == "running" and _no_progress(c) >= NO_PROGRESS_LIMIT:
            c.escalate(f"{NO_PROGRESS_LIMIT} dips in a row without improving on the best match")
            out["escalated"] = True
        return out
    if name == "write_note":
        c.note(args["text"])
        return {"ok": True}
    if name == "escalate":
        return c.escalate(args["reason"])
    if name == "finish":
        if c.status == "running":
            c.status = "finished"
        c.note(args.get("summary", ""), who="agent")
        c.emit("status", status=c.status)
        return {"ok": True}
    return {"error": f"unknown tool {name}"}


def run_with_wonder(c: Campaign, log: CallLog | None = None, max_turns: int = 60) -> None:
    """Drive the campaign with Muse Wonder over the Meta Model API."""
    try:
        _wonder_loop(c, log, max_turns)
    finally:
        c.close()


def _wonder_loop(c: Campaign, log: CallLog | None, max_turns: int) -> None:
    from openai import OpenAI

    log = log or CallLog()
    client = OpenAI(base_url=BASE_URL, api_key=os.environ["MODEL_API_KEY"], timeout=120, max_retries=2)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(dyes=", ".join(DYES), description=PROFILE.description)},
        {"role": "user", "content": "New campaign. Current state:\n" + json.dumps(c.state())},
    ]
    for _ in range(max_turns):
        if c.status != "running":
            return
        started = time.monotonic()
        try:
            resp = client.chat.completions.create(model=WONDER_MODEL, messages=messages, tools=TOOLS)
        except Exception as exc:
            log.add(ApiCall(WONDER_MODEL, "plan next step", time.monotonic() - started, False, str(exc)))
            raise
        usage = getattr(resp, "usage", None)
        log.add(ApiCall(WONDER_MODEL, "plan next step", time.monotonic() - started, True,
                        f"{getattr(usage, 'total_tokens', '?')} tokens"))
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if msg.content:
            c.emit("thought", text=msg.content)
        if not msg.tool_calls:
            messages.append({"role": "user", "content": "Continue with a tool call, or call finish."})
            continue
        for call in msg.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            c.emit("tool", name=call.function.name, args=args)
            result = dispatch(c, call.function.name, args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=float)})
    if c.status == "running":
        c.escalate("agent turn budget exhausted")


def run_deterministic(c: Campaign, on_step: Callable[[], None] | None = None) -> None:
    """Offline policy with the same tools and rules: used in tests and replay mode."""
    try:
        _deterministic_loop(c, on_step)
    finally:
        c.close()


def _deterministic_loop(c: Campaign, on_step: Callable[[], None] | None) -> None:
    while c.status == "running" and len(c.shots) < c.max_shots:
        if c.shots:
            dispatch(c, "fit_model", {})
        cands = [x for x in c.suggest_recipes()["candidates"] if x["gate"] == "pass"]
        if not cands:
            dispatch(c, "escalate", {"reason": "no candidate passes the safety gate"})
            break
        # A colourist starts from the model, then corrects from real dips once close.
        near = c.best is not None and c.best.delta_e < CORRECTION_ZONE_DE00
        correction = [x for x in cands if x["method"] == "colourist_correction"]
        first = [x for x in cands if x["method"] == "glimmer"]
        remembered = [x for x in cands if x["method"] == "lab_memory"
                      and c.memory_source_de is not None and c.memory_source_de < CORRECTION_ZONE_DE00]
        if first:
            pick = first[0]
        elif remembered:
            pick = remembered[0]
        else:
            pick = correction[0] if near and correction else min(cands, key=lambda x: x["predicted_delta_e00"])
        dispatch(c, "write_note", {"text": f"Shot {len(c.shots) + 1}: {pick['method']} predicts dE00 {pick['predicted_delta_e00']}."})
        dispatch(c, "run_shot", {"recipe": pick["recipe"], "rationale": pick["why"]})
        if on_step:
            on_step()
    if c.status == "running":
        dispatch(c, "finish", {"summary": "Shot budget spent."})


def best_delta_e(c: Campaign) -> float:
    b = c.best
    return float("inf") if b is None else b.delta_e


__all__ = ["run_with_wonder", "run_deterministic", "dispatch", "best_delta_e", "np"]
