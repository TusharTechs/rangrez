"""From the buyer's message to a production plan.

Buyers send orders as chat messages ("need 500 kurtas in this teal, cotton poplin, by Friday,
buyer very strict on shade"). Muse Spark reads the message into a structured order; the
numbers that follow are plain arithmetic: how strict the shade must be sets the lab's pass
tolerance, and the approved dip's recipe scales to the order's fabric, dye bath and chemicals.
"""

from __future__ import annotations

import json
import os
import time

from .profiles import PROFILE
from .rig import DYES

SPARK_MODEL = "muse-spark-1.3"
TOLERANCE = {"strict": 1.0, "normal": 1.5, "relaxed": 2.0}  # CIEDE2000 pass threshold by how picky the buyer is
KG_PER_PIECE = {"kurta": 0.30, "shirt": 0.25, "t-shirt": 0.18, "saree": 0.55, "dupatta": 0.15, "kurti": 0.25,
                "towel": 0.40, "bedsheet": 0.70, "trouser": 0.45, "dress": 0.35}
KG_PER_METRE = 0.14  # 110 GSM poplin, 1.1 m wide, plus cutting waste
LIQUOR_RATIO = 10  # litres of dye bath per kg of cotton
SALT_G_PER_L, SODA_G_PER_L = 60, 20  # typical for medium shades of cold reactive dye on cotton

SCHEMA = {
    "type": "object",
    "properties": {
        "garment": {"type": "string", "description": "what is being dyed, singular, e.g. kurta, saree, fabric"},
        "quantity": {"type": "number"},
        "quantity_unit": {"type": "string", "enum": ["pieces", "metres", "kg"]},
        "fabric": {"type": "string"},
        "deadline": {"type": "string", "description": "delivery date as YYYY-MM-DD, or empty if not given"},
        "lab_dip_due": {"type": "string", "description": "when the buyer wants the lab dip, YYYY-MM-DD, or empty"},
        "shade_strictness": {"type": "string", "enum": ["strict", "normal", "relaxed"]},
        "notes": {"type": "string", "description": "anything else that matters for dyeing, short"},
    },
    "required": ["garment", "quantity", "quantity_unit", "fabric", "deadline", "lab_dip_due", "shade_strictness", "notes"],
    "additionalProperties": False,
}


def parse_order(message: str, log=None) -> dict:
    """Muse Spark reads a buyer's chat message into a structured order."""
    from openai import OpenAI

    from .vision import ApiCall

    client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"], timeout=90, max_retries=1)
    today = time.strftime("%A %Y-%m-%d")
    started = time.monotonic()
    try:
        r = client.chat.completions.create(
            model=SPARK_MODEL, reasoning_effort="low",
            messages=[{"role": "system", "content": f"You read dye orders for a small cotton dye house. Today is {today}. "
                                                    "Resolve relative dates. Use only what the message says."},
                      {"role": "user", "content": message}],
            response_format={"type": "json_schema", "json_schema": {"name": "order", "schema": SCHEMA, "strict": True}})
        order = json.loads(r.choices[0].message.content)
    except Exception as exc:
        if log is not None:
            log.add(ApiCall(SPARK_MODEL, "read the order", time.monotonic() - started, False, str(exc)[:200]))
        raise RuntimeError(f"could not read the order: {exc}") from exc
    if log is not None:
        log.add(ApiCall(SPARK_MODEL, "read the order", time.monotonic() - started, True))
    order["tolerance_de00"] = TOLERANCE.get(order.get("shade_strictness", "normal"), 1.5)
    return order


def fabric_kg(order: dict) -> float:
    q, unit = float(order.get("quantity") or 0), order.get("quantity_unit", "pieces")
    if unit == "kg":
        return q
    if unit == "metres":
        return q * KG_PER_METRE
    garment = order.get("garment", "").lower().rstrip("s")
    return q * KG_PER_PIECE.get(garment, 0.30)


def production_plan(order: dict, recipe_conc) -> dict | None:
    """The approved dip's recipe, scaled to the order. Reactive dyes (g/L) only."""
    if PROFILE.unit != "g/L":
        return None
    kg = fabric_kg(order)
    litres = kg * LIQUOR_RATIO
    return {
        "fabric_kg": round(kg, 1),
        "bath_litres": round(litres),
        "dye_g": {d: round(float(c) * litres) for d, c in zip(DYES, recipe_conc)},
        "salt_kg": round(SALT_G_PER_L * litres / 1000, 1),
        "soda_ash_kg": round(SODA_G_PER_L * litres / 1000, 1),
        "assumptions": f"{LIQUOR_RATIO}:1 liquor ratio; {KG_PER_PIECE.get(order.get('garment', '').lower().rstrip('s'), 0.30)} kg "
                       f"per {order.get('garment', 'piece')}; salt {SALT_G_PER_L} g/L and soda ash {SODA_G_PER_L} g/L, typical "
                       "for medium shades of cold reactive dye. Confirm with a bulk trial.",
    }


def draft_reply(message: str, facts: dict, log=None) -> str:
    """The message back to the buyer, in their language, using only the lab's numbers."""
    from openai import OpenAI

    from .vision import ApiCall

    client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"], timeout=90, max_retries=1)
    started = time.monotonic()
    try:
        r = client.chat.completions.create(
            model=SPARK_MODEL, reasoning_effort="low",
            messages=[{"role": "system", "content": (
                "You write the dye house's short chat reply to a buyer about their lab dip. Reply in the same language and "
                "script the buyer used (Hindi in Devanagari if they wrote Hindi, Hinglish if they wrote Hinglish). Use only "
                "the facts given; never invent numbers or promises. At most 70 words, warm and plain, no markdown. Say the "
                "lab-dip report is attached; if a preview exists, say a photo of their garment in the shade is attached.")},
                      {"role": "user", "content": f"Buyer's message:\n{message}\n\nFacts:\n{json.dumps(facts, ensure_ascii=False)}"}])
        text = (r.choices[0].message.content or "").strip()
    except Exception as exc:
        if log is not None:
            log.add(ApiCall(SPARK_MODEL, "reply to the buyer", time.monotonic() - started, False, str(exc)[:200]))
        raise RuntimeError(f"could not draft the reply: {exc}") from exc
    if log is not None:
        log.add(ApiCall(SPARK_MODEL, "reply to the buyer", time.monotonic() - started, True))
    return text
