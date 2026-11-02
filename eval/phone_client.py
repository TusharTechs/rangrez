"""Plays the person (or, with the robot rig, just the customer) against a running dashboard, over HTTP only.

Like a real user it: photographs the customer's kurta beside the printed card and asks
SAM 3.1 for its colour; starts the campaign; then for every dip reads the drop counts
from the live event stream, "dyes" a strip (hidden chemistry, the drops counted the way
a person can), and sends a real-looking phone photo of it, alternating between a photo
upload and the "use webcam" button (the webcam is a video file the person's camera
"records"). Reports the agent's readings against the truth.

    RANGREZ_RIG=assisted RANGREZ_DYES=home RANGREZ_CAMERA=/tmp/rangrez-webcam.mp4 uv run rangrez-lab
    RANGREZ_DYES=home uv run python eval/phone_client.py --camera /tmp/rangrez-webcam.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
import uuid

import cv2
import numpy as np

from rangrez.assisted import AssistedRig
from rangrez.colour import delta_e_2000, lab_to_hex
from rangrez.photoreal import Scene, render
from rangrez.rig import Recipe
from rangrez.sim import VirtualRig


def post(base: str, path: str, body: dict | None = None, photo: bytes | None = None, fields: dict | None = None) -> dict:
    if photo is not None:
        b = uuid.uuid4().hex
        parts = [f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode() for k, v in (fields or {}).items()]
        parts.append(f"--{b}\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"IMG_2041.jpg\"\r\n"
                     "Content-Type: image/jpeg\r\n\r\n".encode() + photo + f"\r\n--{b}--\r\n".encode())
        data, ctype = b"".join(parts), f"multipart/form-data; boundary={b}"
    else:
        data, ctype = json.dumps(body or {}).encode(), "application/json"
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": ctype}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def events(base: str):
    with urllib.request.urlopen(base + "/api/events", timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if line.startswith("data:"):
                yield json.loads(line[5:])


def write_webcam_clip(path: str, jpeg: bytes) -> None:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    tmp = path + ".tmp.mp4"
    out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w - w % 2, h - h % 2))
    for _ in range(14):
        out.write(img[: h - h % 2, : w - w % 2])
    out.release()
    os.replace(tmp, path)


class Person(VirtualRig):
    quantize = AssistedRig.quantize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--camera", help="video file the dashboard's RANGREZ_CAMERA points at")
    ap.add_argument("--scene", default="lamp_desk")
    ap.add_argument("--drops", default="3,9,6", help="drops that dyed the customer's kurta")
    ap.add_argument("--beside", action="store_true", help="lay the customer's cloth beside every dip (scene with_target)")
    args = ap.parse_args()

    person = Person(seed=77, scene="with_target" if args.beside else args.scene, card=True)
    true_target = person.true_swatch_lab(Recipe(tuple(float(d) for d in args.drops.split(","))))
    if args.beside:
        person.target_lab = true_target  # the customer's cloth lies beside each strip
    kurta, _ = render(Scene.load("kurta_target"), target_lab=true_target, card=True, seed=5)
    t0 = time.time()
    read = post(args.base, "/api/target-from-photo", photo=kurta, fields={"concept": "kurta"})
    print(f"kurta photo -> {read['hex']} via {read['calibration']}  (truth {lab_to_hex(true_target)}, "
          f"off by {float(delta_e_2000(np.array(read['lab']), true_target)):.2f} dE00)")
    stream = events(args.base)
    post(args.base, "/api/campaign", {"lab": read["lab"], "tolerance": 2.0})
    dips, truths, api = 0, [], []
    for e in stream:
        if e["kind"] == "api":
            api.append(e)
        if e["kind"] == "assist":
            dips += 1
            amounts = tuple(float(e["amounts"][d]) for d in e["amounts"])
            print(f"dip {e['index']}: dashboard says {e['amounts']} {e['unit']}, dip {e['dip_seconds']} s")
            person.run_shot(Recipe(amounts, e["dip_seconds"]))
            truths.append(person.last_swatch_lab)
            photo = person.render()
            if args.camera and dips % 2 == 0:
                write_webcam_clip(args.camera, photo)
                post(args.base, "/api/assist/webcam")
            else:
                post(args.base, "/api/assist/photo", photo=photo)
        if e["kind"] == "shot":
            truth = f" (true dE00 to the kurta {float(delta_e_2000(truths[-1], true_target)):.2f})" if truths else ""
            print(f"   SAM + card read the strip: dE00 {e['delta_e']:.2f} to target{truth}")
        if e["kind"] == "note" and "side by side" in e.get("text", ""):
            print("   AGENT:", e["text"][:220])
        if e["kind"] == "status" and e.get("status") not in (None, "running", "waiting"):
            print(f"STATUS {e['status']}: {(e.get('reason') or '')[:200]}")
            break
    print(f"{dips} dips, {len(api)} Meta API calls ({sum(not a['ok'] for a in api)} failed), {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
