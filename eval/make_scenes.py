"""Generate photo-realistic scene templates with Muse Image.

Each scene shows an UNDYED white cotton strip (and optionally a target cloth) on
white printer paper. The evaluator recolours the strip to a known true colour,
so every test photo has a right answer. These are synthetic test fixtures, never
real dips.
"""
import base64, io, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
load_dotenv(".env")
from openai import OpenAI
from PIL import Image
from rangrez.photoreal import SCENE_DIR

client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"], timeout=240)
COMMON = ("Realistic unedited smartphone photo, {view}, of a narrow strip of plain undyed white cotton handkerchief fabric, "
          "about 3 cm wide and 12 cm long, with frayed edges and visible weave, lying flat on a plain white A4 printer paper sheet. "
          "{extra} {light} No text, no hands, no other objects.")
SCENES = {
    "lamp_desk": dict(view="top-down", extra="The paper is on a light wooden desk.", light="Lit by a warm yellow desk lamp from the upper right, soft shadows."),
    "daylight_window": dict(view="top-down", extra="The paper is on a grey table.", light="Lit by cool daylight from a window, even and soft."),
    "tilted_phone": dict(view="taken from a slight angle, about 25 degrees from vertical", extra="The paper is on a dark wooden table.", light="Lit by a white ceiling tube light."),
    "vertical_strip": dict(view="top-down", extra="The strip runs vertically down the paper. The paper is on a bedsheet.", light="Lit by a warm room light, a little dim."),
    "with_target": dict(view="top-down", extra="Next to the strip on the same paper lies a larger plain light grey cotton fabric square about 8 cm wide.", light="Lit by a neutral desk lamp."),
    "shadowy": dict(view="top-down", extra="The paper is on a wooden desk, a phone shadow falls across one corner of the paper but not on the strip.", light="Lit by a single overhead bulb."),
    "phone_flash": dict(view="top-down, taken at night with the phone flash on, a bright flash hotspot in the middle of the paper", extra="The paper is on a dark table.", light="Only the phone flash lights the scene."),
    "kitchen_tube": dict(view="top-down", extra="The paper is on a grey granite kitchen counter with a steel tumbler at the far edge of the frame.", light="Lit by a cool white kitchen tube light."),
    "evening_warm": dict(view="top-down", extra="The paper is on a floral bedsheet.", light="Lit by a very warm orange evening bulb, dim and yellowish."),
    "crumpled_strip": dict(view="top-down", extra="The strip is crumpled and creased with deep wrinkles after drying. The paper is on a wooden desk.", light="Lit by soft daylight."),
}
# Scenes that break the usual layout, with their own prompts.
CUSTOM = {
    "no_paper": "Realistic unedited smartphone photo, top-down, of a narrow strip of plain undyed white cotton handkerchief fabric, about 3 cm wide and 12 cm long, "
                "with frayed edges and visible weave, lying directly on a medium brown wooden table. No paper. Lit by a warm room light. No text, no hands, no other objects.",
    "light_box": "Realistic unedited photo from a fixed webcam looking straight down inside a small homemade white light box lit by LED strips, "
                 "of a strip of plain undyed white cotton tape about 2.5 cm wide running across a white printer paper sheet, the tape coming in from the left edge "
                 "and leaving at the right edge of the frame. Slightly cool LED light, a soft vignette in the corners. No text, no hands, no other objects.",
    "kurta_target": "Realistic unedited smartphone photo, top-down, of a plain undyed off-white cotton kurta folded neatly on a white A4 printer paper sheet, "
                    "the paper on a bed with a plain sheet. Lit by daylight from a window. No text, no hands, no other objects.",
}

def make(name):
    t = time.time()
    r = client.images.generate(model="muse-image-1.0", prompt=CUSTOM[name] if name in CUSTOM else COMMON.format(**SCENES[name]))
    d = r.data[0]
    raw = base64.b64decode(d.b64_json)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((1600, 1600))
    img.save(SCENE_DIR / f"{name}.jpg", quality=92)
    return name, round(time.time() - t, 1)

names = sys.argv[1:] or [*SCENES, *CUSTOM]
with ThreadPoolExecutor(4) as pool:
    for name, secs in pool.map(make, names):
        print(name, "ok", secs, "s")
