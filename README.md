# Rangrez

**A self-driving dye lab on Meta models.** Show Rangrez a colour, and a $100 rig mixes real dye, dips real cotton, photographs it, and corrects itself until the cloth matches.

*Rangrez* (रंगरेज़) means *dyer*.

## Why

A textile buyer approves a shade by "lab dip": the dye house mixes a recipe, dyes a swatch, and sends it for approval. If the buyer rejects it, the house corrects and dyes again. Each round costs days. Large mills use spectrophotometers and colour-matching software; the thousands of small dye houses in Tiruppur, Surat, Erode and Bhiwandi mostly use experience and repeated trials.

Rangrez automates the colourist's loop on a phone camera and an old-fashioned set of pumps: recipe, dip, measure, correct.

## How it works

```
target colour ──► Muse Wonder (lead colourist)
                    │ picks the next recipe from deterministic tools:
                    │   Kubelka-Munk dye model · correction learned from real dips
                    ▼
               safety gate ──► pumps · stir · dip cotton tape · dry
                                            │
                     phone camera ◄─────────┘
                          │
                     SAM 3.1 finds the colour card and the dyed segment
                          │
             card-corrected CIELAB ± uncertainty ──► CIEDE2000 to target
                          │
                matched? stop.  closer? refit and correct.  out of gamut? say so.
```

| Meta model | Job |
| --- | --- |
| **SAM 3.1** | Text-prompted segmentation of the 24-patch colour card and the dyed cotton in every frame. The pixels inside the masks become a calibrated colour measurement, so SAM works as a measuring instrument. It also reads the colour of a named object ("kurta") in a customer's photo |
| **Muse Wonder** | Runs the campaign through tool calls: suggests recipes, refits the dye model, writes the lab notebook, decides when to stop or escalate |

### Engineering choices

- **The model never drives a pump.** Every recipe passes a deterministic gate (concentration caps, bath volume, stock levels, dip time). The ESP32 firmware enforces the same limits again, behind a hardware e-stop.
- **Numbers come from tools, not from prose.** Colour maths (CIELAB, CIEDE2000, colour-card correction, Kubelka-Munk) is plain, tested code; the agent reasons over its results.
- **It learns this dye lot.** After every dip the dye model refits per-dye uptake. Near the target, Rangrez switches to a colourist-style correction: it takes the best real dip and corrects it by its measured error, using the refitted model's sensitivity. The model alone always predicts a perfect match, but no model captures everything (saturation, lot strength).
- **It knows what it can't do.** If dips at the safety cap still miss, the shade is outside this dye set's gamut. Rangrez stops and reports the closest dip instead of wasting cotton.
- **It escalates.** Three dips without real improvement (beyond measurement noise) hand control back to a human.
- **It measures cloth, not liquid.** A cup of dye does not predict the shade on fabric, so every reading is taken on a dried, dyed cotton segment.

## Run it

```bash
uv sync
uv run rangrez-lab            # http://127.0.0.1:8000
```

Without an API key it runs the full loop in the built-in simulator with an offline policy and segmenter. The simulator has a hidden "real" dye lot, pump noise, white-balance drift and JPEG frames. Add a key to use hosted SAM 3.1 and Muse Wonder:

```bash
cp .env.example .env          # set MODEL_API_KEY
uv run rangrez-lab
```

### Run it at home for free (assisted mode)

You can run the lab with no hardware: you are the pumps and your phone is the camera.

| Need | Free option |
| --- | --- |
| Colours | Blue, red and yellow sketch-pen ink or food colour |
| Cloth | Strips of an old white cotton T-shirt or handkerchief |
| Measuring | Count drops with a dropper or straw |
| Reference | A sheet of white paper; the printed colour card is better if you have one |
| Camera | Your phone (upload) or a webcam / Continuity Camera |

```bash
RANGREZ_RIG=assisted RANGREZ_DYES=home RANGREZ_HOST=0.0.0.0 uv run rangrez-lab
```

1. Open the dashboard on your laptop, or from your phone at `http://<laptop-ip>:8000`.
2. Photograph your target cloth on the paper and press "Read colour with SAM 3.1".
3. Start the campaign. For every dip the dashboard tells you the drops of each colour and the dip time.
4. Dip, dry, lay the strip on the same paper under the same light, and send the photo.
5. SAM 3.1 measures it and Muse Wonder decides the next dip.

Without the colour card, Rangrez white-balances against the paper. That is about 2–3 ΔE00 less precise, and every reading says so.

On the real rig, see [hardware/BUILD.md](hardware/BUILD.md):

```bash
RANGREZ_RIG=serial RANGREZ_PORT=/dev/cu.usbserial-0001 uv run rangrez-lab
```

Tests:

```bash
uv run pytest
```

The tests cover:
- the CIEDE2000 reference pairs;
- colour-card correction;
- the Kubelka-Munk model;
- measurement accuracy on rendered frames against the simulator's hidden truth;
- the safety gate;
- convergence on reachable shades;
- early stopping on out-of-gamut ones.

## Repository

| Path | What |
| --- | --- |
| `src/rangrez/colour.py` | sRGB / CIELAB / CIEDE2000, colour-card correction |
| `src/rangrez/km.py` | Spectral Kubelka-Munk model of reactive dye on cotton |
| `src/rangrez/vision.py` | SAM 3.1 segmentation, calibrated measurement, target-from-photo |
| `src/rangrez/rig.py` | Recipes, the safety gate, the rig interface |
| `src/rangrez/campaign.py` | The colourist's tools: refit, suggest, dip, escalate |
| `src/rangrez/agent.py` | Muse Wonder tool loop, plus the offline policy |
| `src/rangrez/sim.py` | Virtual dye lab with hidden chemistry and a rendering camera |
| `src/rangrez/serial_rig.py` | Driver for the real rig |
| `firmware/rangrez_rig/` | ESP32 firmware with independent hard limits |
| `hardware/BUILD.md` | Parts, wiring, chemistry, calibration |

## Built for

The Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026), Push the Frontier theme. Everything in this repository was written during the event.
