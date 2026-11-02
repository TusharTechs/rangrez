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
- **It evens out the light.** A desk lamp, a window or a shadow lights the strip and the card unequally. Rangrez fits the light across the frame from the white paper, ignoring cast shadows, and divides it out before measuring. If the light is too steep (a phone flash), or the card is unreadable, it says so and asks for a retake, or falls back to the paper.
- **It compares side by side.** If the customer's cloth lies beside the strip in the dip photo, Rangrez compares the two under the same light and re-anchors the target to that reading. A target read from a photo taken elsewhere carries that photo's light and camera error; the side-by-side reading does not.
- **It ignores the bedsheet.** SAM's "fabric" also finds the bedsheet or tablecloth under the paper. Only cloth lying on the paper is measured.
- **It measures cloth, not liquid.** A cup of dye does not predict the shade on fabric, so every reading is taken on a dried, dyed cotton segment.

## Run it

```bash
uv sync
uv run rangrez-lab            # http://127.0.0.1:8000
```

Without an API key it runs the full loop in the built-in simulator with an offline policy and SAM masks cached for each scene. The simulator has a hidden "real" dye lot and pump noise, and every dip comes back as a real-looking phone photo (see below). Add a key to use hosted SAM 3.1 and Muse Wonder:

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
| Camera | Your phone (upload, or streaming as an IP webcam) or a laptop webcam / Continuity Camera |

```bash
RANGREZ_RIG=assisted RANGREZ_DYES=home RANGREZ_HOST=0.0.0.0 uv run rangrez-lab
```

1. Open the dashboard on your laptop, or from your phone at `http://<laptop-ip>:8000`.
2. Photograph your target cloth on the paper and press "Read colour with SAM 3.1".
3. Start the campaign. For every dip the dashboard tells you the drops of each colour and the dip time.
4. Dip, dry, lay the strip on the same paper under the same light, and send the photo. If you have the customer's cloth, lay it beside the strip.
5. SAM 3.1 measures it and Muse Wonder decides the next dip.

Photos can be JPEG, PNG or iPhone HEIC, any orientation. `RANGREZ_CAMERA` picks the "Use webcam" source: a webcam index (`0`, `1`), a phone streaming as an IP webcam (`http://192.168.1.7:8080/video`), or a video file.

Without the colour card, Rangrez white-balances against the paper. Absolute colour can then be off by up to about 5 ΔE00. Matching still works, because the target and every dip are photographed on the same paper under the same light, so the error is shared. Every reading says which calibration it used.

On the real rig, see [hardware/BUILD.md](hardware/BUILD.md):

```bash
RANGREZ_RIG=serial RANGREZ_PORT=/dev/cu.usbserial-0001 uv run rangrez-lab
```

Without the parts, run the same driver against a virtual ESP32. It speaks the firmware's protocol on a pseudo serial port, with the hard limits read from the `.ino` itself, and has a light-box webcam that records video:

```bash
RANGREZ_RIG=serial RANGREZ_PORT=virtual uv run rangrez-lab
```

Every campaign is recorded to `recordings/<time>_<rig>/`:
- the customer's photo;
- every dip photo, with and without SAM's masks;
- every event: each Meta API call with its timing, Wonder's notes and tool calls, and each measurement.

`uv run python eval/demo_reel.py recordings/<run>` turns a run into a 1080p clip.

Tests:

```bash
uv run pytest
```

The tests cover:
- the CIEDE2000 reference pairs;
- colour-card correction;
- the Kubelka-Munk model;
- measurement against known truth on real-looking photos from 11 rooms and lights, with the card, with paper only, and with neither;
- uneven lamp light, a hard phone shadow, a phone flash, a crumpled strip, a card laid over the strip, a bedsheet under the paper, and a customer's kurta;
- a home-printed colour card, phone photos (rotated 12 MP JPEG, PNG, iPhone HEIC), a phone streaming as an IP webcam, and a video-file camera;
- the robot: the real serial driver against the virtual ESP32, which checks the full rinse–pump–stir–dip–dry–photograph sequence, firmware refusals, the e-stop latch, and a campaign matched from webcam video;
- the safety gate;
- convergence on reachable shades, every dip photographed in a different room;
- early stopping on out-of-gamut ones.

## Does it work on real-looking photos?

Every photo in the simulator and in the tests is a real-looking phone photo. Muse Image generated 12 scenes of undyed cotton on printer paper in Indian homes:
- a desk lamp and window daylight;
- a kitchen tube light and a warm evening bulb;
- a phone shadow, a phone flash and a tilted phone;
- a crumpled strip, a strip on a bare table, a bedsheet under the paper, and a kurta.

SAM 3.1 masked each scene once; the masks are cached in `src/rangrez/scenes/masks`. To photograph a dip, the cloth is recoloured to its true colour under the scene's own uneven light, keeping its real folds, weave and shadows. The paper's true whiteness and tint are randomised and hidden from the pipeline, as with real printer paper. Exposure, white balance, sensor noise and JPEG quality vary per photo. `uv run pytest` runs all of this offline.

Measurement with live SAM 3.1, 44 photos (`uv run python eval/accuracy.py`):

| Reference in the photo | Median ΔE00 | p90 |
| --- | --- | --- |
| Printed colour card | 0.66 | 2.3 |
| White paper only | 1.17 | 5.1 |

The worst readings are the phone-flash photos (5–9 ΔE00). Rangrez flags those and asks for a retake without flash. Photos with neither card nor paper are flagged as uncalibrated.

Full assisted campaigns, live SAM 3.1 and Muse Wonder, household drops profile (`RANGREZ_DYES=home uv run python eval/campaigns.py`). A simulated person follows the drop counts, hidden chemistry dyes the strip, and each dried strip comes back as a realistic phone photo. The target cloth is photographed first, on the same paper.

| Scene | Reference | Dips | Agent's final ΔE00 | True final ΔE00 |
| --- | --- | --- | --- | --- |
| Desk lamp | paper | 4 | 1.41 | 1.42 |
| Daylight | card | 6 | 1.31 | 1.34 |
| Phone shadow | paper | 4 | 1.29 | 1.40 |
| Tilted phone | card | 4 | 1.45 | 1.53 |
| Warm bulb, bedsheet | paper | 5 | 1.31 | 1.26 |
| Kitchen tube light | card | 5 | 1.61 | 1.72 |
| Crumpled strip | card | 5 | 1.03 | 1.20 |
| Bedsheet under paper | paper | 8 | 0.98 | 1.05 |

All 8 matched within the 2.0 tolerance, with no failed API calls; the agent's own reading agrees with the truth within 0.2. These are synthetic photos with simulated chemistry, not real dye on real cloth.

The whole app, end to end through the dashboard's HTTP API and in the browser, live SAM 3.1 and Muse Wonder (`eval/phone_client.py` plays the customer and the person). Every run is recorded:

| Run | Result |
| --- | --- |
| Robot rig (virtual ESP32, light-box webcam video), teal kurta photo | matched in 2 dips: 3.6 → 1.4 |
| Robot rig, plum beyond 10 g/L fuchsia | out of gamut after 2 dips at the cap, closest dip reported |
| Assisted, household drops, kurta photo, the cloth beside every dip | matched in 6 dips; true error 1.5 (the same scenario without the cloth beside the dips ended 7.2 off: the kurta photo and the dips were taken in different rooms) |
| Assisted, household drops, teal | escalated to a human after 3 dips without improvement |
| Assisted, household drops, indigo, cloth beside | 10 dips, best true error 2.8: no match within 2.0 |

Offline benchmark of the same tools, 24 shades each, every dip a realistic photo in a different room (`eval/benchmark.py`): reactive dyes on the robot rig matched 24/24 in a median of 2 dips. Household drops matched 19/24 in a median of 6. Household inks are a rough stand-in, measured in whole drops; strong blue-heavy shades are where they stall.

Reading a garment's colour from a customer's photo is the weakest step. A folded kurta covers half the paper, so the light under it has to be guessed from the paper around it, and live readings were 2.7–3.9 ΔE00 off even with the card. Lay the customer's cloth beside the strip whenever you have it.

## Repository

| Path | What |
| --- | --- |
| `src/rangrez/colour.py` | sRGB / CIELAB / CIEDE2000, colour-card correction |
| `src/rangrez/km.py` | Spectral Kubelka-Munk model of reactive dye on cotton |
| `src/rangrez/vision.py` | SAM 3.1 segmentation, calibrated measurement, target-from-photo |
| `src/rangrez/rig.py` | Recipes, the safety gate, the rig interface |
| `src/rangrez/campaign.py` | The colourist's tools: refit, suggest, dip, escalate |
| `src/rangrez/agent.py` | Muse Wonder tool loop, plus the offline policy |
| `src/rangrez/sim.py` | Virtual dye lab with hidden chemistry |
| `src/rangrez/photoreal.py`, `scenes/` | The simulator's camera: real-looking phone photos with known true colours |
| `src/rangrez/serial_rig.py` | Driver for the real rig |
| `src/rangrez/esp32_emulator.py` | Virtual ESP32: the firmware's protocol and limits on a pseudo serial port |
| `src/rangrez/camera.py` | Webcam, IP-webcam phone or video-file capture |
| `src/rangrez/recorder.py` | Records every campaign for review and the demo |
| `firmware/rangrez_rig/` | ESP32 firmware with independent hard limits |
| `hardware/BUILD.md` | Parts, wiring, chemistry, calibration |
| `eval/` | Scene generation (Muse Image), SAM masking, live accuracy and campaign benchmarks |

## Built for

The Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026), Push the Frontier theme. Everything in this repository was written during the event.
