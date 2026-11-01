# Building the Rangrez rig

About ₹8,000 (≈ $100) of parts on top of a phone or webcam. Prices are typical Indian retail and vary.

## Bill of materials

| Part | Qty | Approx. ₹ | Notes |
| --- | --- | --- | --- |
| ESP32 DevKit (WROOM-32) | 1 | 450 | Runs `firmware/rangrez_rig` |
| 12 V mini peristaltic pump | 5 | 2,250 | Turquoise, fuchsia, yellow, water, drain |
| Logic-level MOSFET module (or IRLZ44N + flyback diodes) | 7 ch | 300 | Pumps, stirrer, LED panel |
| 5 V relay module | 1 | 80 | Blower |
| MG996R servo | 1 | 350 | Lifts the dye cup up to the tape |
| 28BYJ-48 stepper + ULN2003 | 1 | 200 | Advances the cotton tape |
| Small 5 V DC motor + paddle | 1 | 100 | Stirrer |
| 12 V blower fan | 1 | 250 | Dries each dyed segment |
| High-CRI LED panel, 5000-6500 K | 1 | 600 | Fixed light inside the box |
| 12 V 5 A power supply + 5 V buck | 1 | 600 | |
| 50 mL cup, 3 × 250 mL bottles, silicone tubing | — | 300 | |
| Cotton twill tape, 15 mm, 10 m | 1 | 150 | Pre-soaked in soda ash, then dried |
| Cold fibre-reactive dyes: turquoise, fuchsia, lemon yellow | 3 | 1,500 | Procion MX or local reactive equivalents |
| Soda ash (sodium carbonate) | 500 g | 100 | Fixer, applied to the tape before use |
| Printed colour card (`uv run rangrez-card`) | 1 | 50 | Or an X-Rite ColorChecker for absolute accuracy |
| Normally-closed e-stop button | 1 | 100 | Wired to GPIO 34 |
| Cardboard or foam-board light box | 1 | — | Matte grey inside |

## Layout

```
          camera (phone / webcam, fixed exposure)
                      │
   ┌──────────────── light box ───────────────────┐
   │  [colour card]      [dyed tape segment]       │
   │                         ▲ dried by blower     │
   └─────────────────────────┼─────────────────────┘
        tape spool ──► dip station ──► stepper roller
                          ▲
                   cup on servo lift ◄── pumps ◄── 3 dye stocks + water
                          │
                        drain pump ──► waste jar
```

## Wiring (ESP32 GPIO)

| GPIO | Function |
| --- | --- |
| 16, 17, 18 | Dye pumps: turquoise, fuchsia, lemon yellow |
| 19 | Water pump |
| 21 | Drain pump |
| 22 | Stirrer |
| 23 | Blower relay |
| 25 | LED panel |
| 26 | Cup-lift servo |
| 27, 14, 12, 13 | Stepper IN1-IN4 |
| 34 | E-stop (normally closed to GND) |

## Chemistry

- **Stocks:** 20 g/L of each dye in warm water. Mix fresh each session; reactive dyes hydrolyse within hours once dissolved.
- **Tape:** soak in soda ash, 100 g/L, for 15 minutes, then dry before loading. The dye fixes on contact at room temperature.
- **Safety:** dye powders are respiratory sensitisers. Weigh powders wearing a dust mask and gloves; the rig only ever handles dissolved stock.
- **Disposal:** collect waste in the drain jar and dilute heavily before disposal.

## Calibration

1. **Pumps.** Run each pump for 10 s into a cup on a kitchen scale and divide to get ms per mL. Send `{"cmd":"calibrate","id":N,"ms_per_ml":X}` over serial.
2. **Camera.** Lock exposure and white balance. Frame the card and the dry segment together; a 1080p view of the whole light box is plenty.
3. **Check.** Run `RANGREZ_RIG=serial uv run rangrez-lab` and dye a plain white segment (0, 0, 0). It should read L* above 88.
