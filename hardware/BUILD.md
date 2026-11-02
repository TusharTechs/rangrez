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
| Normally-closed e-stop button + 10 kΩ resistor | 1 | 100 | Wired to GPIO 34 with an external pull-up |
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
| 34 | E-stop sense (normally closed to GND), plus a 10 kΩ pull-up from GPIO 34 to 3.3 V. GPIO 34 has no internal pull-up; without the resistor the e-stop input floats |

**Safety wiring, not optional.**
- **The e-stop cuts power, it doesn't just inform the code.** Wire the button's second normally-closed contact in series with the coil of a relay or contactor that switches the 12 V actuator rail (pumps, stirrer, blower, servo). Pressing it stops every motor even if the ESP32 has hung; GPIO 34 only tells the firmware why.
- **100 kΩ pull-down on every MOSFET gate (GPIO 16-23, 25).** While the ESP32 boots its pins float, and a floating gate can switch a pump on.
- **The firmware refuses to clear the e-stop while the button is still pressed**, accepts pump calibration only within ±20% of the compiled value, and runs an 8 s watchdog: a hung loop reboots the board with everything off.
- **Drain is timed, not sensed.** For unattended runs add a float switch (about ₹100) to stop overfilling if the drain clogs; the firmware's 25 mL cup limit assumes the drain worked.

## Chemistry

- **Stocks:** 20 g/L of each dye in warm water. Mix fresh each session; reactive dyes hydrolyse within hours once dissolved.
- **Tape:** soak in soda ash, 100 g/L, for 15 minutes, then dry before loading. Cold reactive dye reacts with the cotton over time, not on contact.
- **Fixing and wash-off:** a lab dip is only comparable with bulk after the dye has fixed and the unfixed dye is washed off. Batch each dyed segment for at least 10 minutes (longer for deep shades), then rinse, soap and dry it before it is measured. The rig as drawn has no rinse station: until it does, run the rinse and soap by hand, or treat its readings as a pre-screen that runs slightly deeper than the washed shade.
- **Safety:** dye powders are respiratory sensitisers. Weigh powders wearing a dust mask and gloves; the rig only ever handles dissolved stock.
- **Disposal:** collect waste in the drain jar and dilute heavily before disposal.

## Calibration

1. **Pumps.** Run each pump for 10 s into a cup on a kitchen scale and divide to get ms per mL. Send `{"cmd":"calibrate","id":N,"ms_per_ml":X}` over serial.
2. **Camera.** Lock exposure and white balance. Frame the card and the dry segment together; a 1080p view of the whole light box is plenty.
3. **Check.** Run `RANGREZ_RIG=serial uv run rangrez-lab` and dye a plain white segment (0, 0, 0). It should read L* above 88.
