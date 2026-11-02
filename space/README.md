---
title: Rangrez
emoji: 🧵
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: A self-driving dye lab on Meta models
---

# Rangrez: a self-driving dye lab

▶ **[Watch the 3-minute demo](https://youtu.be/g4Z30UhPtjA)** · [A recorded live run](?replay=1) · [The 2-minute tour](?tour=1)

Pick a colour and press **Start campaign**. Rangrez mixes a recipe, dyes a cotton strip in its
simulated lab (a hidden dye lot that differs from the supplier's model), photographs it like a
phone would, measures it, and corrects itself until the cloth matches. Every dip photo is a
real-looking scene: desk lamps, daylight, tube lights, shadows, paper of unknown whiteness.

Without the live passcode it runs the offline policy and SAM masks cached for each scene.
With it, hosted **SAM 3.1** measures every photo and **Muse Wonder** runs the campaign,
and you can read a target colour from your own photo ("match this kurta").

**Photograph your outfit:** SAM 3.1 reads its colour (a few free reads a day per visitor) and
Rangrez gives you a recipe to dye a matching cloth at home, in drops of blue, red and yellow
household colour.

With the passcode you can also paste the buyer's message (English or Hindi) for **Muse Spark** to
read into an order, run the lab **hands-free by voice** in English or Hindi (Muse Voice Transcribe),
and see the customer's garment in the approved shade (**Muse Image**, measured and corrected).

One lab, shared by everyone viewing: if a campaign is running, you are watching it.

Code, hardware, firmware and evaluation: https://github.com/TusharTechs/rangrez
Open weights: https://huggingface.co/TechTush/rangrez-glimmer-lora · Dataset: https://huggingface.co/datasets/TechTush/rangrez-lab-dips
