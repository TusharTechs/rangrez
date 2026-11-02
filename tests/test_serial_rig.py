"""The robot path end to end with no parts: the real SerialRig driver on a pseudo serial port,
talking to an emulator of the ESP32 firmware (limits read from the .ino), with its light-box
webcam recording real-looking video that the driver grabs through OpenCV."""

import json
import time

import numpy as np
import pytest
import serial

from rangrez.agent import run_deterministic
from rangrez.campaign import Campaign
from rangrez.colour import delta_e_2000
from rangrez.esp32_emulator import firmware_limits, start
from rangrez.rig import Recipe
from rangrez.serial_rig import RigError, SerialRig


@pytest.fixture
def rig():
    board, port = start(seed=5)
    r = SerialRig(port, camera=board.camera_path, boot_seconds=0.05, settle_seconds=0)
    r.board = board
    yield r
    r.link.close()


def test_emulator_uses_the_firmware_limits():
    lim = firmware_limits()
    assert lim["MAX_PUMP_ML"] == 20.0 and lim["MAX_BATH_ML"] == 25.0 and lim["MAX_DIP_S"] == 300.0
    assert len(lim["msPerMl"]) == 5


def test_a_dip_runs_the_full_protocol_and_photographs_the_tape(rig):
    shot = rig.run_shot(Recipe((3.0, 0.5, 0.5)))
    cmds = [c["cmd"] for c in rig.board.log]
    assert cmds[:5] == ["ping", "drain", "pump", "stir", "drain"]  # rinse first
    assert cmds[-6:] == ["stir", "advance", "dip", "advance", "dry", "light"]  # mix, dip, dry, photograph
    assert shot.image_jpeg[:2] == b"\xff\xd8"  # a JPEG grabbed from the webcam video


def test_firmware_refuses_what_a_buggy_host_might_send(rig):
    with pytest.raises(RigError, match="ml out of range"):
        rig.cmd({"cmd": "pump", "id": 0, "ml": 30})
    with pytest.raises(RigError, match="pump id"):
        rig.cmd({"cmd": "pump", "id": 4, "ml": 1})  # the drain pump is not a dye pump
    rig.cmd({"cmd": "pump", "id": 3, "ml": 20})
    with pytest.raises(RigError, match="overflow"):
        rig.cmd({"cmd": "pump", "id": 3, "ml": 10})
    with pytest.raises(RigError, match="dip seconds"):
        rig.cmd({"cmd": "dip", "seconds": 900})


def test_estop_latches_until_reset(rig):
    rig.board.press_estop()
    with pytest.raises(RigError, match="e-stop latched"):
        rig.cmd({"cmd": "stir", "seconds": 2})
    rig.cmd({"cmd": "reset"})
    rig.cmd({"cmd": "stir", "seconds": 2})


def test_estop_during_a_pump_is_reported_not_counted_as_done():
    board, port = start(seed=2, time_scale=1.0)
    link = serial.Serial(port, 115200, timeout=5)
    link.write(b'{"cmd": "pump", "id": 3, "ml": 5}\n')  # about 4.5 s of pumping
    time.sleep(0.3)
    board.press_estop()
    reply = json.loads(link.readline())
    assert reply["error"] == "e-stop during pump"


def test_robot_campaign_matches_a_shade_end_to_end(rig):
    target = rig.board.rig.true_swatch_lab(Recipe((2.0, 1.2, 0.8)))
    c = Campaign(target, rig, rig.board.rig.segmenter(), tolerance=1.5, max_shots=8)
    run_deterministic(c)
    assert c.status == "matched", c.state()
    # The camera's photos measure what the hidden chemistry really dyed.
    assert float(delta_e_2000(c.shots[-1].measurement.lab, rig.board.true_lab)) < 2.5
    # Aimed at a colour typed in as numbers, the true error is the tolerance plus the light
    # box's measurement error (about 1-2 dE00); a cloth beside the strip removes the second.
    assert float(delta_e_2000(rig.board.true_lab, target)) < 1.5 + 2.0
    assert np.isfinite(c.shots[-1].delta_e)


def test_tape_across_the_paper_is_measured_not_a_piece_of_paper():
    """Live SAM, light box: the tape splits the sheet, and "fabric" also returns the upper half
    of the paper. The tape (which runs off the paper) must still be what gets measured."""
    from rangrez.photoreal import SceneSegmenter
    from rangrez.rig import execute
    from rangrez.sim import VirtualRig
    from rangrez.vision import measure

    rig = VirtualRig(seed=3, scene="light_box")
    jpeg = execute(rig, Recipe((3.0, 0.5, 0.5))).image_jpeg
    upper = np.zeros_like(rig.masks["paper"])
    upper[134:474, 278:1315] = True
    upper &= ~rig.masks["strip"]
    masks = {**rig.masks, "paper": rig.masks["paper"] | upper, "target": upper}  # SAM: a paper piece is "fabric" too
    m = measure(jpeg, SceneSegmenter(masks), with_target=True)
    assert float(delta_e_2000(m.lab, rig.last_swatch_lab)) < 2.5
    assert m.target_lab is None  # the paper piece is not a cloth beside the strip
