"""On a public demo the Meta models stay locked unless the live passcode is given."""

from rangrez.server import Lab


def test_private_lab_with_a_key_is_live(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "x")
    monkeypatch.delenv("RANGREZ_LIVE_PASSCODE", raising=False)
    monkeypatch.delenv("RANGREZ_OFFLINE", raising=False)
    assert Lab().live(None)


def test_public_demo_needs_the_passcode(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "x")
    monkeypatch.setenv("RANGREZ_LIVE_PASSCODE", "indigo-42")
    monkeypatch.delenv("RANGREZ_OFFLINE", raising=False)
    lab = Lab()
    assert not lab.online
    assert not lab.live(None) and not lab.live("wrong")
    assert lab.live("indigo-42")


def test_no_key_is_never_live(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setenv("RANGREZ_LIVE_PASSCODE", "indigo-42")
    assert not Lab().live("indigo-42")


def test_report_describes_the_difference_like_a_colourist():
    from rangrez.report import difference

    d = difference([50, 0, 30], [48, 2, 33])  # a yellow that came out darker, more saturated and redder
    assert d["dL"] < 0 and d["dC"] > 0
    assert "darker" in d["words"] and "redder" in d["words"]
    assert difference([50, 0, 30], [50.1, 0, 30.1])["words"] == "no visible difference"


def test_report_renders_a_campaign(monkeypatch):
    monkeypatch.setenv("RANGREZ_MEMORY", "off")
    from rangrez.agent import run_deterministic
    from rangrez.campaign import Campaign
    from rangrez.report import render
    from rangrez.rig import Recipe
    from rangrez.sim import VirtualRig

    rig = VirtualRig(seed=4, scene="kitchen_tube")
    c = Campaign(rig.true_swatch_lab(Recipe((3.2, 0.6, 0.9))), rig, rig.segmenter())
    run_deterministic(c)
    page = render(c, agent="offline policy", api_calls=0)
    assert "PASS" in page and "Buyer approval" in page and page.count("<tr>") >= len(c.shots) + 4


def test_voice_intents_are_safe_and_simple():
    from rangrez.voice import intent

    assert intent("Done.") == "photo" and intent("Take the photo") == "photo"
    assert intent("repeat that please?") == "repeat" and intent("Stop the campaign.") == "stop"
    assert intent("the strip looks a bit patchy near the top") == "note"
    assert intent("it looks done but the top is patchy and darker") == "note"  # an observation, not a command
    assert intent("") == "none"
