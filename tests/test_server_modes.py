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
