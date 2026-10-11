from __future__ import annotations

from geltilt.app.domain import AppliedRotation
from geltilt.app.state import SampleStateStore

from .conftest import make_outcome


def test_get_or_create_defaults():
    store = SampleStateStore()
    st = store.get_or_create("a.png")
    assert st.applied.source == "none"
    assert st.applied.value == 0.0
    assert st.last_estimate is None
    assert st.failed is False


def test_set_applied_and_estimate():
    store = SampleStateStore()
    store.set_applied("a.png", AppliedRotation(2.0, "manual"))
    store.set_estimate("a.png", make_outcome(angle=1.5))
    st = store.get("a.png")
    assert st.applied.value == 2.0 and st.applied.source == "manual"
    assert st.last_estimate.angle == 1.5


def test_mark_failed():
    store = SampleStateStore()
    store.mark_failed("a.png")
    assert store.get("a.png").failed is True


def test_from_session_seeds_rotations():
    from geltilt.app.domain import SessionState

    session = SessionState(rotations={"a.png": AppliedRotation(3.0, "auto")})
    store = SampleStateStore.from_session(session)
    assert store.get("a.png").applied.value == 3.0
    assert store.rotations()["a.png"].source == "auto"


def test_clear():
    store = SampleStateStore()
    store.get_or_create("a.png")
    store.clear()
    assert store.get("a.png") is None
    assert list(store.names()) == []