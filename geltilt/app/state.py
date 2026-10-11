"""Per-sample in-memory state retained for the session."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from .domain import AppliedRotation, EstimateOutcome, SampleState, SessionState


class SampleStateStore:
    """Keyed by sample name; holds applied rotation, last estimate and failure."""

    def __init__(self, initial: Optional[Dict[str, SampleState]] = None) -> None:
        self._states: Dict[str, SampleState] = dict(initial or {})

    def __contains__(self, name: str) -> bool:
        return name in self._states

    def get(self, name: str) -> Optional[SampleState]:
        return self._states.get(name)

    def get_or_create(self, name: str) -> SampleState:
        state = self._states.get(name)
        if state is None:
            state = SampleState(name=name)
            self._states[name] = state
        return state

    def set_applied(self, name: str, applied: AppliedRotation) -> SampleState:
        state = self.get_or_create(name)
        state.applied = applied
        return state

    def set_estimate(self, name: str, outcome: EstimateOutcome) -> SampleState:
        state = self.get_or_create(name)
        state.last_estimate = outcome
        return state

    def mark_failed(self, name: str, failed: bool = True) -> SampleState:
        state = self.get_or_create(name)
        state.failed = failed
        return state

    def rotations(self) -> Dict[str, AppliedRotation]:
        return {name: state.applied for name, state in self._states.items()}

    def clear(self) -> None:
        self._states.clear()

    def names(self) -> Iterable[str]:
        return self._states.keys()

    @classmethod
    def from_session(cls, session: SessionState) -> "SampleStateStore":
        states = {
            name: SampleState(name=name, applied=rot)
            for name, rot in session.rotations.items()
        }
        return cls(states)