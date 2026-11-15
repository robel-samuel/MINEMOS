"""Answer presentation contract for memory query outcomes.

The engine decides *what* is remembered and *what* a query returns.  This module
defines how a caller should present a query outcome: whether the returned facts
answer the question as a settled state, or whether they are contested,
ambiguous, abstained, unverified, empty, or produced without an eligibility
decision ("ungated").
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

ANSWER_STATUSES = (
    "settled",
    "contested",
    "ambiguous",
    "abstained",
    "unverified",
    "empty",
    "ungated",
)


def answer_contract(status: str, reason: Optional[str],
                    temporal: Mapping[str, Any],
                    row_count: int) -> Dict[str, Any]:
    """Response-level presentation contract for one query outcome.

    ``answer_is_settled`` is the authoritative flag: it is true only when the
    eligibility gate answered the question *and* the temporal analysis found
    no competing or unresolved values among the returned facts.  Every other
    state -- contested, ambiguous, abstained, unverified, empty or ungated --
    must be presented as not settled.  ``answer_status`` names the reason so a
    caller can distinguish, for example, "we found conflicting values" from
    "we could not identify the subject" without reading diagnostic metadata.

    ``ungated`` is never settled even when the ungated rows look temporally
    clean: no eligibility decision was made, so no subject was verified.
    """
    if status == "answered":
        if row_count == 0:
            answer_status = "empty"
        elif temporal.get("answer_is_settled") is True:
            answer_status = "settled"
        else:
            answer_status = "contested"
    elif status == "abstained":
        answer_status = (
            "ambiguous" if reason == "subject_ambiguous" else "abstained")
    elif status == "disabled":
        answer_status = "ungated"
    else:
        answer_status = "unverified"
    settled = answer_status == "settled"
    return {
        "answer_status": answer_status,
        "answer_is_settled": settled,
        "safe_to_display_as_settled": settled,
    }