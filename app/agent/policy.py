"""Project-wide policy constants used by every refusal path.

Per the Phase 4 refusal policy: one canonical user-facing refusal string for
all refusals (planner ``refuse`` tool, compliance precheck hard-block,
compliance postcheck override, verifier-driven refusal). Every gate stamps
its own ``source`` and trace fields so reviewers can tell which mechanism
caught the request, but the user sees the same words regardless of path.
"""

from typing import Final

CANONICAL_REFUSAL: Final[str] = (
    "This is not really what I was trained for, therefore I cannot answer. Try again."
)
