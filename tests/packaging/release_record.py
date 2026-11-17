"""How the code may move past a frozen release record: only through a recorded transition.

A release freezes what it shipped (``docs/release-v0.1.0.md``, ``examples/v0.1.0``) and that
record never changes. Between releases the code may diverge from it, a schema version or an
identity of the frozen example, only when ``CHANGELOG.md`` records the transition under
``## [Não lançado]`` as ``<name> <frozen value> → <code value>``.
"""

from __future__ import annotations


def unrecorded_changes(code: dict[str, str], frozen: dict[str, str], changelog: str) -> list[str]:
    """Every value that left the frozen record without its transition under ``[Não lançado]``.

    A transition is written ``<name> <frozen value> → <code value>``; backticks around the
    name or the values are allowed. Only the unreleased section counts: a transition belongs
    to the code that has not shipped yet.

    Args:
        code: The value the code produces today, per name.
        frozen: The value the release record holds, per name.
        changelog: The text of ``CHANGELOG.md``.

    Returns:
        The missing transitions, written as they are expected; empty when every divergent
        value is recorded.
    """
    unreleased = changelog.split("## [Não lançado]", 1)[1].split("\n## [", 1)[0]
    unreleased = unreleased.replace("`", "")
    transitions = (
        f"{name} {frozen[name]} → {value}" for name, value in code.items() if value != frozen[name]
    )
    return [item for item in transitions if item not in unreleased]
