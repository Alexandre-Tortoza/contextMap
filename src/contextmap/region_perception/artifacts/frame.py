"""Auditable outcome and image preparation metadata for one selected observation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from contextmap.visual_perception import PreparedImage


@dataclass(frozen=True, kw_only=True)
class RegionPerceptionFrame:
    """One physical frame outcome, including successful empty regional evidence."""

    image: PreparedImage
    outcome: Literal["succeeded", "failed"] = "succeeded"
    failure_detail: str | None = None

    def __post_init__(self) -> None:
        """Require explicit failure detail only for failed frames."""
        if self.outcome not in ("succeeded", "failed"):
            raise ValueError("regional frame outcome must be succeeded or failed")
        if self.outcome == "failed" and not self.failure_detail:
            raise ValueError("failed regional frame requires failure_detail")
        if self.outcome == "succeeded" and self.failure_detail is not None:
            raise ValueError("successful regional frame cannot have failure_detail")
