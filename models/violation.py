"""
Violation model for story validation.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class Violation(BaseModel):
    """Represents a constraint violation."""
    violation_type: Literal["constraint_violation", "state_contradiction", "temporal_inconsistency"] = Field(
        description="Type of violation detected"
    )
    character: str = Field(description="Character involved in the violation")
    severity: int = Field(description="Severity level from 1 (minor) to 10 (critical)", ge=1, le=10)
    baseline_constraint: Optional[str] = Field(default=None, description="The constraint from baseline being violated")
    proposed_event: str = Field(description="The event/constraint from proposed story causing violation")
    explanation: str = Field(description="Human-readable explanation of the violation")
