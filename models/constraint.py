"""
Constraint model for character constraints.
"""

from typing import Literal
from pydantic import BaseModel, Field, field_validator


class Constraint(BaseModel):
    """Structured constraint with validation."""
    character: str = Field(description="Character who has this constraint")
    constraint_type: Literal["ability", "prohibition", "trait", "state"] = Field(
        description="Type of constraint"
    )
    value: str = Field(description="Description of the constraint", min_length=3)
    temporal_tag: Literal["past", "present", "future", "habitual"] = Field(
        description="When this constraint applies"
    )
    chunk_index: int = Field(description="Which chunk this was extracted from")
    source_chunk: str = Field(description="Original text chunk")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    
    @field_validator('value')
    @classmethod
    def value_not_empty(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("Constraint value cannot be empty")
        return v.strip()
