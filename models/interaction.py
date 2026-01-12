"""
Interaction model for character-to-character interactions.
"""

from typing import Literal
from pydantic import BaseModel, Field, field_validator


class Interaction(BaseModel):
    """Structured interaction with validation."""
    character1: str = Field(description="Character who performs the action (actor)")
    character2: str = Field(description="Character who receives the action (receiver)")
    interaction_type: Literal["meets", "fights", "helps", "speaks", "warns", "assigns", "promises", "trusts"] = Field(
        description="Type of interaction"
    )
    description: str = Field(description="Full description of the interaction", min_length=3)
    temporal_tag: Literal["past", "present", "future"] = Field(
        description="When this interaction occurred"
    )
    chunk_index: int = Field(description="Which chunk this was extracted from")
    is_bidirectional: bool = Field(default=False, description="Whether this is a mutual interaction")
    source_chunk: str = Field(description="Original text chunk")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    
    @field_validator('character2')
    @classmethod
    def characters_not_same(cls, v: str, info) -> str:
        if info.data.get('character1') and v == info.data['character1']:
            raise ValueError("Cannot have self-interaction")
        return v
