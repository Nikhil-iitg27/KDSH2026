"""
Pydantic schemas for structured LLM extraction outputs.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ExtractedCharacters(BaseModel):
    """Schema for character extraction."""
    characters: List[str] = Field(description="List of character names found in the text")


class ExtractedConstraint(BaseModel):
    """Schema for a single constraint extraction."""
    character: str = Field(description="The character this constraint applies to")
    constraint_type: str = Field(description="Type of constraint: state, prohibition, obligation, permission, trait, ability")
    value: str = Field(description="Description of the constraint")
    temporal_tag: Optional[str] = Field(default="present", description="Time tag: past, present, future, habitual")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0, description="Confidence score")


class ExtractedConstraints(BaseModel):
    """Schema for constraint extraction."""
    constraints: List[ExtractedConstraint] = Field(description="List of extracted constraints")


class ExtractedInteraction(BaseModel):
    """Schema for a single interaction extraction."""
    character1: str = Field(description="First character in the interaction (actor)")
    character2: str = Field(description="Second character in the interaction (receiver)")
    interaction_type: str = Field(description="Type: dialogue, physical, emotional, transactional, informational, meets, fights, helps, speaks, warns, assigns, promises, trusts")
    description: str = Field(description="Description of the interaction")
    temporal_tag: Optional[str] = Field(default="present", description="Time tag: past, present, future")
    is_bidirectional: bool = Field(default=False, description="Whether the interaction goes both ways")


class ExtractedInteractions(BaseModel):
    """Schema for interaction extraction."""
    interactions: List[ExtractedInteraction] = Field(description="List of extracted interactions")


class ValidationResult(BaseModel):
    """Schema for constraint validation."""
    is_valid: bool = Field(description="Whether the constraint is valid")
    issues: List[str] = Field(default_factory=list, description="List of validation issues found")


class ViolationCheck(BaseModel):
    """Schema for story validation violation check."""
    has_violation: bool = Field(description="Whether a violation was detected")
    character: Optional[str] = Field(default=None, description="Character involved in violation")
    severity: Optional[int] = Field(default=None, ge=1, le=10, description="Severity level from 1 (minor) to 10 (critical)")
    explanation: Optional[str] = Field(default=None, description="Explanation of the violation")
    baseline_constraint: Optional[str] = Field(default=None, description="The backstory constraint that was violated")
