"""
ExtractionState model for maintaining state across the LangGraph workflow.
"""

from typing import Any, List, Dict
from pydantic import BaseModel, ConfigDict

from models.constraint import Constraint
from models.interaction import Interaction


class ExtractionState(BaseModel):
    """State maintained across the graph."""
    story_text: str = ""
    chunks: List[str] = []
    current_chunk_index: int = 0
    
    # Extracted data
    characters: List[str] = []
    constraints: List[Constraint] = []
    interactions: List[Interaction] = []
    
    # Character name mapping (variant -> canonical)
    name_mapping: Dict[str, str] = {}
    
    # Validation flags
    validation_errors: List[str] = []
    needs_correction: bool = False
    
    # Models (not serialized)
    llm_model: Any = None
    tokenizer: Any = None
    embedding_model: Any = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
