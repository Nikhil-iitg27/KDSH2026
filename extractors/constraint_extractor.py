"""
ConstraintExtractor for handling constraint extraction logic using structured outputs.
"""

from typing import List, TYPE_CHECKING

from models import Constraint, ExtractedConstraints
from utils import ConstraintFilter

if TYPE_CHECKING:
    from core import QwenLLM


class ConstraintExtractor:
    """Handles constraint extraction logic with structured LLM outputs."""
    
    def __init__(self, llm: 'QwenLLM', filter_: ConstraintFilter, name_mapping: dict = None):
        self.llm = llm
        self.filter = filter_
        self.name_mapping = name_mapping or {}
    
    def extract_from_chunks(self, chunks: List[str], characters: List[str]) -> List[Constraint]:
        """Extract constraints from all chunks using structured outputs."""
        constraints = []
        
        for i, chunk in enumerate(chunks):
            print(f"  Processing chunk {i+1}/{len(chunks)}")
            
            chunk_constraints = self._extract_from_chunk(chunk, characters, i)
            constraints.extend(chunk_constraints)
        
        print(f"Extracted {len(constraints)} constraints")
        return constraints
    
    def _extract_from_chunk(self, chunk: str, characters: List[str], chunk_index: int) -> List[Constraint]:
        """Extract constraints from a single chunk using structured output."""
        prompt = f"""Extract character constraints from this text. A constraint belongs to the character WHO HAS IT.

Time tags - CRITICAL FOR BACKSTORY SEPARATION:
- "past": Backstory events, completed states, historical facts. USE FOR:
  * "was once", "used to be", "had been", "years ago"
  * Prohibitions/rules from backstory: "must never", "cannot ever", "prohibited from"
  * Past relationships: "mentored", "worked together", "was partners with"
- "present": Current story events happening now (even if narrated in past tense like "Bob walked")
- "future": Plans or intentions ("will", "going to")
- "habitual": Ongoing traits/abilities that apply generally ("is", "can", "always")

Types:
- ability: Can do something ("can fly", "able to investigate")
- prohibition: Explicit rule/restriction ("must never", "cannot", "prohibited from", "not allowed to")
- trait: Personality or characteristic ("honest", "dedicated", "corrupt")
- state: Current/past condition or action ("is a detective", "was a police officer", "refused to investigate")

CRITICAL - prohibition vs state:
- "Alice REFUSED to investigate" → state (past action/choice)
- "Alice CANNOT investigate" → prohibition (explicit rule)
- "Alice MUST NEVER return" → prohibition (binding restriction)

IMPORTANT: 
- Skip interactions between characters (like "told", "met", "warned")
- Only extract properties/states that belong to one character
- Character names must match exactly from: {', '.join(characters[:5])}{'...' if len(characters) > 5 else ''}
- Backstory prohibitions like "must never X" should have temporal_tag="past"

Text:
{chunk}

Respond with ONLY valid JSON starting with {{ and ending with }}.
Root object must have a "constraints" field containing an array.
Each constraint object must have:
- character: exact character name (string)
- constraint_type: one of ability, prohibition, trait, state (string)
- value: description of the constraint (string)
- temporal_tag: one of past, present, future, habitual (string)
- confidence: numeric value between 0 and 1, like 0.9 (NOT "high" or "low")
No explanatory text before or after the JSON.

JSON:"""
        
        try:
            structured_llm = self.llm.with_structured_output(ExtractedConstraints)
            result = structured_llm.invoke(prompt)
            
            constraints = []
            for extracted in result.constraints:
                # Canonicalize character name
                char_name = self.name_mapping.get(extracted.character, extracted.character)
                
                # Validate character
                if char_name not in characters:
                    continue
                
                # Validate description
                if not self.filter.is_valid_description(extracted.value):
                    continue
                
                # Skip if looks like interaction
                if self.filter.is_interaction_not_constraint(extracted.value, char_name, characters):
                    continue
                
                # Normalize type
                c_type_final = self.filter.normalize_constraint_type(extracted.constraint_type, extracted.value)
                if not c_type_final:
                    continue
                
                # Normalize temporal - use temporal_tag if available, otherwise infer
                temporal = getattr(extracted, 'temporal_tag', 'present')
                temporal = self.filter.normalize_temporal_tag(temporal, extracted.value)
                
                # Create final Constraint model with all fields
                try:
                    constraint = Constraint(
                        character=char_name,  # Use canonical name
                        constraint_type=c_type_final,
                        value=extracted.value,
                        temporal_tag=temporal,
                        chunk_index=chunk_index,
                        source_chunk=chunk,
                        confidence=extracted.confidence
                    )
                    constraints.append(constraint)
                    print(f"    ✓ {char_name}: {extracted.value[:50]}...")
                except Exception as e:
                    print(f"    ✗ Failed to create constraint: {e}")
                    continue
            
            return constraints
            
        except Exception as e:
            print(f"    ✗ Structured extraction error: {e}")
            return []
