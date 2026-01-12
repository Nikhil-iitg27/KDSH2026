"""
InteractionExtractor for handling interaction extraction logic using structured outputs.
"""

from typing import List, TYPE_CHECKING

from models import Interaction, ExtractedInteractions
from utils import InteractionFilter

if TYPE_CHECKING:
    from core import QwenLLM


class InteractionExtractor:
    """Handles interaction extraction logic with structured LLM outputs."""
    
    def __init__(self, llm: 'QwenLLM', filter_: InteractionFilter, name_mapping: dict = None):
        self.llm = llm
        self.filter = filter_
        self.name_mapping = name_mapping or {}
    
    def extract_from_chunks(self, chunks: List[str], characters: List[str]) -> List[Interaction]:
        """Extract interactions from all chunks using structured outputs."""
        interactions = []
        
        for i, chunk in enumerate(chunks):
            print(f"  Processing chunk {i+1}/{len(chunks)}")
            
            chunk_interactions = self._extract_from_chunk(chunk, characters, i)
            interactions.extend(chunk_interactions)
        
        print(f"Extracted {len(interactions)} interactions")
        return interactions
    
    def _extract_from_chunk(self, chunk: str, characters: List[str], chunk_index: int) -> List[Interaction]:
        """Extract interactions from a single chunk using structured output."""
        prompt = f"""Extract character interactions between TWO DIFFERENT characters. ACTOR does action TO RECEIVER.

Time tags:
- "past": ONLY backstory with markers like "was once", "years ago", "had met"
- "present": Main story events (even if narrated in past tense)
- "future": Planned actions ("will", "going to")

Types: meets, fights, helps, speaks, warns, assigns, promises, trusts

IMPORTANT:
- Actor comes first, receiver second
- No self-interactions (character1 != character2)
- Skip trait descriptions like "is dedicated"
- Character names must match exactly from: {', '.join(characters[:5])}{'...' if len(characters) > 5 else ''}

Text:
{chunk}

Respond with ONLY valid JSON starting with {{ and ending with }}.
Root object must have an "interactions" field containing an array.
Each interaction object must have:
- character1: actor name (string)
- character2: receiver name (string)
- interaction_type: one of meets, fights, helps, speaks, warns, assigns, promises, trusts (string)
- description: full description (string)
- temporal_tag: one of past, present, future (string)
- is_bidirectional: true or false (boolean, not string)
No explanatory text before or after the JSON.

JSON:"""
        
        try:
            structured_llm = self.llm.with_structured_output(ExtractedInteractions)
            result = structured_llm.invoke(prompt)
            
            interactions = []
            for extracted in result.interactions:
                # Clean and canonicalize character names
                char1 = self.filter.clean_character_name(extracted.character1)
                char2 = self.filter.clean_character_name(extracted.character2)
                
                # Map to canonical names
                char1 = self.name_mapping.get(char1, char1)
                char2 = self.name_mapping.get(char2, char2)
                
                # Validate characters
                if char1 not in characters or char2 not in characters:
                    continue
                
                if not char1 or not char2 or not extracted.description:
                    continue
                
                # Skip self-interactions
                if char1 == char2:
                    continue
                
                # Validate description
                if not self.filter.is_valid_description(extracted.description):
                    continue
                
                # Normalize type
                i_type_final = self.filter.normalize_interaction_type(extracted.interaction_type)
                if not i_type_final:
                    continue
                
                # Normalize temporal - use temporal_tag if available
                temporal = getattr(extracted, 'temporal_tag', 'present')
                temporal = self.filter.normalize_temporal_tag(temporal)
                
                # Create final Interaction model
                try:
                    interaction = Interaction(
                        character1=char1,
                        character2=char2,
                        interaction_type=i_type_final,
                        description=extracted.description,
                        temporal_tag=temporal,
                        chunk_index=chunk_index,
                        source_chunk=chunk,
                        confidence=0.8,
                        is_bidirectional=extracted.is_bidirectional
                    )
                    interactions.append(interaction)
                    print(f"    ✓ {char1} → {char2}: {extracted.description[:40]}...")
                except Exception as e:
                    print(f"    ✗ Failed to create interaction: {e}")
                    continue
            
            return interactions
            
        except Exception as e:
            print(f"    ✗ Structured extraction error: {e}")
            return []
