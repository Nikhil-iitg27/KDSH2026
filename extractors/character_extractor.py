"""
CharacterExtractor for handling character extraction logic using structured outputs.
"""

from typing import List, Tuple, TYPE_CHECKING

from utils import TextChunker, PromptTemplates
from models import ExtractedCharacters

if TYPE_CHECKING:
    from core import QwenLLM


class CharacterExtractor:
    """Handles character extraction logic with structured outputs."""
        # Pronouns to filter out
    PRONOUNS = {'he', 'she', 'they', 'it', 'him', 'her', 'them', 'his', 'hers', 'their', 'theirs', 'i', 'you', 'we', 'us', 'me', 'my', 'mine', 'your', 'yours', 'our', 'ours'}
    def __init__(self, llm: 'QwenLLM', chunker: TextChunker):
        self.llm = llm
        self.chunker = chunker
    
    def extract_from_story(self, story_text: str) -> Tuple[List[str], List[str]]:
        """Extract character names from story using structured outputs. Returns (characters, chunks)."""
        # Chunk the story
        chunks = self.chunker.chunk_by_words(story_text)
        print(f"Created {len(chunks)} chunks")
        
        # Extract characters from all chunks using structured output
        all_characters = set()
        structured_llm = self.llm.with_structured_output(ExtractedCharacters)
        
        for i, chunk in enumerate(chunks):
            print(f"  Processing chunk {i+1}/{len(chunks)}")
            
            prompt = f"""Extract all character names from this text.
Character names are proper nouns referring to people.

Text:
{chunk}

Return a JSON object with a 'characters' field containing the list of character names."""
            
            try:
                result = structured_llm.invoke(prompt)
                for char in result.characters:
                    if char and char.strip():
                        char_clean = char.strip()
                        # Filter out pronouns
                        if char_clean.lower() not in self.PRONOUNS:
                            all_characters.add(char_clean)
                
                filtered_chars = [c for c in result.characters if c.strip().lower() not in self.PRONOUNS]
                print(f"    Found: {', '.join(filtered_chars) if filtered_chars else 'none'}")
            except Exception as e:
                print(f"    Error extracting from chunk {i+1}: {e}")
                continue
        
        characters = sorted(list(all_characters))
        print(f"\nFound {len(characters)} characters: {', '.join(characters)}")
        
        return characters, chunks
