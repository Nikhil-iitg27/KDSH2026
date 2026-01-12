"""
Constraint extraction from story text using Local LLM.
Extracts character constraints and interactions for logical coherence verification.
"""

import re
import os
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Constraint:
    """Represents a constraint on a character."""
    character: str
    constraint_type: str  # e.g., 'ability', 'prohibition', 'trait', 'state'
    value: str
    temporal_tag: str = "present"  # 'past', 'present', 'future', 'habitual'
    chunk_index: int = 0  # Which chunk (T) this was extracted from
    source_chunk: str = ""
    confidence: float = 0.0


@dataclass
class Interaction:
    """Represents an interaction between characters."""
    character1: str
    character2: str
    interaction_type: str  # e.g., 'meets', 'fights', 'helps', 'speaks_to'
    description: str = ""
    temporal_tag: str = "present"  # 'past', 'present', 'future', 'habitual'
    chunk_index: int = 0  # Which chunk (T) this was extracted from
    is_bidirectional: bool = False  # True if this is a mutual interaction
    source_chunk: str = ""
    confidence: float = 0.0


class StoryChunker:
    """Chunks story text into manageable pieces for LLM processing."""
    
    def __init__(self, chunk_size: int = 20, overlap: int = 2):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """Chunk text by sentences with overlap."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Keep last few sentences for overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    s_len = len(s.split())
                    if overlap_length + s_len <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += s_len
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str) -> List[str]:
        """Chunk text by paragraphs."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para.split())
            
            if current_length + para_length > self.chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(para)
            current_length += para_length
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks


class LLMConstraintExtractor:
    """Extracts constraints and interactions using a local LLM."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        """
        Initialize the LLM-based extractor.
        
        Args:
            model_name: HuggingFace model name (default: Qwen2.5-7B-Instruct)
        """
        print(f"Loading model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        # Initialize embedding model for semantic deduplication
        print("Loading embedding model for deduplication...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, 384D
        self.dedup_threshold = 0.90  # High threshold for conservative deduplication
        
        self.chunker = StoryChunker(chunk_size=20, overlap=2)
        self.characters: List[str] = []
        self.seen_constraints: set = set()  # Track unique constraints
        self.seen_interactions: set = set()  # Track unique interactions
        self.constraints: List[Constraint] = []
        self.interactions: List[Interaction] = []
        self.story_text: str = ""  # Store original story for validation
    
    def generate_response(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate response from the LLM."""
        # Tokenize with larger max_length to avoid truncation
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Check if truncation occurred
        prompt_tokens = inputs['input_ids'].shape[1]
        if prompt_tokens >= 2048:
            print(f"  WARNING: Prompt truncated to {prompt_tokens} tokens")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
    
    def extract_characters_from_chunk(self, chunk: str) -> List[str]:
        """Extract character names from a chunk using LLM."""
        prompt = f"""Extract only the proper character names mentioned in this text. Do not add any names not present. List one name per line.

Text: {chunk}

Character names:"""
        
        response = self.generate_response(prompt, max_new_tokens=50)
        
        # Parse response for character names
        names = []
        for line in response.split('\n'):
            line = line.strip()
            # Remove numbering, bullets, markdown formatting
            line = re.sub(r'^[\d\-\*\.]+\s*', '', line)
            line = re.sub(r'\*\*', '', line)
            line = re.sub(r'[`:]', '', line)
            
            if line and len(line.split()) <= 3:  # Likely a name
                # Capitalize properly
                name = ' '.join(word.capitalize() for word in line.split())
                # Validate: name should appear in the chunk
                if name and name not in names and name in chunk:
                    names.append(name)
        
        return names
    
    def extract_constraints_from_chunk(self, chunk: str, characters: List[str], chunk_index: int) -> List[Constraint]:
        """Extract constraints from a chunk using LLM."""
        char_list = ', '.join(characters) if characters else "any characters"
        
        prompt = f"""Extract character constraints. A constraint belongs to the character WHO HAS IT. Distinguish backstory from main story:

Time tags:
- [past]: ONLY for backstory with markers like "was once", "used to be", "had been", "years ago"
- [present]: Main story events (even if written in past tense like "Alice met Bob")
- [future]: Plans or intentions ("will", "going to")
- [habitual]: Ongoing traits/abilities ("can", "is", "always")

CORRECT:
"Bob was once a police officer" → Bob | state | was once a police officer | past
"Alice is dedicated" → Alice | trait | is dedicated | habitual
"Alice met Bob" → (skip - this is an event, not a constraint)

WRONG:
"Carol trusts Alice" → Alice | trait | ... (Alice doesn't have this, Carol does)

Format: CHARACTER_NAME | TYPE | DESCRIPTION | TIME
Types: ability, prohibition, trait, state

Text: {chunk}

Output:"""
        
        response = self.generate_response(prompt, max_new_tokens=100)
        
        constraints = []
        for line in response.split('\n'):
            line = line.strip()
            
            # Clean markdown and formatting
            line = re.sub(r'^[\d\-\*\.]+\s*', '', line)
            line = re.sub(r'\*\*', '', line)
            
            if '|' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3:
                    char = parts[0]
                    c_type = parts[1]
                    desc = parts[2]
                    temporal = parts[3] if len(parts) >= 4 else "present"
                    
                    # Skip if any part is empty
                    if not char or not c_type or not desc:
                        continue
                    
                    # Validate: character name should be in our character list
                    if char not in self.characters:
                        continue
                    
                    # Skip if description is too short (likely incomplete)
                    if len(desc) < 3:
                        continue
                    
                    # Skip if description contains format instructions
                    if 'output should' in desc.lower() or 'adhere to' in desc.lower() or 'format' in desc.lower():
                        continue
                    
                    # Filter invalid descriptions (meta-commentary, placeholders, etc.)
                    invalid_patterns = [
                        '(none)', 'output should', 'based on', 'format:', 'example:', 
                        'extract', 'character', 'constraint', 'description', 'text:',
                        'critical:', 'important:', 'wrong', 'correct'
                    ]
                    if any(pattern in desc.lower() for pattern in invalid_patterns):
                        continue
                    
                    # Check for minimal completeness - avoid single-word fragments
                    # For traits/abilities, we expect descriptive phrases not just adjectives
                    if len(desc.split()) < 2 and c_type.lower() in ['trait', 'ability']:
                        # Single word traits/abilities are likely incomplete (e.g., "experienced" vs "experienced detective")
                        continue
                    
                    # Clean description of implicit/explicit references
                    desc = re.sub(r'\s*\(implied.*?\)', '', desc, flags=re.IGNORECASE)
                    desc = re.sub(r'\s*\(explicit.*?\)', '', desc, flags=re.IGNORECASE)
                    desc = desc.strip()
                    
                    # Normalize temporal tag
                    temporal = temporal.lower().strip()
                    if temporal not in ['past', 'present', 'future', 'habitual']:
                        temporal = 'present'  # Default
                    
                    # Validate constraint type
                    c_type_lower = c_type.lower()
                    if any(t in c_type_lower for t in ['ability', 'prohibition', 'trait', 'state']):
                        # Semantic deduplication - check if this is too similar to existing constraints
                        is_duplicate = False
                        for existing in constraints:
                            if (existing.character == char and 
                                existing.constraint_type == c_type_lower):
                                # Check if descriptions are semantically similar (one contains the other)
                                if desc.lower() in existing.value.lower() or existing.value.lower() in desc.lower():
                                    is_duplicate = True
                                    break
                        
                        if is_duplicate:
                            continue
                        
                        # Create unique key for deduplication (more specific)
                        constraint_key = f"{char.lower()}|{c_type_lower}|{desc.lower()}"
                        
                        if constraint_key not in self.seen_constraints:
                            self.seen_constraints.add(constraint_key)
                            constraints.append(Constraint(
                                character=char,
                                constraint_type=c_type_lower,
                                value=desc,
                                temporal_tag=temporal,
                                chunk_index=chunk_index,
                                source_chunk=chunk,
                                confidence=0.8
                            ))
        
        return constraints
    
    def extract_interactions_from_chunk(self, chunk: str, characters: List[str], chunk_index: int) -> List[Interaction]:
        """Extract interactions from a chunk using LLM."""
        char_list = ', '.join(characters) if characters else "any characters"
        
        prompt = f"""Extract character interactions between TWO DIFFERENT characters. ACTOR does action TO RECEIVER.

Time tags:
- [past]: ONLY backstory with markers like "was once", "years ago", "had met"
- [present]: Main story events (even if narrated in past tense)
- [future]: Planned actions ("will", "going to")

CORRECT:
"Bob told Alice about conspiracy" → Bob | Alice | speaks | told Alice about conspiracy | present
"Carol assigned the case" → Carol | Alice | assigns | assigned the case | present
"Bob was once a police officer" → (skip - not an interaction)
"She is dedicated" → (skip - trait description, not interaction)

WRONG:
"Bob warned Alice" → Alice | Bob | warns | ... (reverses direction)
"Alice is dedicated" → Alice | Alice | meets | ... (trait, not interaction; also self-loop)

Format: ACTOR | RECEIVER | TYPE | FULL_DESCRIPTION | TIME
Types: meets, fights, helps, speaks, warns, assigns, promises

Text: {chunk}

Output:"""
        
        response = self.generate_response(prompt, max_new_tokens=100)
        
        interactions = []
        for line in response.split('\n'):
            line = line.strip()
            
            # Clean markdown and formatting
            line = re.sub(r'^[\d\-\*\.]+\s*', '', line)
            line = re.sub(r'\*\*', '', line)
            
            if '|' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 4:
                    char1 = parts[0]
                    char2 = parts[1]
                    i_type = parts[2]
                    desc = parts[3]
                    temporal = parts[4] if len(parts) >= 5 else "present"
                    
                    # Remove common prefixes from character names
                    char1 = re.sub(r'^(output|format|interaction|name1|character1)\s*:?\s*', '', char1, flags=re.IGNORECASE).strip()
                    char2 = re.sub(r'^(output|format|interaction|name2|character2)\s*:?\s*', '', char2, flags=re.IGNORECASE).strip()
                    
                    # Skip if no valid character names or description
                    if not char1 or not char2 or not desc:
                        continue
                    
                    # Skip self-interactions (character interacting with themselves)
                    if char1 == char2:
                        continue
                    
                    # Validate: both character names should be in our character list
                    if char1 not in self.characters or char2 not in self.characters:
                        continue
                    
                    # Skip if description is too short
                    if len(desc) < 3:
                        continue
                    
                    # Skip format instructions
                    if 'short_description' in desc.lower() or 'description' == desc.lower():
                        continue
                    
                    # Filter invalid descriptions (meta-commentary, placeholders, etc.)
                    invalid_patterns = [
                        '(none)', 'output should', 'based on', 'format:', 'example:', 
                        'extract', 'character', 'interaction', 'description', 'text:',
                        'actor', 'receiver', 'critical:', 'important:'
                    ]
                    if any(pattern in desc.lower() for pattern in invalid_patterns):
                        continue
                    
                    # Check that description includes the action verb (basic completeness check)
                    # Common action verbs in interactions
                    action_verbs = ['warn', 'assign', 'promise', 'meet', 'fight', 'help', 'speak', 'tell', 'ask', 'trust', 'give']
                    # Check if at least one verb stem appears in the description
                    has_verb = any(verb in desc.lower() for verb in action_verbs)
                    if not has_verb and len(desc.split()) < 3:  # If very short and no verb, likely incomplete
                        continue
                    
                    # Clean description of implicit/explicit references
                    desc = re.sub(r'\s*\(implied.*?\)', '', desc, flags=re.IGNORECASE)
                    desc = re.sub(r'\s*\(explicit.*?\)', '', desc, flags=re.IGNORECASE)
                    desc = desc.strip()
                    
                    # Normalize temporal tag
                    temporal = temporal.lower().strip()
                    if temporal not in ['past', 'present', 'future']:
                        temporal = 'present'  # Default
                    
                    # Semantic deduplication - check if this is too similar to existing interactions
                    is_duplicate = False
                    for existing in interactions:
                        if (existing.character1 == char1 and 
                            existing.character2 == char2 and 
                            existing.interaction_type == i_type.lower()):
                            # Check if descriptions are semantically similar (one contains the other)
                            if desc.lower() in existing.description.lower() or existing.description.lower() in desc.lower():
                                is_duplicate = True
                                break
                    
                    if is_duplicate:
                        continue
                    
                    # Create unique key for deduplication (more specific)
                    interaction_key = f"{char1.lower()}|{char2.lower()}|{i_type.lower()}|{desc.lower()}"
                    
                    if interaction_key not in self.seen_interactions:
                        self.seen_interactions.add(interaction_key)
                        interactions.append(Interaction(
                            character1=char1,
                            character2=char2,
                            interaction_type=i_type.lower(),
                            description=desc,
                            temporal_tag=temporal,
                            chunk_index=chunk_index,
                            source_chunk=chunk,
                            confidence=0.8
                        ))
        
        return interactions
    
    def extract_from_story(self, story: str) -> Dict[str, Any]:
        """Main extraction method using LLM."""
        self.story_text = story  # Store for validation
        
        print("Chunking story...")
        chunks = self.chunker.chunk_by_sentences(story)
        print(f"Created {len(chunks)} chunks")
        
        # Extract characters from all chunks
        print("\nExtracting characters...")
        all_characters = set()
        for i, chunk in enumerate(chunks):
            print(f"  Processing chunk {i+1}/{len(chunks)}")
            chars = self.extract_characters_from_chunk(chunk)
            all_characters.update(chars)
        
        # Validate characters: ensure they actually appear in the story
        validated_characters = [c for c in all_characters if c in story]
        self.characters = sorted(list(validated_characters))
        print(f"Found {len(self.characters)} characters: {', '.join(self.characters)}")
        
        # Extract constraints from each chunk
        print("\nExtracting constraints...")
        for i, chunk in enumerate(chunks):
            print(f"  Processing chunk {i+1}/{len(chunks)}")
            constraints = self.extract_constraints_from_chunk(chunk, self.characters, chunk_index=i)
            self.constraints.extend(constraints)
        
        print(f"Found {len(self.constraints)} constraints")
        
        # Extract interactions from each chunk
        print("\nExtracting interactions...")
        for i, chunk in enumerate(chunks):
            print(f"  Processing chunk {i+1}/{len(chunks)}")
            interactions = self.extract_interactions_from_chunk(chunk, self.characters, chunk_index=i)
            self.interactions.extend(interactions)
        
        print(f"Found {len(self.interactions)} interactions")
        
        # Apply embedding-based deduplication
        self.deduplicate_constraints_by_embedding()
        self.deduplicate_interactions_by_embedding()
        
        # Consolidate bidirectional interactions
        self.consolidate_bidirectional_interactions()
        
        return {
            'characters': self.characters,
            'constraints': self.constraints,
            'interactions': self.interactions
        }
    
    def get_character_constraints(self, character: str) -> List[Constraint]:
        """Get all constraints for a specific character."""
        return [c for c in self.constraints if character.lower() in c.character.lower()]
    
    def get_character_interactions(self, character: str) -> List[Interaction]:
        """Get all interactions involving a specific character."""
        return [i for i in self.interactions 
                if character.lower() in i.character1.lower() or 
                   character.lower() in i.character2.lower()]
    
    def deduplicate_constraints_by_embedding(self) -> None:
        """Remove semantically similar constraints using embedding similarity."""
        if not self.constraints:
            return
        
        print(f"\nDeduplicating {len(self.constraints)} constraints using embeddings (threshold={self.dedup_threshold})...")
        
        # Create text representations for embedding
        texts = [f"{c.character}: {c.constraint_type}: {c.value}" for c in self.constraints]
        
        # Compute embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        
        # Compute pairwise cosine similarity
        similarities = cosine_similarity(embeddings)
        
        # Mark items to keep (avoid duplicates)
        keep_indices = []
        for i in range(len(self.constraints)):
            # Check if this constraint is too similar to any already kept constraint
            is_duplicate = False
            for j in keep_indices:
                if similarities[i, j] >= self.dedup_threshold:
                    is_duplicate = True
                    print(f"  Removing duplicate: '{texts[i]}' (similar to '{texts[j]}', similarity={similarities[i,j]:.3f})")
                    break
            
            if not is_duplicate:
                keep_indices.append(i)
        
        # Update constraints list
        original_count = len(self.constraints)
        self.constraints = [self.constraints[i] for i in keep_indices]
        print(f"Reduced from {original_count} to {len(self.constraints)} constraints")
    
    def deduplicate_interactions_by_embedding(self) -> None:
        """Remove semantically similar interactions using embedding similarity."""
        if not self.interactions:
            return
        
        print(f"\nDeduplicating {len(self.interactions)} interactions using embeddings (threshold={self.dedup_threshold})...")
        
        # Create text representations for embedding
        texts = [f"{i.character1} {i.interaction_type} {i.character2}: {i.description}" 
                for i in self.interactions]
        
        # Compute embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        
        # Compute pairwise cosine similarity
        similarities = cosine_similarity(embeddings)
        
        # Mark items to keep (avoid duplicates)
        keep_indices = []
        for i in range(len(self.interactions)):
            # Check if this interaction is too similar to any already kept interaction
            is_duplicate = False
            for j in keep_indices:
                if similarities[i, j] >= self.dedup_threshold:
                    is_duplicate = True
                    print(f"  Removing duplicate: '{texts[i][:80]}...' (similar to '{texts[j][:80]}...', similarity={similarities[i,j]:.3f})")
                    break
            
            if not is_duplicate:
                keep_indices.append(i)
        
        # Update interactions list
        original_count = len(self.interactions)
        self.interactions = [self.interactions[i] for i in keep_indices]
        print(f"Reduced from {original_count} to {len(self.interactions)} interactions")
    
    def consolidate_bidirectional_interactions(self) -> None:
        """Consolidate bidirectional interactions (e.g., Alice meets Bob + Bob meets Alice)."""
        if not self.interactions:
            return
        
        print(f"\nConsolidating bidirectional interactions...")
        
        # Types that are inherently bidirectional
        bidirectional_types = {'meets', 'fights', 'speaks'}
        
        # Track which interactions to keep
        to_remove = set()
        
        for i, interaction in enumerate(self.interactions):
            if i in to_remove:
                continue
            
            # Only check bidirectional types
            if interaction.interaction_type not in bidirectional_types:
                continue
            
            # Look for reverse interaction
            for j in range(i + 1, len(self.interactions)):
                if j in to_remove:
                    continue
                
                other = self.interactions[j]
                
                # Check if this is the reverse (Alice->Bob vs Bob->Alice)
                if (interaction.character1 == other.character2 and
                    interaction.character2 == other.character1 and
                    interaction.interaction_type == other.interaction_type):
                    
                    # Mark the one from the later chunk as bidirectional, remove the earlier
                    if interaction.chunk_index <= other.chunk_index:
                        interaction.is_bidirectional = True
                        to_remove.add(j)
                        print(f"  Merged: {interaction.character1}<->{interaction.character2} ({interaction.interaction_type})")
                    else:
                        other.is_bidirectional = True
                        to_remove.add(i)
                        print(f"  Merged: {other.character1}<->{other.character2} ({other.interaction_type})")
                        break
        
        # Remove marked interactions
        if to_remove:
            original_count = len(self.interactions)
            self.interactions = [inter for idx, inter in enumerate(self.interactions) if idx not in to_remove]
            print(f"Reduced from {original_count} to {len(self.interactions)} interactions after consolidation")


def write_extraction_results(extractor: LLMConstraintExtractor, output_file: str) -> None:
    """Write extraction results to a JSON file with proper hierarchy, separating backstory from current story."""
    
    # Build hierarchical structure
    output_data = {
        "summary": {
            "total_characters": len(extractor.characters),
            "total_constraints": len(extractor.constraints),
            "total_interactions": len(extractor.interactions),
            "total_chunks": max([c.chunk_index for c in extractor.constraints] + [i.chunk_index for i in extractor.interactions], default=0) + 1
        },
        "characters": []
    }
    
    # Organize data by character
    for char in extractor.characters:
        char_data = {
            "name": char,
            "backstory": {
                "constraints": [],
                "events": []
            },
            "current_story": {
                "constraints": [],
                "interactions": {
                    "outgoing": [],
                    "incoming": [],
                    "bidirectional": []
                }
            }
        }
        
        # Separate constraints by temporal tag
        constraints = extractor.get_character_constraints(char)
        for c in constraints:
            constraint_data = {
                "type": c.constraint_type,
                "value": c.value,
                "temporal_tag": c.temporal_tag,
                "chunk_index": c.chunk_index,
                "source_chunk": c.source_chunk,
                "confidence": c.confidence
            }
            
            # Backstory: past or habitual traits
            if c.temporal_tag in ['past', 'habitual']:
                char_data["backstory"]["constraints"].append(constraint_data)
            else:
                char_data["current_story"]["constraints"].append(constraint_data)
        
        # Separate interactions by temporal tag and direction
        for interaction in extractor.interactions:
            if interaction.character1 != char and interaction.character2 != char:
                continue
            
            interaction_data = {
                "type": interaction.interaction_type,
                "description": interaction.description,
                "temporal_tag": interaction.temporal_tag,
                "chunk_index": interaction.chunk_index,
                "is_bidirectional": interaction.is_bidirectional,
                "source_chunk": interaction.source_chunk,
                "confidence": interaction.confidence
            }
            
            # Add to backstory if past event
            if interaction.temporal_tag == 'past':
                if interaction.character1 == char:
                    interaction_data["other_character"] = interaction.character2
                    interaction_data["direction"] = "outgoing"
                else:
                    interaction_data["other_character"] = interaction.character1
                    interaction_data["direction"] = "incoming"
                char_data["backstory"]["events"].append(interaction_data)
            
            # Add to current story
            elif interaction.is_bidirectional and interaction.character1 == char:
                # Only add once for bidirectional
                interaction_data["other_character"] = interaction.character2
                char_data["current_story"]["interactions"]["bidirectional"].append(interaction_data)
            elif not interaction.is_bidirectional:
                # Add as outgoing or incoming
                if interaction.character1 == char:
                    interaction_data["target"] = interaction.character2
                    char_data["current_story"]["interactions"]["outgoing"].append(interaction_data)
                elif interaction.character2 == char:
                    interaction_data["source"] = interaction.character1
                    char_data["current_story"]["interactions"]["incoming"].append(interaction_data)
        
        output_data["characters"].append(char_data)
    
    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Test story
    story = """
    Alice is a skilled detective who can solve complex cases. She has worked in the city 
    for ten years. Alice cannot ignore injustice and must investigate every lead. She is 
    dedicated and thorough.
    
    Bob is a mysterious informant who helps Alice with information. He can access secret 
    records but cannot reveal his identity. Bob was once a police officer but left under 
    mysterious circumstances.
    
    One day, Alice met Bob in a dark alley. Bob told Alice about a conspiracy. Alice said 
    she would investigate immediately. Bob warned Alice to be careful, as powerful people 
    were involved.
    
    Carol is the police chief who trusts Alice completely. Carol assigned the case to Alice 
    and told her to follow all leads. Alice promised Carol she would solve it.
    """
    
    print("STORY:")
    print("-" * 80)
    print(story)
    print("-" * 80)
    
    # Initialize LLM-based extractor
    print("\nInitializing LLM Constraint Extractor...")
    extractor = LLMConstraintExtractor(model_name="Qwen/Qwen2.5-7B-Instruct")
    
    # Extract constraints and interactions
    print("\nProcessing story with LLM...")
    results = extractor.extract_from_story(story)
    
    # Write results to file
    output_path = os.path.join(os.path.dirname(__file__), "constraint.json")
    print(f"\nWriting results to {output_path}...")
    write_extraction_results(extractor, output_path)
    
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_path}")
    print(f"Total characters: {len(results['characters'])}")
    print(f"Total constraints: {len(results['constraints'])}")
    print(f"Total interactions: {len(results['interactions'])}")
