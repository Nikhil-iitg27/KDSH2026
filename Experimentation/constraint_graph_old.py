"""
LangGraph-based constraint extraction with structured outputs and multi-agent validation.
"""

import os
import json
from typing import List, Dict, Any, Optional, Literal, Annotated
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
import re


# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ============================================================================

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
    
    @validator('value')
    def value_not_empty(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Constraint value cannot be empty")
        return v.strip()


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
    
    @validator('character1', 'character2')
    def characters_not_same(cls, v, values):
        if 'character1' in values and v == values['character1']:
            raise ValueError("Cannot have self-interaction")
        return v


# ============================================================================
# GRAPH STATE
# ============================================================================

class ExtractionState(BaseModel):
    """State maintained across the graph."""
    story_text: str = ""
    chunks: List[str] = []
    current_chunk_index: int = 0
    
    # Extracted data
    characters: List[str] = []
    constraints: List[Constraint] = []
    interactions: List[Interaction] = []
    
    # Validation flags
    validation_errors: List[str] = []
    needs_correction: bool = False
    
    # Models (not serialized)
    llm_model: Any = None
    tokenizer: Any = None
    embedding_model: Any = None
    
    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# LLM WRAPPER FOR QWEN
# ============================================================================

class QwenLLM:
    """Wrapper for Qwen model to work with LangGraph."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
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
    
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate response from prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
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
        # Remove the prompt from response
        response = response[len(prompt):].strip()
        return response


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def chunk_story(text: str, chunk_size: int = 20, overlap: int = 2) -> List[str]:
    """Chunk text by sentences with overlap."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            
            # Keep overlap
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk):
                s_len = len(s.split())
                if overlap_length + s_len <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += s_len
                else:
                    break
            
            current_chunk = overlap_sentences
            current_length = overlap_length
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


# ============================================================================
# NODES: EXTRACTION
# ============================================================================

def extract_characters_node(state: ExtractionState) -> ExtractionState:
    """Node 1: Extract character names from all chunks."""
    print("\n" + "="*80)
    print("NODE 1: EXTRACTING CHARACTERS")
    print("="*80)
    
    # Chunk the story
    state.chunks = chunk_story(state.story_text)
    print(f"Created {len(state.chunks)} chunks")
    
    # Extract characters from all chunks
    all_characters = set()
    
    for i, chunk in enumerate(state.chunks):
        print(f"  Processing chunk {i+1}/{len(state.chunks)}")
        
        prompt = f"""Extract ONLY the proper character names mentioned in this text. List one name per line. Do not add names not present.

Text: {chunk}

Character names:"""
        
        response = state.llm_model.generate(prompt, max_new_tokens=50)
        
        # Parse character names
        for line in response.split('\n'):
            line = line.strip()
            # Remove numbering, bullets, markdown
            line = re.sub(r'^[\d\-\*\.]+\s*', '', line)
            line = re.sub(r'\*\*', '', line)
            line = re.sub(r'[`:]', '', line)
            
            if line and len(line.split()) <= 3:  # Likely a name
                name = ' '.join(word.capitalize() for word in line.split())
                # Validate: name should appear in story
                if name and name in state.story_text:
                    all_characters.add(name)
    
    # Update state
    state.characters = sorted(list(all_characters))
    print(f"\nFound {len(state.characters)} characters: {', '.join(state.characters)}")
    
    return state


def extract_constraints_node(state: ExtractionState) -> ExtractionState:
    """Node 2: Extract constraints from each chunk with structured output."""
    print("\n" + "="*80)
    print("NODE 2: EXTRACTING CONSTRAINTS")
    print("="*80)
    
    for i, chunk in enumerate(state.chunks):
        print(f"  Processing chunk {i+1}/{len(state.chunks)}")
        
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
        
        response = state.llm_model.generate(prompt, max_new_tokens=100)
        
        # Parse response
        for line in response.split('\n'):
            line = line.strip()
            
            # Clean markdown
            line = re.sub(r'^[\d\-\*\.]+\s*', '', line)
            line = re.sub(r'\*\*', '', line)
            
            if '|' not in line:
                continue
            
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 3:
                continue
            
            char = parts[0]
            c_type = parts[1]
            desc = parts[2]
            temporal = parts[3] if len(parts) >= 4 else "present"
            
            # Validate character exists
            if char not in state.characters:
                continue
            
            # Skip if too short
            if len(desc) < 3:
                continue
            
            # Filter invalid patterns
            invalid_patterns = [
                '(none)', 'output should', 'based on', 'format:', 'example:',
                'extract', 'character', 'constraint', 'description', 'text:',
                'critical:', 'important:', 'wrong', 'correct'
            ]
            if any(pattern in desc.lower() for pattern in invalid_patterns):
                continue
            
            # Skip if this looks like an interaction (verbs with direct objects to other characters)
            interaction_verbs = ['told', 'warned', 'met', 'assigned', 'promised', 'said', 'spoke', 'helped']
            other_char_names = [c for c in state.characters if c != char]
            # Check if description contains action verb + another character name
            if any(verb in desc.lower() for verb in interaction_verbs):
                if any(other_name in desc for other_name in other_char_names):
                    continue  # This is an interaction, not a constraint
            
            # Skip single-word traits (likely incomplete)
            if len(desc.split()) < 2 and c_type_lower in ['trait', 'ability']:
                continue
            
            # Normalize temporal tag
            temporal = temporal.lower().strip()
            if temporal not in ['past', 'present', 'future', 'habitual']:
                temporal = 'present'
            
            # Normalize constraint type
            c_type_lower = c_type.lower()
            if not any(t in c_type_lower for t in ['ability', 'prohibition', 'trait', 'state']):
                continue
            
            # Map to valid type
            if 'ability' in c_type_lower:
                c_type_final = 'ability'
            elif 'prohibition' in c_type_lower:
                c_type_final = 'prohibition'
            elif 'trait' in c_type_lower:
                c_type_final = 'trait'
            elif 'state' in c_type_lower:
                c_type_final = 'state'
            else:
                continue
            
            # Create Pydantic model (validates automatically)
            try:
                constraint = Constraint(
                    character=char,
                    constraint_type=c_type_final,
                    value=desc,
                    temporal_tag=temporal,
                    chunk_index=i,
                    source_chunk=chunk,
                    confidence=0.8
                )
                state.constraints.append(constraint)
            except Exception as e:
                print(f"    Validation failed: {e}")
                continue
    
    print(f"Extracted {len(state.constraints)} constraints")
    return state


def validate_constraints_node(state: ExtractionState) -> ExtractionState:
    """Node 3: Validate extracted constraints for logical consistency."""
    print("\n" + "="*80)
    print("NODE 3: VALIDATING CONSTRAINTS")
    print("="*80)
    
    validated_constraints = []
    state.validation_errors = []
    
    seen_keys = set()  # For deduplication
    
    for constraint in state.constraints:
        errors = []
        
        # Check 1: Character exists
        if constraint.character not in state.characters:
            errors.append(f"Character '{constraint.character}' not in character list")
        
        # Check 2: Temporal logic - "will" should be future, "was" should be past
        desc_lower = constraint.value.lower()
        if 'will' in desc_lower and constraint.temporal_tag not in ['future', 'present']:
            errors.append(f"Contains 'will' but marked as {constraint.temporal_tag}")
            # Auto-correct
            constraint.temporal_tag = 'future'
        
        if any(marker in desc_lower for marker in ['was once', 'used to be', 'had been']) and constraint.temporal_tag != 'past':
            errors.append(f"Contains past marker but marked as {constraint.temporal_tag}")
            # Auto-correct
            constraint.temporal_tag = 'past'
        
        # Check 3: Ability/trait consistency - ongoing things should be habitual
        if constraint.constraint_type in ['ability', 'trait'] and 'can' in desc_lower:
            if constraint.temporal_tag not in ['habitual', 'present']:
                constraint.temporal_tag = 'habitual'
        
        # Check 4: Look for misclassified interactions (actions between characters)
        interaction_indicators = ['told', 'warned', 'assigned', 'met', 'said', 'promised']
        other_chars = [c for c in state.characters if c != constraint.character]
        if any(verb in desc_lower for verb in interaction_indicators):
            if any(char_name in constraint.value for char_name in other_chars):
                errors.append(f"Looks like an interaction, not a constraint: '{constraint.value}'")
                continue  # Skip this constraint
        
        # Check 5: Deduplication by semantic key
        constraint_key = f"{constraint.character.lower()}|{constraint.constraint_type}|{constraint.value.lower()}"
        if constraint_key in seen_keys:
            errors.append(f"Duplicate constraint")
            continue
        seen_keys.add(constraint_key)
        
        # Check 6: Validate description quality
        if len(constraint.value.split()) < 2 and constraint.constraint_type in ['trait', 'ability']:
            errors.append(f"Description too short: '{constraint.value}'")
            continue
        
        # Log validation errors
        if errors:
            state.validation_errors.extend([f"{constraint.character}: {err}" for err in errors])
            # Only skip if critical error (not auto-correctable)
            if any('interaction' in err.lower() or 'duplicate' in err.lower() or 'too short' in err.lower() for err in errors):
                print(f"  ✗ Rejected: {constraint.character} - {constraint.value} ({', '.join(errors)})")
                continue
        
        validated_constraints.append(constraint)
    
    # Update state
    original_count = len(state.constraints)
    state.constraints = validated_constraints
    
    print(f"\nValidation complete:")
    print(f"  Original: {original_count} constraints")
    print(f"  Validated: {len(state.constraints)} constraints")
    print(f"  Rejected: {original_count - len(state.constraints)} constraints")
    print(f"  Errors found: {len(state.validation_errors)}")
    
    # Set flag if we need corrections
    state.needs_correction = len(state.validation_errors) > 5  # Threshold for re-extraction
    
    return state


def extract_interactions_node(state: ExtractionState) -> ExtractionState:
    """Node 5: Extract interactions from each chunk with structured output."""
    print("\n" + "="*80)
    print("NODE 5: EXTRACTING INTERACTIONS")
    print("="*80)
    
    for i, chunk in enumerate(state.chunks):
        print(f"  Processing chunk {i+1}/{len(state.chunks)}")
        
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
Types: meets, fights, helps, speaks, warns, assigns, promises, trusts

Text: {chunk}

Output:"""
        
        response = state.llm_model.generate(prompt, max_new_tokens=100)
        
        # Parse response
        for line in response.split('\n'):
            line = line.strip()
            
            # Clean markdown
            line = re.sub(r'^[\d\-\*\.]+\s*', '', line)
            line = re.sub(r'\*\*', '', line)
            
            if '|' not in line:
                continue
            
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 4:
                continue
            
            char1 = parts[0]
            char2 = parts[1]
            i_type = parts[2]
            desc = parts[3]
            temporal = parts[4] if len(parts) >= 5 else "present"
            
            # Remove common prefixes
            char1 = re.sub(r'^(output|format|interaction|name1|character1)\s*:?\s*', '', char1, flags=re.IGNORECASE).strip()
            char2 = re.sub(r'^(output|format|interaction|name2|character2)\s*:?\s*', '', char2, flags=re.IGNORECASE).strip()
            
            # Validate both characters exist
            if char1 not in state.characters or char2 not in state.characters:
                continue
            
            # Skip if empty
            if not char1 or not char2 or not desc:
                continue
            
            # Skip if too short
            if len(desc) < 3:
                continue
            
            # Skip self-interactions (already handled by Pydantic validator, but double-check)
            if char1 == char2:
                continue
            
            # Filter invalid patterns
            invalid_patterns = [
                '(none)', 'output should', 'based on', 'format:', 'example:',
                'extract', 'character', 'interaction', 'description', 'text:',
                'actor', 'receiver', 'critical:', 'important:'
            ]
            if any(pattern in desc.lower() for pattern in invalid_patterns):
                continue
            
            # Check for action verb
            action_verbs = ['warn', 'assign', 'promise', 'meet', 'fight', 'help', 'speak', 'tell', 'ask', 'trust', 'give']
            has_verb = any(verb in desc.lower() for verb in action_verbs)
            if not has_verb and len(desc.split()) < 3:
                continue
            
            # Normalize temporal tag
            temporal = temporal.lower().strip()
            if temporal not in ['past', 'present', 'future']:
                temporal = 'present'
            
            # Normalize interaction type
            i_type_lower = i_type.lower().strip()
            valid_types = ['meets', 'fights', 'helps', 'speaks', 'warns', 'assigns', 'promises', 'trusts']
            if i_type_lower not in valid_types:
                # Try to map common variations
                if 'meet' in i_type_lower:
                    i_type_lower = 'meets'
                elif 'fight' in i_type_lower:
                    i_type_lower = 'fights'
                elif 'help' in i_type_lower:
                    i_type_lower = 'helps'
                elif 'speak' in i_type_lower or 'talk' in i_type_lower or 'tell' in i_type_lower:
                    i_type_lower = 'speaks'
                elif 'warn' in i_type_lower:
                    i_type_lower = 'warns'
                elif 'assign' in i_type_lower:
                    i_type_lower = 'assigns'
                elif 'promise' in i_type_lower:
                    i_type_lower = 'promises'
                elif 'trust' in i_type_lower:
                    i_type_lower = 'trusts'
                else:
                    continue  # Skip invalid types
            
            # Create Pydantic model (validates automatically)
            try:
                interaction = Interaction(
                    character1=char1,
                    character2=char2,
                    interaction_type=i_type_lower,
                    description=desc,
                    temporal_tag=temporal,
                    chunk_index=i,
                    source_chunk=chunk,
                    confidence=0.8
                )
                state.interactions.append(interaction)
            except Exception as e:
                print(f"    Validation failed: {e}")
                continue
    
    print(f"Extracted {len(state.interactions)} interactions")
    return state


# ============================================================================
# MAIN GRAPH CONSTRUCTION
# ============================================================================

def create_extraction_graph() -> StateGraph:
    """Build the LangGraph workflow."""
    
    # Create graph
    workflow = StateGraph(ExtractionState)
    
    # Add nodes
    workflow.add_node("extract_characters", extract_characters_node)
    workflow.add_node("extract_constraints", extract_constraints_node)
    workflow.add_node("validate_constraints", validate_constraints_node)
    workflow.add_node("extract_interactions", extract_interactions_node)
    
    # Define correction decision function
    def should_correct(state: ExtractionState) -> str:
        """Node 4: Decide if we need correction loop."""
        if state.needs_correction:
            print("\n⚠️  Too many validation errors - skipping correction for now (would loop back in full implementation)")
            # In production, this would route back to extract_constraints with feedback
            # For now, we continue forward
        return "extract_interactions"
    
    # Set entry point
    workflow.set_entry_point("extract_characters")
    
    # Add edges
    workflow.add_edge("extract_characters", "extract_constraints")
    workflow.add_edge("extract_constraints", "validate_constraints")
    
    # Node 4: Conditional edge - correction or continue
    workflow.add_conditional_edges(
        "validate_constraints",
        should_correct,
        {
            "extract_interactions": "extract_interactions"
        }
    )
    
    workflow.add_edge("extract_interactions", END)
    
    return workflow.compile()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

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
    
    # Initialize models
    print("\nInitializing models...")
    llm = QwenLLM(model_name="Qwen/Qwen2.5-7B-Instruct")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create initial state
    initial_state = ExtractionState(
        story_text=story,
        llm_model=llm,
        embedding_model=embedding_model
    )
    
    # Build and run graph
    print("\nBuilding extraction graph...")
    graph = create_extraction_graph()
    
    print("\nRunning extraction pipeline...")
    final_state = graph.invoke(initial_state)
    
    # Display results (graph returns dict, not ExtractionState)
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE - NODES 1-5")
    print("="*80)
    print(f"Characters extracted: {final_state['characters']}")
    print(f"Total chunks created: {len(final_state['chunks'])}")
    print(f"Constraints extracted: {len(final_state['constraints'])}")
    print(f"Interactions extracted: {len(final_state['interactions'])}")
    print(f"Validation errors: {len(final_state['validation_errors'])}")
    
    print(f"\nValidated constraints:")
    for constraint in final_state['constraints'][:8]:
        print(f"  - {constraint.character} [{constraint.constraint_type}] {constraint.value} ({constraint.temporal_tag})")
    
    print(f"\nExtracted interactions:")
    for interaction in final_state['interactions'][:8]:
        print(f"  - {interaction.character1} --[{interaction.interaction_type}]--> {interaction.character2}: {interaction.description} ({interaction.temporal_tag})")
    
    if final_state['validation_errors']:
        print(f"\nSample validation errors:")
        for error in final_state['validation_errors'][:5]:
            print(f"  ! {error}")
