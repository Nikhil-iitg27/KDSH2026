"""
InteractionValidator for validating extracted interactions.
"""

from typing import List, Tuple

from models import Interaction


class InteractionValidator:
    """Validates extracted interactions with detailed error reporting."""
    
    # Interaction types that can be bidirectional
    BIDIRECTIONAL_TYPES = ['meets', 'fights', 'speaks']
    
    def __init__(self, characters: List[str]):
        self.characters = characters
        self.seen_keys = set()
        self.bidirectional_candidates = []  # Store potential bidirectional pairs
    
    def validate(self, interaction: Interaction) -> Tuple[bool, List[str]]:
        """Validate an interaction. Returns (is_valid, errors)."""
        errors = []
        
        # Check both characters exist
        if interaction.character1 not in self.characters:
            errors.append(f"Character '{interaction.character1}' not in character list")
            return False, errors
        
        if interaction.character2 not in self.characters:
            errors.append(f"Character '{interaction.character2}' not in character list")
            return False, errors
        
        # Self-loop check (should be caught by Pydantic already, but double-check)
        if interaction.character1 == interaction.character2:
            errors.append("Self-interaction detected")
            return False, errors
        
        # Temporal logic validation
        desc_lower = interaction.description.lower()
        
        # Check for future markers
        if 'will' in desc_lower and interaction.temporal_tag not in ['future', 'present']:
            errors.append(f"Contains 'will' but marked as {interaction.temporal_tag}")
        
        # Check for past markers (backstory)
        past_markers = ['was once', 'used to be', 'had met', 'had been', 'years ago', 'long ago']
        if any(marker in desc_lower for marker in past_markers) and interaction.temporal_tag != 'past':
            errors.append(f"Contains past marker but marked as {interaction.temporal_tag}")
        
        # Direction validation: Check if actor/receiver seem reversed
        # Look for passive voice indicators that suggest reversed direction
        passive_indicators = ['was told by', 'was warned by', 'was assigned by', 'was helped by']
        if any(indicator in desc_lower for indicator in passive_indicators):
            errors.append("May have reversed direction (passive voice detected)")
        
        # Check for bidirectional candidate
        if interaction.interaction_type in self.BIDIRECTIONAL_TYPES:
            # Create reverse key to check for bidirectional pair
            reverse_key = f"{interaction.character2}|{interaction.character1}|{interaction.interaction_type}"
            forward_key = f"{interaction.character1}|{interaction.character2}|{interaction.interaction_type}"
            
            if reverse_key in self.seen_keys:
                # Found the reverse - this is bidirectional!
                self.bidirectional_candidates.append((
                    interaction.character1,
                    interaction.character2,
                    interaction.interaction_type
                ))
                errors.append(f"Bidirectional candidate with reverse pair")
        
        # Deduplication
        interaction_key = f"{interaction.character1}|{interaction.character2}|{interaction.interaction_type}|{interaction.description.lower()}"
        if interaction_key in self.seen_keys:
            errors.append("Duplicate interaction")
            return False, errors  # Critical error
        
        self.seen_keys.add(interaction_key)
        # Also add simplified key for bidirectional checking
        simple_key = f"{interaction.character1}|{interaction.character2}|{interaction.interaction_type}"
        self.seen_keys.add(simple_key)
        
        # Description quality check - ensure non-empty
        if len(interaction.description.strip()) < 2:
            errors.append("Description too short")
            return False, errors
        
        # Check that description contains an action toward another character
        action_words = ['told', 'warned', 'met', 'assigned', 'helped', 'fought', 'promised', 'trusted', 'spoke']
        if not any(word in desc_lower for word in action_words):
            # Check if this is actually a state/action that doesn't involve interaction
            if 'will review' in desc_lower or 'will investigate' in desc_lower:
                errors.append("Future action, not interpersonal interaction")
                return False, errors
            else:
                errors.append("Description lacks clear action verb")
        
        return True, errors
    
    def get_bidirectional_candidates(self) -> List[Tuple[str, str, str]]:
        """Get list of bidirectional interaction candidates."""
        return self.bidirectional_candidates
