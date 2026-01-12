"""
ConstraintValidator for validating extracted constraints.
"""

from typing import List, Tuple

from models import Constraint


class ConstraintValidator:
    """Validates extracted constraints with detailed error reporting."""
    
    def __init__(self, characters: List[str]):
        self.characters = characters
        self.seen_keys = set()
    
    def validate(self, constraint: Constraint) -> Tuple[bool, List[str]]:
        """Validate a constraint. Returns (is_valid, errors)."""
        errors = []
        
        # Check character exists
        if constraint.character not in self.characters:
            errors.append(f"Character '{constraint.character}' not in character list")
        
        # Temporal logic validation
        desc_lower = constraint.value.lower()
        
        if 'will' in desc_lower and constraint.temporal_tag not in ['future', 'present']:
            errors.append(f"Contains 'will' but marked as {constraint.temporal_tag}")
        
        if any(marker in desc_lower for marker in ['was once', 'used to be', 'had been']) and constraint.temporal_tag != 'past':
            errors.append(f"Contains past marker but marked as {constraint.temporal_tag}")
        
        # Auto-correct constraint type if needed
        # Future actions ("will X") should be states, not prohibitions
        if constraint.constraint_type == 'prohibition' and 'will' in desc_lower:
            if any(action in desc_lower for action in ['review', 'investigate', 'examine', 'do', 'go', 'make']):
                # This should be a state, not a prohibition
                # Note: We can't modify the constraint directly, but we can flag it
                errors.append("Future action misclassified as prohibition (should be state)")
                # For now, just flag it - ideally we'd correct it
        
        # Check for misclassified interactions
        interaction_indicators = ['told', 'warned', 'assigned', 'met', 'said', 'promised', 'spoke', 'helped']
        other_chars = [c for c in self.characters if c != constraint.character]
        
        if any(verb in desc_lower for verb in interaction_indicators):
            if any(char_name in constraint.value for char_name in other_chars):
                errors.append(f"Looks like an interaction, not a constraint")
                return False, errors  # Critical error
        
        # Additional check: imperative verbs without subject likely from interactions
        # "be careful", "go there", "do that" are likely from "X told Y to [imperative]"
        if constraint.constraint_type in ['prohibition', 'ability']:
            words = constraint.value.lower().split()
            if len(words) >= 1:
                imperative_verbs = ['be', 'do', 'go', 'come', 'stay', 'tell', 'warn']
                if words[0] in imperative_verbs and len(words) <= 3:
                    # Short imperative phrases are likely from interaction context
                    errors.append(f"Short imperative likely from interaction context")
                    return False, errors  # Critical error
        
        # Deduplication
        constraint_key = f"{constraint.character.lower()}|{constraint.constraint_type}|{constraint.value.lower()}"
        if constraint_key in self.seen_keys:
            errors.append("Duplicate constraint")
            return False, errors  # Critical error
        
        self.seen_keys.add(constraint_key)
        
        # Description quality - allow single-word traits like 'skilled', 'dedicated'
        if len(constraint.value.strip()) == 0:
            errors.append("Empty description")
            return False, errors  # Critical error
        
        # Auto-correctable errors are not critical
        return True, errors
