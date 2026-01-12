"""
ConstraintFilter for filtering and validating constraint data.
"""

from typing import List, Optional


class ConstraintFilter:
    """Filters and validates constraint data."""
    
    INVALID_PATTERNS = [
        '(none)', 'output should', 'based on', 'format:', 'example:',
        'extract', 'character', 'constraint', 'description', 'text:',
        'critical:', 'important:', 'wrong', 'correct'
    ]
    
    # Verbs that indicate actions toward another character (interactions, not constraints)
    INTERACTION_VERBS = [
        'told', 'tell', 'warned', 'warn', 'met', 'meet', 
        'assigned', 'assign', 'promised', 'promise', 
        'said', 'say', 'spoke', 'speak', 'helped', 'help',
        'fought', 'fight', 'trusted', 'trust', 'gave', 'give'
    ]
    
    @staticmethod
    def is_valid_description(desc: str) -> bool:
        """Check if description is valid."""
        if len(desc) < 3:
            return False
        
        if any(pattern in desc.lower() for pattern in ConstraintFilter.INVALID_PATTERNS):
            return False
        
        return True
    
    @staticmethod
    def is_interaction_not_constraint(desc: str, character: str, all_characters: List[str]) -> bool:
        """Check if this looks like an interaction rather than a constraint."""
        desc_lower = desc.lower()
        other_chars = [c for c in all_characters if c != character]
        
        # Check for interaction verbs WITH another character mentioned
        if any(verb in desc_lower for verb in ConstraintFilter.INTERACTION_VERBS):
            if any(char_name.lower() in desc_lower for char_name in other_chars):
                return True
        
        # Additional check: if description starts with a verb in imperative form
        # like "warn Alice", "tell Bob", "be careful", it's likely part of an interaction
        words = desc_lower.split()
        if len(words) >= 1:
            first_word = words[0]
            
            # Imperative verbs that are likely from interaction contexts
            imperative_verbs = ConstraintFilter.INTERACTION_VERBS + ['be', 'do', 'go', 'come', 'stay']
            
            if first_word in imperative_verbs:
                # If it's an imperative without a subject, it's likely from indirect speech
                # like "Bob warned Alice to [be careful]" where "be careful" shouldn't be Bob's constraint
                return True
        
        return False
    
    @staticmethod
    def normalize_constraint_type(c_type: str, description: str = "") -> Optional[str]:
        """Normalize constraint type to valid value based on type string and description."""
        c_type_lower = c_type.lower()
        desc_lower = description.lower()
        
        # Check if description indicates a past choice/action rather than a rule
        choice_verbs = ['refused', 'chose not', 'decided not', 'declined', 'rejected']
        if any(verb in desc_lower for verb in choice_verbs):
            return 'state'  # It's a past action/state, not a prohibition
        
        # First check explicit type keywords
        if 'ability' in c_type_lower:
            return 'ability'
        elif 'prohibition' in c_type_lower:
            # Double-check: if it's about future actions ("will do X"), it's a state, not prohibition
            if 'will' in desc_lower and any(action in desc_lower for action in ['review', 'investigate', 'do', 'go', 'make']):
                return 'state'
            return 'prohibition'
        elif 'trait' in c_type_lower:
            return 'trait'
        elif 'state' in c_type_lower:
            return 'state'
        
        # If no match, infer from description
        # Future actions are states
        if 'will' in desc_lower:
            return 'state'
        
        # "is/are" usually indicates trait or state
        if desc_lower.startswith('is ') or desc_lower.startswith('are '):
            return 'trait'
        
        # "can/cannot" indicates ability/prohibition
        if 'can ' in desc_lower:
            return 'ability'
        if 'cannot ' in desc_lower or "can't" in desc_lower:
            return 'prohibition'
        
        return None
    
    @staticmethod
    def normalize_temporal_tag(temporal: str, description: str) -> str:
        """Normalize and auto-correct temporal tag based on description."""
        temporal = temporal.lower().strip()
        desc_lower = description.lower()
        
        # Auto-correct based on markers
        if 'will' in desc_lower and temporal not in ['future', 'present']:
            return 'future'
        
        if any(marker in desc_lower for marker in ['was once', 'used to be', 'had been']):
            return 'past'
        
        # Validate
        if temporal not in ['past', 'present', 'future', 'habitual']:
            return 'present'
        
        return temporal
