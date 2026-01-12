"""
InteractionFilter for filtering and validating interaction data.
"""

import re
from typing import Optional


class InteractionFilter:
    """Filters and validates interaction data."""
    
    INVALID_PATTERNS = [
        '(none)', 'output should', 'based on', 'format:', 'example:',
        'extract', 'character', 'interaction', 'description', 'text:',
        'actor', 'receiver', 'critical:', 'important:'
    ]
    
    ACTION_VERBS = ['warn', 'assign', 'promise', 'meet', 'fight', 'help', 'speak', 'tell', 'ask', 'trust', 'give']
    
    @staticmethod
    def clean_character_name(name: str) -> str:
        """Remove common prefixes from character names."""
        return re.sub(
            r'^(output|format|interaction|name1|name2|character1|character2)\s*:?\s*',
            '', name, flags=re.IGNORECASE
        ).strip()
    
    @staticmethod
    def is_valid_description(desc: str) -> bool:
        """Check if interaction description is valid."""
        if len(desc) < 3:
            return False
        
        if any(pattern in desc.lower() for pattern in InteractionFilter.INVALID_PATTERNS):
            return False
        
        # Check for action verb
        desc_lower = desc.lower()
        has_verb = any(verb in desc_lower for verb in InteractionFilter.ACTION_VERBS)
        if not has_verb and len(desc.split()) < 3:
            return False
        
        # Reject if it's just a future state without clear interpersonal action
        # "will review the case" is Carol's action, not an interaction with Alice
        if desc_lower.startswith('will ') and 'review' in desc_lower:
            # This is likely a future action/state, not an interaction
            return False
        
        return True
    
    @staticmethod
    def normalize_interaction_type(i_type: str) -> Optional[str]:
        """Normalize interaction type to valid value."""
        i_type_lower = i_type.lower().strip()
        
        type_mapping = {
            'meet': 'meets', 'fight': 'fights', 'help': 'helps',
            'speak': 'speaks', 'talk': 'speaks', 'tell': 'speaks',
            'warn': 'warns', 'assign': 'assigns', 'promise': 'promises',
            'trust': 'trusts'
        }
        
        valid_types = ['meets', 'fights', 'helps', 'speaks', 'warns', 'assigns', 'promises', 'trusts']
        
        if i_type_lower in valid_types:
            return i_type_lower
        
        for key, value in type_mapping.items():
            if key in i_type_lower:
                return value
        
        return None
    
    @staticmethod
    def normalize_temporal_tag(temporal: str) -> str:
        """Normalize temporal tag for interactions."""
        temporal = temporal.lower().strip()
        
        if temporal not in ['past', 'present', 'future']:
            return 'present'
        
        return temporal
