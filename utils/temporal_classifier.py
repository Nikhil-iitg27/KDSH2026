"""
TemporalClassifier for reclassifying temporal tags using additional context and markers.
"""

from typing import List

from models import Constraint, Interaction


class TemporalClassifier:
    """Reclassifies temporal tags using additional context and markers."""
    
    BACKSTORY_MARKERS = [
        'was once', 'used to be', 'had been', 'had met', 'had worked',
        'years ago', 'long ago', 'in the past', 'previously', 'formerly',
        'back then', 'at that time'
    ]
    
    FUTURE_MARKERS = [
        'will', 'going to', 'plans to', 'intends to', 'promises to',
        'tomorrow', 'next', 'soon', 'later', 'eventually'
    ]
    
    HABITUAL_MARKERS = [
        'always', 'never', 'usually', 'often', 'sometimes', 'can',
        'is able to', 'has the ability', 'tends to', 'habitually'
    ]
    
    @staticmethod
    def reclassify_constraint(constraint: Constraint) -> Constraint:
        """Reclassify constraint temporal tag based on detailed analysis."""
        desc_lower = constraint.value.lower()
        
        # Check for backstory markers
        if any(marker in desc_lower for marker in TemporalClassifier.BACKSTORY_MARKERS):
            constraint.temporal_tag = 'past'
            return constraint
        
        # Check for future markers
        if any(marker in desc_lower for marker in TemporalClassifier.FUTURE_MARKERS):
            constraint.temporal_tag = 'future'
            return constraint
        
        # Check for habitual markers (for traits/abilities)
        if constraint.constraint_type in ['trait', 'ability']:
            if any(marker in desc_lower for marker in TemporalClassifier.HABITUAL_MARKERS):
                constraint.temporal_tag = 'habitual'
                return constraint
        
        # If no clear markers and it's a trait, default to habitual
        if constraint.constraint_type == 'trait' and constraint.temporal_tag == 'present':
            constraint.temporal_tag = 'habitual'
        
        return constraint
    
    @staticmethod
    def reclassify_interaction(interaction: Interaction) -> Interaction:
        """Reclassify interaction temporal tag based on detailed analysis."""
        desc_lower = interaction.description.lower()
        
        # Check for backstory markers
        if any(marker in desc_lower for marker in TemporalClassifier.BACKSTORY_MARKERS):
            interaction.temporal_tag = 'past'
            return interaction
        
        # Check for future markers
        if any(marker in desc_lower for marker in TemporalClassifier.FUTURE_MARKERS):
            interaction.temporal_tag = 'future'
            return interaction
        
        # Default to present for main story events
        return interaction
