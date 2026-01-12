"""
BidirectionalConsolidator for consolidating bidirectional interactions.
"""

from typing import List

from models import Interaction


class BidirectionalConsolidator:
    """Consolidates bidirectional interactions."""
    
    BIDIRECTIONAL_TYPES = ['meets', 'fights', 'speaks']
    
    @staticmethod
    def consolidate(interactions: List[Interaction]) -> List[Interaction]:
        """Consolidate bidirectional interaction pairs."""
        # Create a map to find pairs
        interaction_map = {}
        consolidated = []
        processed_pairs = set()
        
        # First pass: identify all interactions
        for interaction in interactions:
            if interaction.interaction_type not in BidirectionalConsolidator.BIDIRECTIONAL_TYPES:
                consolidated.append(interaction)
                continue
            
            # Create keys for forward and reverse
            forward_key = f"{interaction.character1}|{interaction.character2}|{interaction.interaction_type}"
            reverse_key = f"{interaction.character2}|{interaction.character1}|{interaction.interaction_type}"
            
            # Check if we've already processed this pair
            if forward_key in processed_pairs or reverse_key in processed_pairs:
                continue
            
            # Check if reverse exists
            reverse_interaction = None
            for other in interactions:
                if (other.character1 == interaction.character2 and 
                    other.character2 == interaction.character1 and 
                    other.interaction_type == interaction.interaction_type):
                    reverse_interaction = other
                    break
            
            if reverse_interaction:
                # Found a bidirectional pair - merge them
                merged = Interaction(
                    character1=interaction.character1,
                    character2=interaction.character2,
                    interaction_type=interaction.interaction_type,
                    description=f"{interaction.character1} and {interaction.character2} {interaction.interaction_type}",
                    temporal_tag=interaction.temporal_tag,
                    chunk_index=interaction.chunk_index,
                    is_bidirectional=True,
                    source_chunk=interaction.source_chunk,
                    confidence=min(interaction.confidence, reverse_interaction.confidence)
                )
                consolidated.append(merged)
                processed_pairs.add(forward_key)
                processed_pairs.add(reverse_key)
            else:
                # No reverse found, keep as unidirectional
                consolidated.append(interaction)
                processed_pairs.add(forward_key)
        
        return consolidated
