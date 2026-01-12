"""
JSONAssembler for assembling final hierarchical JSON output.
"""

from typing import List, Dict, Any

from models import Constraint, Interaction


class JSONAssembler:
    """Assembles final hierarchical JSON output."""
    
    @staticmethod
    def assemble(characters: List[str], constraints: List[Constraint], 
                 interactions: List[Interaction], chunks: List[str],
                 validation_errors: List[str] = None) -> Dict[str, Any]:
        """Assemble all extracted data into hierarchical JSON structure."""
        
        # Parse violations from validation_errors
        violations = []
        if validation_errors:
            for err in validation_errors:
                if err.startswith("[VIOLATION]"):
                    violations.append(err.replace("[VIOLATION] ", ""))
        
        output = {
            "summary": {
                "total_characters": len(characters),
                "total_constraints": len(constraints),
                "total_interactions": len(interactions),
                "total_chunks": len(chunks),
                "total_violations": len(violations)
            },
            "violations": violations,
            "characters": []
        }
        
        for char_name in characters:
            # Get constraints for this character
            char_constraints = [c for c in constraints if c.character == char_name]
            
            # Split constraints by temporal tag
            backstory_constraints = [c for c in char_constraints if c.temporal_tag == 'past']
            current_constraints = [c for c in char_constraints if c.temporal_tag != 'past']
            
            # Get interactions
            # Split by temporal tag - 'past' goes to backstory, others to current_story
            outgoing = [i for i in interactions if i.character1 == char_name and not i.is_bidirectional]
            incoming = [i for i in interactions if i.character2 == char_name and not i.is_bidirectional]
            bidirectional = [i for i in interactions if i.is_bidirectional and char_name in [i.character1, i.character2]]
            
            # Separate past vs current interactions
            outgoing_past = [i for i in outgoing if i.temporal_tag == 'past']
            outgoing_current = [i for i in outgoing if i.temporal_tag != 'past']
            incoming_past = [i for i in incoming if i.temporal_tag == 'past']
            incoming_current = [i for i in incoming if i.temporal_tag != 'past']
            bidirectional_past = [i for i in bidirectional if i.temporal_tag == 'past']
            bidirectional_current = [i for i in bidirectional if i.temporal_tag != 'past']
            
            char_data = {
                "name": char_name,
                "backstory": {
                    "constraints": [
                        {
                            "type": c.constraint_type,
                            "value": c.value,
                            "temporal_tag": c.temporal_tag,
                            "chunk_index": c.chunk_index,
                            "source_chunk": c.source_chunk,
                            "confidence": c.confidence
                        }
                        for c in backstory_constraints
                    ],
                    "interactions": {
                        "outgoing": [
                            {
                                "target": i.character2,
                                "type": i.interaction_type,
                                "description": i.description,
                                "temporal_tag": i.temporal_tag,
                                "chunk_index": i.chunk_index
                            }
                            for i in outgoing_past
                        ],
                        "incoming": [
                            {
                                "source": i.character1,
                                "type": i.interaction_type,
                                "description": i.description,
                                "temporal_tag": i.temporal_tag,
                                "chunk_index": i.chunk_index
                            }
                            for i in incoming_past
                        ],
                        "bidirectional": [
                            {
                                "other_character": i.character2 if i.character1 == char_name else i.character1,
                                "type": i.interaction_type,
                                "description": i.description,
                                "temporal_tag": i.temporal_tag,
                                "chunk_index": i.chunk_index
                            }
                            for i in bidirectional_past
                        ]
                    }
                },
                "current_story": {
                    "constraints": [
                        {
                            "type": c.constraint_type,
                            "value": c.value,
                            "temporal_tag": c.temporal_tag,
                            "chunk_index": c.chunk_index,
                            "source_chunk": c.source_chunk,
                            "confidence": c.confidence
                        }
                        for c in current_constraints
                    ],
                    "interactions": {
                        "outgoing": [
                            {
                                "target": i.character2,
                                "type": i.interaction_type,
                                "description": i.description,
                                "temporal_tag": i.temporal_tag,
                                "chunk_index": i.chunk_index
                            }
                            for i in outgoing_current
                        ],
                        "incoming": [
                            {
                                "source": i.character1,
                                "type": i.interaction_type,
                                "description": i.description,
                                "temporal_tag": i.temporal_tag,
                                "chunk_index": i.chunk_index
                            }
                            for i in incoming_current
                        ],
                        "bidirectional": [
                            {
                                "other_character": i.character2 if i.character1 == char_name else i.character1,
                                "type": i.interaction_type,
                                "description": i.description,
                                "temporal_tag": i.temporal_tag,
                                "chunk_index": i.chunk_index
                            }
                            for i in bidirectional_current
                        ]
                    }
                }
            }
            
            output["characters"].append(char_data)
        
        return output
