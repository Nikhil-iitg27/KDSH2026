"""
Workflow construction and pipeline execution.
Builds the LangGraph workflow and runs the extraction pipeline.
"""

import json
from typing import Any

from langgraph.graph import StateGraph, END

from config import OUTPUT_FILE
from models import ExtractionState
from utils import JSONAssembler
from core.llm import QwenLLM
from core.nodes import (
    extract_characters_node,
    extract_constraints_node,
    validate_constraints_node,
    should_correct,
    extract_interactions_node,
    validate_interactions_node,
    temporal_classification_node,
    consolidate_bidirectional_node,
    semantic_deduplication_node,
    story_validation_node
)


def assemble_json_node(state: ExtractionState) -> ExtractionState:
    """Node 11: Assemble final hierarchical JSON output."""
    
    # Assemble JSON
    output = JSONAssembler.assemble(
        state.characters,
        state.constraints,
        state.interactions,
        state.chunks,
        state.validation_errors
    )
    
    # Write to file
    output_file = OUTPUT_FILE
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Wrote output to {output_file}")
    print(f"  - {output['summary']['total_characters']} characters")
    print(f"  - {output['summary']['total_constraints']} constraints")
    print(f"  - {output['summary']['total_interactions']} interactions")
    print(f"  - {output['summary']['total_violations']} violations")
    
    return state


def create_extraction_graph() -> Any:
    """Create the complete LangGraph workflow with all 11 nodes."""
    workflow = StateGraph(ExtractionState)
    
    # Add all nodes
    workflow.add_node("extract_characters", extract_characters_node)
    workflow.add_node("extract_constraints", extract_constraints_node)
    workflow.add_node("validate_constraints", validate_constraints_node)
    workflow.add_node("extract_interactions", extract_interactions_node)
    workflow.add_node("validate_interactions", validate_interactions_node)
    workflow.add_node("temporal_classification", temporal_classification_node)
    workflow.add_node("consolidate_bidirectional", consolidate_bidirectional_node)
    workflow.add_node("semantic_deduplication", semantic_deduplication_node)
    workflow.add_node("story_validation", story_validation_node)
    workflow.add_node("assemble_json", assemble_json_node)
    
    # Define edges - linear flow for now
    workflow.set_entry_point("extract_characters")
    workflow.add_edge("extract_characters", "extract_constraints")
    workflow.add_edge("extract_constraints", "validate_constraints")
    
    # Conditional edge from constraint validation
    workflow.add_conditional_edges(
        "validate_constraints",
        should_correct,
        {
            "correct": "extract_interactions",  # Could loop back in full implementation
            "continue": "extract_interactions"
        }
    )
    
    # Continue with interaction pipeline
    workflow.add_edge("extract_interactions", "validate_interactions")
    workflow.add_edge("validate_interactions", "temporal_classification")
    workflow.add_edge("temporal_classification", "consolidate_bidirectional")
    workflow.add_edge("consolidate_bidirectional", "semantic_deduplication")
    workflow.add_edge("semantic_deduplication", "story_validation")
    workflow.add_edge("story_validation", "assemble_json")
    workflow.add_edge("assemble_json", END)
    
    return workflow.compile()


def run_extraction_pipeline(story_text: str) -> dict:
    """
    Main function to run the extraction pipeline on a story.
    
    Args:
        story_text: The input story text to process
        
    Returns:
        Dictionary with final state containing extracted data
    """
    # Initialize models
    print("Initializing models...")
    llm = QwenLLM()
    
    # Create initial state
    initial_state = ExtractionState(
        story_text=story_text,
        llm_model=llm
    )
    
    # Create and run graph
    print("\nCreating extraction graph...")
    graph = create_extraction_graph()
    
    print("\nRunning extraction pipeline...")
    final_state = graph.invoke(initial_state)
    
    # Generate validation summary if there are violations
    if final_state['validation_errors']:
        # Parse violations from validation_errors
        violations_data = []
        for err in final_state['validation_errors']:
            # Extract severity from error string
            if 'severity:' in err:
                severity_str = err.split('severity:')[-1].strip().rstrip(')')
                try:
                    severity = int(severity_str)
                    violations_data.append((severity, err))
                except:
                    pass
        
        # Calculate validity
        if violations_data:
            max_severity = max(v[0] for v in violations_data)
            high_severity_count = sum(1 for v in violations_data if v[0] >= 6)
            medium_severity_count = sum(1 for v in violations_data if v[0] >= 4)
            
            is_valid = True
            if max_severity >= 8:
                is_valid = False
            elif high_severity_count >= 2:
                is_valid = False
            elif medium_severity_count >= 3:
                is_valid = False
            
            print(f"\n" + "="*80)
            if is_valid:
                print("✓ STORY IS VALID")
            else:
                print("✗ STORY IS INVALID")
            print("="*80)
            
            # If invalid, provide concise reasoning
            if not is_valid:
                print("\nReason for invalidity:")
                if max_severity >= 8:
                    severe_violations = [v for v in violations_data if v[0] >= 8]
                    print(f"  Critical contradiction detected (severity {max_severity}/10).")
                    print(f"  The story contains factually impossible elements or direct contradictions.")
                elif high_severity_count >= 2:
                    print(f"  Multiple high-severity issues ({high_severity_count} violations ≥6/10).")
                    print(f"  The story has several significant inconsistencies that undermine credibility.")
                else:
                    print(f"  Multiple medium-severity issues ({medium_severity_count} violations ≥4/10).")
                    print(f"  The cumulative inconsistencies make the story implausible.")
        else:
            print(f"\n" + "="*80)
            print("✓ STORY IS VALID")
            print("="*80)
    else:
        print(f"\n" + "="*80)
        print("✓ STORY IS VALID")
        print("="*80)
    
    print(f"\nOutput saved to: {OUTPUT_FILE}")
    print("="*80)
    
    return final_state

