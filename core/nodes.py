"""
Graph nodes for the extraction pipeline.
All 10 nodes that process the extraction workflow.
"""

from config import CHUNK_SIZE_CHARACTERS, CHUNK_SIZE_INTERACTIONS, CHUNK_SIZE_CONSTRAINTS, OVERLAP_CHARACTERS, OVERLAP_INTERACTIONS, OVERLAP_CONSTRAINTS, SIMILARITY_THRESHOLD, CORRECTION_THRESHOLD
from models import ExtractionState
from utils import (
    TextChunker, ResponseParser,
    ConstraintFilter, InteractionFilter,
    TemporalClassifier, BidirectionalConsolidator,
    SemanticDeduplicator, NameCanonicalizer
)
from validators import ConstraintValidator, InteractionValidator, StoryValidator
from extractors import CharacterExtractor, ConstraintExtractor, InteractionExtractor


def extract_characters_node(state: ExtractionState) -> ExtractionState:
    """Node 1: Extract character names from all chunks and canonicalize them."""
    print("\n" + "="*80)
    print("NODE 1: EXTRACTING CHARACTERS")
    print("="*80)
    
    # Initialize components - use LARGE chunks for better context and pronoun resolution
    chunker = TextChunker(chunk_size=CHUNK_SIZE_CHARACTERS, overlap=OVERLAP_CHARACTERS)
    extractor = CharacterExtractor(state.llm_model, chunker)
    
    # Extract characters and chunks
    raw_characters, chunks = extractor.extract_from_story(state.story_text)
    
    # Canonicalize character names
    canonicalizer = NameCanonicalizer(raw_characters)
    canonical_characters = canonicalizer.get_canonical_names()
    name_mapping = canonicalizer.get_mapping()
    
    # Print mapping for transparency
    canonicalizer.print_mapping()
    
    # Update state
    state.characters = canonical_characters
    state.chunks = chunks
    state.name_mapping = name_mapping  # Store mapping for later use
    
    return state


def extract_constraints_node(state: ExtractionState) -> ExtractionState:
    """Node 2: Extract constraints from each chunk with structured output."""
    print("\n" + "="*80)
    print("NODE 2: EXTRACTING CONSTRAINTS")
    print("="*80)
    
    # Re-chunk with SMALLER chunks for precise constraint extraction
    print(f"\nRe-chunking text with smaller size ({CHUNK_SIZE_CONSTRAINTS} words) for precision...")
    constraint_chunker = TextChunker(chunk_size=CHUNK_SIZE_CONSTRAINTS, overlap=OVERLAP_CONSTRAINTS)
    constraint_chunks = constraint_chunker.chunk_text(state.story_text)
    print(f"Created {len(constraint_chunks)} constraint-focused chunks")
    
    # Initialize components
    filter_ = ConstraintFilter()
    name_mapping = getattr(state, 'name_mapping', {})
    extractor = ConstraintExtractor(state.llm_model, filter_, name_mapping)
    
    # Extract constraints using smaller chunks
    constraints = extractor.extract_from_chunks(constraint_chunks, state.characters)
    
    # Update state
    state.constraints = constraints
    
    return state


def validate_constraints_node(state: ExtractionState) -> ExtractionState:
    """Node 3: Validate constraints and auto-correct issues."""
    print("\n" + "="*80)
    print("NODE 3: VALIDATING CONSTRAINTS")
    print("="*80)
    
    # Initialize validator
    validator = ConstraintValidator(state.characters)
    
    # Validate each constraint
    valid_constraints = []
    rejected_count = 0
    auto_corrected = []
    
    for constraint in state.constraints:
        is_valid, errors = validator.validate(constraint)
        
        if not is_valid:
            rejected_count += 1
            state.validation_errors.extend([f"Rejected: {constraint.value[:50]}... - {err}" for err in errors])
            continue
        
        if errors:  # Has warnings but is valid
            auto_corrected.append(constraint.value[:50])
        
        valid_constraints.append(constraint)
    
    # Update state
    state.constraints = valid_constraints
    state.needs_correction = rejected_count > CORRECTION_THRESHOLD
    
    print(f"Validated: {len(valid_constraints)} passed, {rejected_count} rejected")
    if auto_corrected:
        print(f"Auto-corrected {len(auto_corrected)} constraints")
    
    if state.needs_correction:
        print("⚠️ High rejection rate - may need correction loop")
    
    return state


def should_correct(state: ExtractionState) -> str:
    """Conditional edge: decide if correction is needed."""
    if state.needs_correction:
        return "correct"
    else:
        return "continue"


def extract_interactions_node(state: ExtractionState) -> ExtractionState:
    """Node 5: Extract interactions from chunks."""
    print("\n" + "="*80)
    print("NODE 5: EXTRACTING INTERACTIONS")
    print("="*80)
    
    # Re-chunk with MEDIUM chunks for interaction extraction (balance context and precision)
    print(f"\nRe-chunking text with medium size ({CHUNK_SIZE_INTERACTIONS} words) for interactions...")
    interaction_chunker = TextChunker(chunk_size=CHUNK_SIZE_INTERACTIONS, overlap=OVERLAP_INTERACTIONS)
    interaction_chunks = interaction_chunker.chunk_text(state.story_text)
    print(f"Created {len(interaction_chunks)} interaction-focused chunks")
    
    # Initialize components
    filter_ = InteractionFilter()
    name_mapping = getattr(state, 'name_mapping', {})
    extractor = InteractionExtractor(state.llm_model, filter_, name_mapping)
    
    # Extract interactions using smaller chunks
    interactions = extractor.extract_from_chunks(interaction_chunks, state.characters)
    
    # Update state
    state.interactions = interactions
    
    return state


def validate_interactions_node(state: ExtractionState) -> ExtractionState:
    """Node 6: Validate interactions and flag bidirectional candidates."""
    print("\n" + "="*80)
    print("NODE 6: VALIDATING INTERACTIONS")
    print("="*80)
    
    # Initialize validator
    validator = InteractionValidator(state.characters)
    
    # Validate each interaction
    valid_interactions = []
    rejected_count = 0
    warning_count = 0
    
    for interaction in state.interactions:
        is_valid, errors = validator.validate(interaction)
        
        if not is_valid:
            rejected_count += 1
            state.validation_errors.extend(
                [f"Rejected interaction: {interaction.character1}→{interaction.character2} - {err}" 
                 for err in errors]
            )
            continue
        
        if errors:  # Has warnings but is valid
            warning_count += 1
            # Log warnings but keep the interaction
            for err in errors:
                if 'Bidirectional candidate' in err:
                    print(f"  ✓ Found bidirectional: {interaction.character1} ↔ {interaction.character2} ({interaction.interaction_type})")
                elif 'direction' in err.lower():
                    print(f"  ⚠ Direction warning: {interaction.character1}→{interaction.character2}: {err}")
        
        valid_interactions.append(interaction)
    
    # Get bidirectional candidates
    bidirectional_pairs = validator.get_bidirectional_candidates()
    
    # Update state
    state.interactions = valid_interactions
    
    print(f"\nValidated: {len(valid_interactions)} passed, {rejected_count} rejected, {warning_count} warnings")
    
    if bidirectional_pairs:
        print(f"Found {len(bidirectional_pairs)} bidirectional interaction pairs:")
        for char1, char2, i_type in bidirectional_pairs:
            print(f"  • {char1} ↔ {char2}: {i_type}")
    
    return state


def temporal_classification_node(state: ExtractionState) -> ExtractionState:
    """Node 7: Reclassify temporal tags with additional context."""
    print("\n" + "="*80)
    print("NODE 7: TEMPORAL CLASSIFICATION")
    print("="*80)
    
    # Reclassify constraints
    reclassified_constraints = []
    changes_count = 0
    
    for constraint in state.constraints:
        old_tag = constraint.temporal_tag
        new_constraint = TemporalClassifier.reclassify_constraint(constraint)
        
        if new_constraint.temporal_tag != old_tag:
            changes_count += 1
            print(f"  Reclassified: {constraint.character} | {old_tag} → {new_constraint.temporal_tag}")
        
        reclassified_constraints.append(new_constraint)
    
    # Reclassify interactions
    reclassified_interactions = []
    for interaction in state.interactions:
        old_tag = interaction.temporal_tag
        new_interaction = TemporalClassifier.reclassify_interaction(interaction)
        
        if new_interaction.temporal_tag != old_tag:
            changes_count += 1
            print(f"  Reclassified: {interaction.character1}→{interaction.character2} | {old_tag} → {new_interaction.temporal_tag}")
        
        reclassified_interactions.append(new_interaction)
    
    # Update state
    state.constraints = reclassified_constraints
    state.interactions = reclassified_interactions
    
    print(f"\nReclassified {changes_count} temporal tags")
    
    return state


def consolidate_bidirectional_node(state: ExtractionState) -> ExtractionState:
    """Node 8: Consolidate bidirectional interactions."""
    print("\n" + "="*80)
    print("NODE 8: BIDIRECTIONAL CONSOLIDATION")
    print("="*80)
    
    original_count = len(state.interactions)
    consolidated = BidirectionalConsolidator.consolidate(state.interactions)
    
    bidirectional_count = sum(1 for i in consolidated if i.is_bidirectional)
    
    state.interactions = consolidated
    
    print(f"Consolidated {original_count} → {len(consolidated)} interactions")
    print(f"Created {bidirectional_count} bidirectional interactions")
    
    return state


def semantic_deduplication_node(state: ExtractionState) -> ExtractionState:
    """Node 9: Remove semantically similar duplicates using embeddings."""
    print("\n" + "="*80)
    print("NODE 9: SEMANTIC DEDUPLICATION")
    print("="*80)
    
    # Initialize deduplicator
    deduplicator = SemanticDeduplicator(threshold=SIMILARITY_THRESHOLD)
    
    # Deduplicate constraints
    original_constraint_count = len(state.constraints)
    state.constraints = deduplicator.deduplicate_constraints(state.constraints)
    constraint_removed = original_constraint_count - len(state.constraints)
    
    print(f"Constraints: {original_constraint_count} → {len(state.constraints)} (removed {constraint_removed} duplicates)")
    
    # Deduplicate interactions
    original_interaction_count = len(state.interactions)
    state.interactions = deduplicator.deduplicate_interactions(state.interactions)
    interaction_removed = original_interaction_count - len(state.interactions)
    
    print(f"Interactions: {original_interaction_count} → {len(state.interactions)} (removed {interaction_removed} duplicates)")
    
    return state


def story_validation_node(state: ExtractionState) -> ExtractionState:
    """Node 10: Validate main story against backstory constraints."""
    print("\n" + "="*80)
    print("NODE 10: STORY VALIDATION (Backstory vs Main Story)")
    print("="*80)
    
    # Separate backstory from main story
    backstory_constraints = [c for c in state.constraints if c.temporal_tag == 'past']
    main_story_constraints = [c for c in state.constraints if c.temporal_tag != 'past']
    
    backstory_interactions = [i for i in state.interactions if i.temporal_tag == 'past']
    main_story_interactions = [i for i in state.interactions if i.temporal_tag != 'past']
    
    if not backstory_constraints and not backstory_interactions:
        print("No backstory found - skipping validation")
        return state
    
    if not main_story_constraints and not main_story_interactions:
        print("No main story events found - skipping validation")
        return state
    
    print(f"Validating {len(main_story_constraints)} constraints and {len(main_story_interactions)} interactions")
    print(f"Against {len(backstory_constraints)} backstory constraints and {len(backstory_interactions)} past interactions")
    print("Using LLM for event-by-event violation detection...\n")
    
    # Initialize validator with backstory and LLM
    validator = StoryValidator(
        backstory_constraints, 
        backstory_interactions,
        llm=state.llm_model
    )
    
    # Validate main story
    violations = validator.validate_constraints_and_interactions(
        main_story_constraints,
        main_story_interactions
    )
    
    # Store violations in state
    if violations:
        for v in violations:
            state.validation_errors.append(
                f"[VIOLATION] {v.character}: {v.explanation} (severity: {v.severity})"
            )
        
        # Print validation report
        report = validator.get_validation_report(violations)
        print("\n" + report)
    else:
        print("\n✓ No violations found. Main story is consistent with backstory.")
    
    return state
