"""
Configuration constants for constraint extraction pipeline.
All hyperparameters and model paths are centralized here.
"""

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# LLM Model
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 2048  # Default token limit for generation
MAX_NEW_TOKENS_CONSTRAINT = 300  # For constraint extraction
MAX_NEW_TOKENS_INTERACTION = 300  # For interaction extraction

# Embedding Model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================================
# TEXT PROCESSING CONFIGURATION
# ============================================================================

# Chunking Parameters
CHUNK_SIZE_CHARACTERS = 400  # Large chunks for character extraction (better context, pronoun resolution)
CHUNK_SIZE_INTERACTIONS = 250  # Medium chunks for interaction extraction (balance context and precision)
CHUNK_SIZE_CONSTRAINTS = 150  # Smallest chunks for constraint extraction (maximum precision)
OVERLAP_CHARACTERS = 40  # 20% overlap for character extraction
OVERLAP_INTERACTIONS = 30  # 20% overlap for interaction extraction
OVERLAP_CONSTRAINTS = 20  # 20% overlap for constraint extraction

# ============================================================================
# DEDUPLICATION CONFIGURATION
# ============================================================================

# Semantic Similarity Threshold (0.0 to 1.0)
# Higher values = stricter deduplication (only very similar items removed)
SIMILARITY_THRESHOLD = 0.85  # Lowered to catch more similar duplicates

# ============================================================================
# VALIDATION CONFIGURATION
# ============================================================================

# Validation Thresholds
CORRECTION_THRESHOLD = 5  # Number of rejected constraints before correction needed

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Output File
OUTPUT_FILE = "report.json"
