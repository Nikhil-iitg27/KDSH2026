"""
SemanticDeduplicator for removing semantically similar constraints and interactions using embeddings.
"""

import torch
from typing import List
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, SIMILARITY_THRESHOLD
from models import Constraint, Interaction


class SemanticDeduplicator:
    """Removes semantically similar constraints and interactions using embeddings."""
    
    def __init__(self, model_name: str = None, threshold: float = None):
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self.threshold = threshold or SIMILARITY_THRESHOLD
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
    
    def deduplicate_constraints(self, constraints: List[Constraint]) -> List[Constraint]:
        """Remove semantically duplicate constraints."""
        if not constraints:
            return constraints
        
        # Group by character and type
        groups = {}
        for c in constraints:
            key = f"{c.character}|{c.constraint_type}"
            if key not in groups:
                groups[key] = []
            groups[key].append(c)
        
        deduplicated = []
        removed_count = 0
        
        for key, group in groups.items():
            if len(group) == 1:
                deduplicated.extend(group)
                continue
            
            # Compute embeddings
            texts = [c.value for c in group]
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            
            # Find unique constraints
            kept_indices = [0]  # Always keep first
            
            for i in range(1, len(group)):
                is_unique = True
                max_similarity = 0.0
                for j in kept_indices:
                    similarity = torch.nn.functional.cosine_similarity(
                        embeddings[i].unsqueeze(0),
                        embeddings[j].unsqueeze(0)
                    ).item()
                    max_similarity = max(max_similarity, similarity)
                    
                    if similarity >= self.threshold:
                        is_unique = False
                        print(f"    Removing duplicate: '{texts[i][:50]}...' (similarity: {similarity:.3f} to '{texts[j][:50]}...')")
                        removed_count += 1
                        break
                
                if is_unique:
                    kept_indices.append(i)
            
            deduplicated.extend([group[i] for i in kept_indices])
        
        print(f"  Total removed: {removed_count} semantically similar constraints")
        return deduplicated
    
    def deduplicate_interactions(self, interactions: List[Interaction]) -> List[Interaction]:
        """Remove semantically duplicate interactions."""
        if not interactions:
            return interactions
        
        # Group by character pair and type
        groups = {}
        for inter in interactions:
            key = f"{inter.character1}|{inter.character2}|{inter.interaction_type}"
            if key not in groups:
                groups[key] = []
            groups[key].append(inter)
        
        deduplicated = []
        removed_count = 0
        
        for key, group in groups.items():
            if len(group) == 1:
                deduplicated.extend(group)
                continue
            
            # Compute embeddings
            texts = [i.description for i in group]
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            
            # Find unique interactions
            kept_indices = [0]
            
            for i in range(1, len(group)):
                is_unique = True
                max_similarity = 0.0
                for j in kept_indices:
                    similarity = torch.nn.functional.cosine_similarity(
                        embeddings[i].unsqueeze(0),
                        embeddings[j].unsqueeze(0)
                    ).item()
                    max_similarity = max(max_similarity, similarity)
                    
                    if similarity >= self.threshold:
                        is_unique = False
                        print(f"    Removing duplicate: '{texts[i][:50]}...' (similarity: {similarity:.3f} to '{texts[j][:50]}...')")
                        removed_count += 1
                        break
                
                if is_unique:
                    kept_indices.append(i)
            
            deduplicated.extend([group[i] for i in kept_indices])
        
        print(f"  Total removed: {removed_count} semantically similar interactions")
        return deduplicated
