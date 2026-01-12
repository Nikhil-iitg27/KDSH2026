"""
Name canonicalizer for mapping character name variants to canonical names.
"""

from typing import List, Dict, Set
import re


class NameCanonicalizer:
    """Canonicalizes character names by mapping variants to primary names."""
    
    # Common titles to strip
    TITLES = ['detective', 'chief', 'officer', 'captain', 'lieutenant', 'sergeant', 
              'dr', 'doctor', 'professor', 'mr', 'mrs', 'ms', 'miss']
    
    def __init__(self, raw_names: List[str]):
        """Initialize with raw extracted names."""
        self.raw_names = raw_names
        self.canonical_names: List[str] = []
        self.name_mapping: Dict[str, str] = {}
        self._canonicalize()
    
    def _strip_title(self, name: str) -> str:
        """Remove titles from name."""
        words = name.split()
        if len(words) > 1 and words[0].lower().rstrip('.') in self.TITLES:
            return ' '.join(words[1:])
        return name
    
    def _is_substring_match(self, short_name: str, long_name: str) -> bool:
        """Check if short_name is a meaningful substring of long_name."""
        # Simple heuristic: check if short name is the last word (surname)
        # or if short name appears in long name
        short_lower = short_name.lower()
        long_lower = long_name.lower()
        
        if short_lower == long_lower:
            return True
        
        # Check if short name is last word of long name (surname match)
        long_words = long_lower.split()
        if short_lower in long_words:
            return True
        
        # Check if short name is contained in long name
        if short_lower in long_lower:
            return True
        
        return False
    
    def _canonicalize(self):
        """Create mapping from variants to canonical names."""
        # Step 1: Strip titles and create base names
        base_names = {}
        for name in self.raw_names:
            stripped = self._strip_title(name)
            if stripped not in base_names:
                base_names[stripped] = []
            base_names[stripped].append(name)
        
        # Step 2: Group names by substring matching
        grouped: Dict[str, Set[str]] = {}
        processed = set()
        
        sorted_names = sorted(base_names.keys(), key=len, reverse=True)
        
        for long_name in sorted_names:
            if long_name in processed:
                continue
            
            # Start a new group with this name as canonical
            group = {long_name}
            group.update(base_names[long_name])
            
            # Find shorter names that match
            for short_name in sorted_names:
                if short_name == long_name or short_name in processed:
                    continue
                
                if self._is_substring_match(short_name, long_name):
                    group.add(short_name)
                    group.update(base_names[short_name])
                    processed.add(short_name)
            
            # Use longest full name as canonical (prefer "Alice Rodriguez" over "Alice")
            canonical = max(group, key=lambda x: (len(x.split()), len(x)))
            grouped[canonical] = group
            processed.add(long_name)
        
        # Step 3: Build mapping
        for canonical, variants in grouped.items():
            self.canonical_names.append(canonical)
            for variant in variants:
                self.name_mapping[variant] = canonical
        
        # Sort canonical names for consistency
        self.canonical_names.sort()
    
    def get_canonical_name(self, name: str) -> str:
        """Get canonical name for a given variant."""
        # Try direct lookup
        if name in self.name_mapping:
            return self.name_mapping[name]
        
        # Try without title
        stripped = self._strip_title(name)
        if stripped in self.name_mapping:
            return self.name_mapping[stripped]
        
        # If not found, return original name
        return name
    
    def get_canonical_names(self) -> List[str]:
        """Get list of canonical names."""
        return self.canonical_names
    
    def get_mapping(self) -> Dict[str, str]:
        """Get the full name mapping dictionary."""
        return self.name_mapping
    
    def print_mapping(self):
        """Print the name mapping for debugging."""
        print("\nCharacter Name Canonicalization:")
        print("=" * 60)
        for canonical in self.canonical_names:
            variants = [k for k, v in self.name_mapping.items() if v == canonical and k != canonical]
            if variants:
                print(f"{canonical}")
                for variant in sorted(variants):
                    print(f"  ← {variant}")
        print("=" * 60)
