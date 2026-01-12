"""
TextChunker for handling text chunking with overlap.
"""

import re
from typing import List


class TextChunker:
    """Handles text chunking with overlap."""
    
    def __init__(self, chunk_size: int, overlap: int):
        """Initialize chunker with required chunk_size and overlap parameters."""
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_words(self, text: str) -> List[str]:
        """Chunk text by sentences with word-based overlap."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Keep overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    s_len = len(s.split())
                    if overlap_length + s_len <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += s_len
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_text(self, text: str) -> List[str]:
        """Alias for chunk_by_words for backward compatibility."""
        return self.chunk_by_words(text)
