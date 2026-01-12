"""
ResponseParser for parsing LLM responses into structured data.
"""

import re
from typing import List, Optional


class ResponseParser:
    """Parses LLM responses into structured data."""
    
    @staticmethod
    def clean_line(line: str) -> str:
        """Remove markdown and formatting from a line."""
        line = re.sub(r'^[\d\-\*\.]+\s*', '', line)
        line = re.sub(r'\*\*', '', line)
        line = re.sub(r'[`:]', '', line)
        return line.strip()
    
    @staticmethod
    def parse_pipe_delimited(line: str) -> Optional[List[str]]:
        """Parse pipe-delimited line, return None if invalid."""
        if '|' not in line:
            return None
        parts = [p.strip() for p in line.split('|')]
        return parts if len(parts) >= 3 else None
    
    @staticmethod
    def extract_character_names(response: str, story_text: str) -> List[str]:
        """Parse character names from LLM response."""
        names = []
        for line in response.split('\n'):
            line = ResponseParser.clean_line(line)
            
            if line and len(line.split()) <= 3:  # Likely a name
                name = ' '.join(word.capitalize() for word in line.split())
                if name and name in story_text:
                    names.append(name)
        
        return names
