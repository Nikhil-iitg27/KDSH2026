"""
QwenLLM wrapper using LangChain for structured outputs.
"""

import torch
from typing import Type, TypeVar, Optional
from pydantic import BaseModel
from langchain_huggingface import HuggingFacePipeline
from langchain_core.language_models import BaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from config import MODEL_NAME

T = TypeVar('T', bound=BaseModel)


class QwenLLM:
    """Wrapper for Qwen model using LangChain interface with structured output support."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or MODEL_NAME
        print(f"Loading model: {self.model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # For CPU, move model manually
        if self.device == "cpu":
            model = model.to(self.device)
        
        # Create HuggingFace pipeline
        # Note: When device_map="auto" is used, don't specify device in pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            max_new_tokens=2048,  # Increased for structured JSON output with multiple items
            temperature=0.5,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,  # Reduce repetition/echoing
            device=-1 if self.device == "cpu" else None  # None when using device_map="auto"
        )
        
        # Wrap in LangChain
        self.llm: BaseLLM = HuggingFacePipeline(pipeline=pipe)
    
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate response from prompt (for backward compatibility)."""
        # Update pipeline max_new_tokens if different
        if max_new_tokens != 256:
            self.llm.pipeline.max_new_tokens = max_new_tokens
        
        response = self.llm.invoke(prompt)
        return response.strip()
    
    def with_structured_output(self, schema: Type[T]) -> 'StructuredLLM[T]':
        """Return a structured output version that returns Pydantic models."""
        return StructuredLLM(self.llm, schema, self.tokenizer)


class StructuredLLM:
    """Wrapper for structured output generation using Pydantic schemas."""
    
    def __init__(self, llm: BaseLLM, schema: Type[T], tokenizer):
        self.llm = llm
        self.schema = schema
        self.tokenizer = tokenizer
    
    def invoke(self, prompt: str) -> T:
        """Generate structured output matching the Pydantic schema."""
        import json
        import re
        
        # Get schema info for better prompting
        schema_str = self.schema.model_json_schema()
        root_key = None
        if 'properties' in schema_str:
            # Get the first property name (like "constraints" or "interactions")
            root_key = list(schema_str['properties'].keys())[0] if schema_str['properties'] else None
        
        enhanced_prompt = f"""{prompt}

You must respond with ONLY a valid JSON object starting with {{ and ending with }}.
{f'The root object must have a "{root_key}" field.' if root_key else ''}
No extra text before or after the JSON.

JSON output:"""
        
        response = self.llm.invoke(enhanced_prompt)
        
        # Extract JSON from response
        response = response.strip()
        
        # Look for JSON after common markers
        json_markers = ['JSON output:', 'JSON:', 'Output:', '```json', '```']
        for marker in json_markers:
            if marker in response:
                response = response.split(marker, 1)[1].strip()
                break
        
        # Remove markdown code blocks if present
        response = response.replace('```json', '').replace('```', '').strip()
        
        # Find the first complete JSON object
        # Use a more robust approach: find first { and matching }
        start_idx = response.find('{')
        if start_idx == -1:
            raise ValueError(f"No JSON object found in response: {response[:500]}...")
        
        # Count braces to find the matching closing brace
        brace_count = 0
        end_idx = -1
        for i in range(start_idx, len(response)):
            if response[i] == '{':
                brace_count += 1
            elif response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
        
        if end_idx != -1:
            json_str = response[start_idx:end_idx+1]
            try:
                data = json.loads(json_str)
                
                # Handle empty object - add empty array for root key if missing
                if root_key and root_key not in data:
                    data[root_key] = []
                
                return self.schema.model_validate(data)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Failed to parse JSON: {e}")
                print(f"JSON string attempted: {json_str[:500]}...")
                raise
        else:
            raise ValueError(f"No complete JSON object found in response: {response[:500]}...")

