"""Embedding encoding module for text vectorization and tokenization.

This module provides utilities for embedding text using sentence transformers
and managing token counts. It handles the loading of pre-trained models and 
offers functions for converting text to embeddings and analyzing token usage.

Constants:
    MODEL_NAME: The pre-trained sentence transformer model identifier.
    EMBEDDING_SIZE: Dimensionality of the generated embeddings (384 for MiniLM).
    TOKEN_LIMIT: Maximum tokens allowed per chunk (default 256).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_SIZE = 384
TOKEN_LIMIT = 256


@dataclass
class EncoderBundle:
    """Container for model and tokenizer instances.
    
    Attributes:
        model: Sentence transformer model for generating embeddings.
        tokenizer: HuggingFace tokenizer for token counting and encoding.
    """
    model: SentenceTransformer
    tokenizer: AutoTokenizer


def load_encoder(model_name: str = MODEL_NAME) -> EncoderBundle:
    """Load a sentence transformer model and its tokenizer.
    
    Args:
        model_name: HuggingFace model identifier. Defaults to all-MiniLM-L6-v2.
        
    Returns:
        EncoderBundle containing the loaded model and tokenizer.
    """
    model = SentenceTransformer(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return EncoderBundle(model=model, tokenizer=tokenizer)


def embed_texts(model: SentenceTransformer, texts: Iterable[str]) -> List[List[float]]:
    """Generate normalized embeddings for multiple texts.
    
    Args:
        model: The sentence transformer model to use for encoding.
        texts: Iterable of text strings to embed.
        
    Returns:
        List of embedding vectors (normalized to unit length).
    """
    return model.encode(list(texts), normalize_embeddings=True).tolist()


def embed_text(model: SentenceTransformer, text: str) -> List[float]:
    """Generate a normalized embedding for a single text string.
    
    Args:
        model: The sentence transformer model to use for encoding.
        text: Text string to embed.
        
    Returns:
        Embedding vector (normalized to unit length).
    """
    return model.encode(text, normalize_embeddings=True).tolist()


def count_tokens(tokenizer: AutoTokenizer, text: str) -> int:
    """Count the number of tokens in a text string.
    
    Args:
        tokenizer: The tokenizer to use for encoding.
        text: Text string to tokenize.
        
    Returns:
        Number of tokens in the text (excluding special tokens).
    """
    return len(tokenizer.encode(text, add_special_tokens=False))


def token_overflow(tokenizer: AutoTokenizer, text: str, limit: int = TOKEN_LIMIT) -> int:
    """Calculate how many tokens exceed the specified limit.
    
    Args:
        tokenizer: The tokenizer to use for encoding.
        text: Text string to analyze.
        limit: Maximum allowed tokens (default TOKEN_LIMIT).
        
    Returns:
        Number of tokens exceeding the limit (0 if within limit).
    """
    tokens = count_tokens(tokenizer, text)
    return max(0, tokens - limit)


def inspect_text_tokens(name: str, text: str, tokenizer: AutoTokenizer, limit: int = TOKEN_LIMIT) -> str:
    """Generate a formatted report of token usage for debugging.
    
    Args:
        name: Movie name for reporting.
        text: Text to analyze.
        tokenizer: The tokenizer to use for encoding.
        limit: Maximum allowed tokens (default TOKEN_LIMIT).
        
    Returns:
        Formatted string showing token count and overflow analysis.
    """
    tokens = count_tokens(tokenizer, text)
    overflow = tokens - limit
    overflow_str = f"+{overflow}" if overflow > 0 else "0"
    return f"Movie: {name}\nTokens: {tokens}\nOverflow: {overflow_str}"
