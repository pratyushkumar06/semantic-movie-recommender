"""Text chunking strategies for breaking documents into manageable pieces.

This module implements three distinct chunking approaches:
1. Fixed token chunking: Splits by token count with overlap
2. Sentence-based chunking: Preserves sentence boundaries
3. Semantic chunking: Groups semantically similar content

These methods are benchmarked to compare trade-offs between chunk coherence,
embedding quality, and computational efficiency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from transformers import AutoTokenizer

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


@dataclass
class Chunkers:
    """Container for multiple text chunking strategies.
    
    Attributes:
        token_limit: Maximum tokens per chunk.
        overlap: Token overlap between consecutive chunks.
        tokenizer: HuggingFace tokenizer for token-based splitting.
        semantic_model_name: Model name for semantic embedding-based splitting.
    """
    token_limit: int
    overlap: int
    tokenizer: AutoTokenizer
    semantic_model_name: str

    def __post_init__(self) -> None:
        """Initialize chunking strategies. Validates overlap < token_limit."""
        if self.overlap >= self.token_limit:
            raise ValueError("overlap must be smaller than token_limit")
        self._sentence_splitter = SentenceSplitter(
            chunk_size=self.token_limit,
            chunk_overlap=self.overlap,
        )
        self._semantic_embed = HuggingFaceEmbedding(model_name=self.semantic_model_name)
        self._semantic_splitter = SemanticSplitterNodeParser(
            embed_model=self._semantic_embed,
            chunk_size=self.token_limit,
        )

    def fixed_token_chunks(self, text: str) -> List[str]:
        """Split text into fixed-size token chunks with overlap.
        
        Strategy: Divides text into chunks of exactly token_limit size,
        with configurable overlap between consecutive chunks. Simple and
        predictable but may split sentences mid-way.
        
        Args:
            text: Input text to chunk.
            
        Returns:
            List of text chunks.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        step = self.token_limit - self.overlap
        chunks: List[str] = []
        for start in range(0, len(tokens), step):
            end = start + self.token_limit
            chunk_tokens = tokens[start:end]
            if not chunk_tokens:
                break
            chunks.append(self.tokenizer.decode(chunk_tokens))
        return chunks

    def sentence_chunks(self, text: str) -> List[str]:
        """Split text into sentence-aware chunks.
        
        Strategy: Uses sentence boundary detection to create chunks that
        respect natural language structure. Better coherence than fixed tokens
        but may exceed token_limit on long sentences.
        
        Args:
            text: Input text to chunk.
            
        Returns:
            List of text chunks respecting sentence boundaries.
        """
        doc = Document(text=text)
        nodes = self._sentence_splitter.get_nodes_from_documents([doc])
        return [node.get_content().strip() for node in nodes if node.get_content().strip()]

    def semantic_chunks(self, text: str) -> List[str]:
        """Split text into semantically coherent chunks using embeddings.
        
        Strategy: Uses embedding-based semantic similarity to group related
        content together. Produces most coherent chunks but is computationally
        expensive due to embedding generation during chunking.
        
        Args:
            text: Input text to chunk.
            
        Returns:
            List of semantically coherent text chunks.
        """
        doc = Document(text=text)
        nodes = self._semantic_splitter.get_nodes_from_documents([doc])
        return [node.get_content().strip() for node in nodes if node.get_content().strip()]
