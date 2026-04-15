"""Embedding module - handles document embeddings with Chroma DB."""

from E00_embedding.embedding import Embedding
from E00_embedding.chroma_db import ChromaDB

__all__ = ["Embedding", "ChromaDB"]
