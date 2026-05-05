#!/usr/bin/env python3
"""
Main document processor with support for multiple chunking strategies.
"""
import sys
from pathlib import Path

# Add modules directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from typing import List, Type
from C00_document_db_manager.document_manager import Document_Manager
from E00_embedding.embedding import Embedding
from E00_embedding.chroma_db import ChromaDB

# Import available chunkers
from D00_chunking.recursive_chunker import RecursiveChunker
from D00_chunking.semantic_chunker import SemanticChunker

# from D00_chunking.character_chunker import CharacterChunker

# Registry of chunking strategies
CHUNKER_REGISTRY = {
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
    # "character": CharacterChunker,
}

# Strategies that actually use chunk_size and chunk_overlap
STRATEGIES_USE_SIZE_OVERLAP = {"recursive"}  # add "character" later if needed


def get_chunker(strategy_name: str, chunk_size, chunk_overlap):
    """
    Factory function to get a chunker instance by name.
    - For chunkers that declare USES_SIZE_OVERLAP = False, they are instantiated without arguments.
    - For others, chunk_size and chunk_overlap are passed (must be provided by the caller).
    """
    strategy_name = strategy_name.lower()
    if strategy_name not in CHUNKER_REGISTRY:
        available = ", ".join(CHUNKER_REGISTRY.keys())
        raise ValueError(f"Unknown chunking strategy: {strategy_name}. Available: {available}")
    chunker_class = CHUNKER_REGISTRY[strategy_name]

    # Check if this chunker uses size/overlap
    if hasattr(chunker_class, 'USES_SIZE_OVERLAP') and not chunker_class.USES_SIZE_OVERLAP:
        # Instantiate without size/overlap
        return chunker_class()
    else:
        # Instantiate with size/overlap (caller must provide valid values)
        return chunker_class(chunk_size=chunk_size, chunk_overlap=chunk_overlap)



def process_documents(reset: bool = False, chunk_size: int = 800,
                      chunk_overlap: int = 80, chunking_strategy: str = "recursive"):
    """
    Process documents using the specified chunking strategy.
    Returns the number of chunks created.
    """
    print("\n" + "=" * 60)
    print("DOCUMENT PROCESSOR WITH BASECHUNKER")
    print("=" * 60)

    # Only show chunk size/overlap if the strategy actually uses them
    if chunking_strategy in STRATEGIES_USE_SIZE_OVERLAP:
        print(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    else:
        print(f"Chunking strategy: {chunking_strategy} (size/overlap not applicable)")

    print(f"Chunking strategy: {chunking_strategy}")

    if reset:
        print("\n✨ Clearing Database")
        ChromaDB.clear_database()

    # Load documents
    print("\n📄 Loading documents...")
    documents = Document_Manager.load_documents()
    print(f"✅ Loaded {len(documents)} documents")

    # Initialize chunker using selected strategy
    print(f"\n🔪 Initializing {chunking_strategy} chunker...")
    chunker = get_chunker(chunking_strategy, chunk_size, chunk_overlap)

    # Split documents
    print("\n✂️  Splitting documents into chunks...")
    chunks = chunker.split_documents(documents)

    # Verify BaseChunker features
    if chunks:
        print(f"\n🔍 VERIFICATION:")
        print(f"  - First chunk ID: {chunks[0].metadata.get('id', 'MISSING')}")
        print(f"  - Chunking strategy: {chunks[0].metadata.get('chunking_strategy', 'MISSING')}")
        print(f"  - Total chunks: {len(chunks)}")

        sample_id = chunks[0].metadata.get('id', '')
        if ':' in sample_id and sample_id.count(':') == 2:
            print(f"  ✅ Chunk ID format correct (source:page:index)")
        else:
            print(f"  ⚠️  Chunk ID format might be incorrect: {sample_id}")

    print(f"✅ Created {len(chunks)} chunks")

    # Add to ChromaDB
    print("\n💾 Adding chunks to ChromaDB...")
    Embedding.add_to_chroma(chunks)

    print("\n" + "=" * 60)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 60)

    return len(chunks)


def main():
    """CLI entry point (kept for backward compatibility)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=80, help="Chunk overlap")
    parser.add_argument("--strategy", type=str, default="recursive",
                        help=f"Chunking strategy: {', '.join(CHUNKER_REGISTRY.keys())}")
    args = parser.parse_args()

    process_documents(
        reset=args.reset,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_strategy=args.strategy
    )


if __name__ == "__main__":
    main()