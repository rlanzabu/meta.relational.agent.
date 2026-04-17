"""
Main document processor using BaseChunker strategies
"""

import sys
from pathlib import Path

# Add modules directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from C00_document_db_manager.document_manager import Document_Manager
from E00_embedding.embedding import Embedding
from E00_embedding.chroma_db import ChromaDB

# Import the new chunking strategies
from D00_chunking.recursive_chunker import RecursiveChunker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=80, help="Chunk overlap")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("DOCUMENT PROCESSOR WITH BASECHUNKER")
    print("=" * 60)
    print(f"Chunk size: {args.chunk_size}, Overlap: {args.chunk_overlap}")

    if args.reset:
        print("\n✨ Clearing Database")
        ChromaDB.clear_database()

    # Load documents
    print("\n📄 Loading documents...")
    documents = Document_Manager.load_documents()
    print(f"✅ Loaded {len(documents)} documents")

    # Initialize chunker
    print("\n🔪 Initializing RecursiveChunker...")
    chunker = RecursiveChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    # Split documents
    print("\n✂️  Splitting documents into chunks...")
    chunks = chunker.split_documents(documents)

    # Verify BaseChunker features
    if chunks:
        print(f"\n🔍 VERIFICATION:")
        print(f"  - First chunk ID: {chunks[0].metadata.get('id', 'MISSING')}")
        print(f"  - Chunking strategy: {chunks[0].metadata.get('chunking_strategy', 'MISSING')}")
        print(f"  - Total chunks: {len(chunks)}")

        # Check if chunk IDs follow the expected pattern
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


if __name__ == "__main__":
    main()