"""
Test script to verify BaseChunker and chunking strategies work
Run this first to ensure everything is set up correctly
"""

import sys
from pathlib import Path

# Add modules directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
from D00_chunking.base_chunker import BaseChunker
from D00_chunking.recursive_chunker import RecursiveChunker


def test_base_chunker():
    """Test if BaseChunker works"""
    print("\n" + "=" * 60)
    print("🧪 TESTING BASE CHUNKER")
    print("=" * 60)

    # Create a simple test document
    test_doc = Document(
        page_content="This is a test document. " * 50,
        metadata={"source": "test.pdf", "page": 1}
    )

    print(f"✅ Created test document: {len(test_doc.page_content)} characters")

    # Test RecursiveChunker (which inherits from BaseChunker)
    print("\n📊 Testing RecursiveChunker...")
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.split_documents([test_doc])

    # Verify BaseChunker features work
    print(f"\n✅ Split into {len(chunks)} chunks")

    # Check if calculate_chunk_ids worked
    if chunks and "id" in chunks[0].metadata:
        print(f"✅ Chunk IDs created: {chunks[0].metadata['id']}")

    # Check if chunking_strategy was recorded
    if chunks and "chunking_strategy" in chunks[0].metadata:
        print(f"✅ Strategy recorded: {chunks[0].metadata['chunking_strategy']}")

    # Test preview functionality
    print("\n📖 Testing preview_chunks method...")
    chunker.preview_chunks(chunks, num_chunks=1)

    print("\n" + "=" * 60)
    print("✅ BaseChunker test passed!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        test_base_chunker()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()