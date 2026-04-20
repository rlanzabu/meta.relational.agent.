#!/usr/bin/env python3
"""
Allows the user to select the chunking strategy interactively.
"""
import sys
from pathlib import Path

# Add the parent directory of I00_orchestrator (i.e., the 'modules' directory) to sys.path
MODULES_DIR = Path(__file__).resolve().parent.parent  # This gives the 'modules' directory
sys.path.insert(0, str(MODULES_DIR))

# Now import from the I00_orchestrator package
from I00_orchestrator.main_docs import process_documents, CHUNKER_REGISTRY
from I00_orchestrator.main_query import run_query


def main():
    print("Starting chatbot (import version)...")
    print(f"Available chunking strategies: {', '.join(CHUNKER_REGISTRY.keys())}")

    # --- Step 1: Get chunking parameters from user ---
    reset_db = input("Reset database? (y/N): ").strip().lower() == 'y'

    strategy = input(f"Choose chunking strategy [{', '.join(CHUNKER_REGISTRY.keys())}]: ").strip()
    if not strategy:
        strategy = "recursive"
    if strategy not in CHUNKER_REGISTRY:
        print(f"❌ Unknown strategy. Using 'recursive'.")
        strategy = "recursive"

    try:
        chunk_size = int(input("Chunk size (default 800): ").strip() or "800")
        chunk_overlap = int(input("Chunk overlap (default 80): ").strip() or "80")
    except ValueError:
        print("Invalid number, using defaults.")
        chunk_size, chunk_overlap = 800, 80

    # --- Step 2: Process documents ---
    print("\n📄 Processing documents...")
    try:
        num_chunks = process_documents(
            reset=reset_db,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy=strategy
        )
        print(f"✅ Processing complete. Created {num_chunks} chunks.")
    except Exception as e:
        print(f"❌ Failed to process documents: {e}")
        sys.exit(1)

    # --- Step 3: Ask for query and run it ---
    query = input("\n💬 What would you like to ask? ")
    print("🤖 Generating answer...")
    ret_code = run_query(query)
    if ret_code != 0:
        print("❌ Failed to generate answer")
        sys.exit(1)

    print("✅ Done!")


if __name__ == "__main__":
    main()