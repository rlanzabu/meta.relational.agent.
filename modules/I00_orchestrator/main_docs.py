import sys
from pathlib import Path

# Add modules directory to path so we can import sibling packages
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from C00_document_db_manager import Document_Manager
from D00_chunking import Chunking
from E00_embedding import Embedding, ChromaDB

def main_docs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        ChromaDB.clear_database()

    documents = Document_Manager.load_documents()
    chunks = Chunking.split_documents(documents)
    Embedding.add_to_chroma(chunks)
    print("✅ Documents processed and added to Chroma DB")


if __name__ == "__main__":
    main_docs()