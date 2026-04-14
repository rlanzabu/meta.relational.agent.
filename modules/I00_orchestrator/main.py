import argparse
from C00_document_db_manager import document_manager
from D00_chunking import chunking
from E00_embedding import embedding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        embedding.clear_database()

    documents = document_manager.load_documents()
    chunks = chunking.split_documents(documents)
    embedding.add_to_chroma(chunks)


if __name__ == "__main__":
    main()