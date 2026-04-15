
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from D00_chunking import Chunking
from E00_embedding.chroma_db import ChromaDB



class Embedding:
    def get_embedding_function():
        return OllamaEmbeddings(model="nomic-embed-text")
    
    def connect_to_db():
        db = ChromaDB.connect_to_chroma_db()
        return db
    
    def add_to_chroma(chunks: list[Document]):
        db = Embedding.connect_to_db()

        chunks_with_ids = Chunking.calculate_chunk_ids(chunks)

        existing_items = db._collection.get()
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")


        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            print("✅ No new documents to add")

