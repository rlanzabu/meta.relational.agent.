from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from D00_chunking import chunking
import os
import shutil

CHROMA_PATH = "modules/50_embeds_db"

class Embedding:
    def get_embedding_function():
        return OllamaEmbeddings(model="nomic-embed-text")
    
    def add_to_chroma(chunks: list[Document]):
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=Embedding.get_embedding_function()
        )

        chunks_with_ids = chunking.calculate_chunk_ids(chunks)

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

    def clear_database():
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)