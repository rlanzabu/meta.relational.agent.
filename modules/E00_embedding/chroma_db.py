from langchain_chroma import Chroma
import os
import shutil

CHROMA_PATH = "modules/F00_embeds_db/CHROMA"

class ChromaDB:
    def connect_to_chroma_db():
        from E00_embedding.embedding import Embedding
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=Embedding.get_embedding_function()
            )
        return db
    
    def clear_database():
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)