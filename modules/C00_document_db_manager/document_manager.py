from langchain_community.document_loaders import PyPDFDirectoryLoader
from pathlib import Path

# DATA_PATH should be relative to the workspace root
DATA_PATH = Path(__file__).resolve().parent.parent / "B00_document_db"

class Document_Manager:
    def load_documents():
        document_loader = PyPDFDirectoryLoader(str(DATA_PATH))
        documents = document_loader.load()
        print(f"✅ Loaded {len(documents)} pages (documents) from PDF")
        return documents
    
