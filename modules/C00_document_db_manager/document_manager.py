from langchain_community.document_loaders import PyPDFDirectoryLoader

DATA_PATH = "modules/10_document_db"

class Document_Manager:
    def load_documents():
        document_loader = PyPDFDirectoryLoader(DATA_PATH)
        documents = document_loader.load()
        print(f"✅ Loaded {len(documents)} pages (documents) from PDF")
        return documents
    
