from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
)
from pathlib import Path
from typing import List
from langchain_core.documents import Document

# DATA_PATH is relative to the workspace root
DATA_PATH = Path(__file__).resolve().parent.parent / "B00_document_db"


class Document_Manager:
    @staticmethod
    def load_documents() -> List[Document]:
        """Load all supported documents from DATA_PATH.
        Supported: .pdf, .md, .txt, .html, .htm
        """
        all_documents = []
        supported_extensions = {
            ".pdf": PyPDFLoader,
            ".md": UnstructuredMarkdownLoader,
            ".txt": TextLoader,
            ".html": UnstructuredHTMLLoader,
            ".htm": UnstructuredHTMLLoader,
        }

        # Collect all files with supported extensions
        for ext, LoaderClass in supported_extensions.items():
            for file_path in DATA_PATH.glob(f"*{ext}"):
                try:
                    loader = LoaderClass(str(file_path))
                    documents = loader.load()
                    all_documents.extend(documents)
                    print(f"✅ Loaded {len(documents)} documents from {file_path.name}")
                except Exception as e:
                    print(f"⚠️ Failed to load {file_path.name}: {e}")

        print(f"\n📊 Total documents loaded: {len(all_documents)}")
        return all_documents