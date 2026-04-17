# Change this:
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# To this:
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from .base_chunker import BaseChunker


class RecursiveChunker(BaseChunker):
    """Recursive chunking strategy - best for structured documents"""

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents using recursive character text splitter"""

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)
        print(f"✅ Recursive chunking: {len(documents)} docs → {len(chunks)} chunks")

        # Use BaseChunker's methods
        chunks = self.calculate_chunk_ids(chunks)
        self.preview_chunks(chunks)

        return chunks