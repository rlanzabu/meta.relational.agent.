from abc import ABC, abstractmethod
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies"""

    # Indicates whether this chunker uses fixed chunk size and overlap
    USES_SIZE_OVERLAP = True  # Most chunkers do

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks using the specific strategy"""
        pass

    def calculate_chunk_ids(self, chunks: List[Document]) -> List[Document]:
        """Calculate and assign unique IDs to each chunk"""
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", 0)
            current_page_id = f"{source}:{page}"

            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id
            chunk.metadata["id"] = chunk_id
            chunk.metadata["chunking_strategy"] = self.__class__.__name__

        return chunks

    def preview_chunks(self, chunks: List[Document], num_chunks: int = 1):
        """Preview the first few chunks"""
        if not chunks:
            print("No chunks to preview.")
            return

        for i in range(min(num_chunks, len(chunks))):
            print(f"\n--- CHUNK {i + 1} PREVIEW ---")
            print(f"Source: {chunks[i].metadata.get('source')}")
            print(f"Page: {chunks[i].metadata.get('page')}")
            print(f"Strategy: {chunks[i].metadata.get('chunking_strategy')}")
            print(f"Text length: {len(chunks[i].page_content)} chars")
            print(f"Text preview:\n{chunks[i].page_content[:500]}...")
            print("---------------------------\n")