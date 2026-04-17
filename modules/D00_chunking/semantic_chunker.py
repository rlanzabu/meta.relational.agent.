from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from typing import List
from .base_chunker import BaseChunker


class SemanticChunker(BaseChunker):
    """
    Splits text based on semantic similarity.
    Best for: Complex knowledge bases, understanding meaning
    Complexity: ⭐⭐⭐⭐
    Context Quality: ⭐⭐⭐⭐⭐
    Speed: ⭐⭐
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 80,
                 breakpoint_threshold_type: str = "percentile"):
        super().__init__(chunk_size, chunk_overlap)
        self.breakpoint_threshold_type = breakpoint_threshold_type

        # Initialize embeddings for semantic chunking
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Create semantic splitter
        self.semantic_splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,  # "percentile", "standard_deviation", "interquartile"
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents using semantic similarity"""
        all_chunks = []

        for doc in documents:
            # Semantic chunking works on text directly
            texts = self.semantic_splitter.split_text(doc.page_content)

            # Create Document objects from chunks
            for i, text in enumerate(texts):
                chunk_metadata = doc.metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["original_doc_id"] = doc.metadata.get("id", "unknown")

                chunk = Document(
                    page_content=text,
                    metadata=chunk_metadata
                )
                all_chunks.append(chunk)

        print(f"✅ Semantic chunking: {len(documents)} docs → {len(all_chunks)} chunks")

        # Add IDs to chunks
        all_chunks = self.calculate_chunk_ids(all_chunks)
        self.preview_chunks(all_chunks)

        return all_chunks