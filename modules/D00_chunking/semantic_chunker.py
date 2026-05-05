from langchain_experimental.text_splitter import SemanticChunker as LangchainSemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from typing import List
from .base_chunker import BaseChunker

class SemanticChunker(BaseChunker):
    """
    Splits text based on semantic similarity.
    Does NOT use fixed chunk_size or chunk_overlap.
    """
    USES_SIZE_OVERLAP = False   # Tell factory not to pass these

    def __init__(self, breakpoint_threshold_type: str = "percentile",
                 max_chunk_size_for_embedding: int = 2000):
        # No need to call super().__init__ because we don't use size/overlap
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.max_chunk_size_for_embedding = max_chunk_size_for_embedding

        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.semantic_splitter = LangchainSemanticChunker(
            self.embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type
        )
        self.pre_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size_for_embedding,
            chunk_overlap=0,   # overlap not needed for pre‑splitting; can be set if desired
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        all_chunks = []
        for doc in documents:
            safe_pieces = self.pre_splitter.split_text(doc.page_content)
            for piece in safe_pieces:
                semantic_chunks = self.semantic_splitter.split_text(piece)
                for i, text in enumerate(semantic_chunks):
                    chunk_metadata = doc.metadata.copy()
                    chunk_metadata["chunk_index"] = i
                    chunk_metadata["original_doc_id"] = doc.metadata.get("id", "unknown")
                    all_chunks.append(Document(page_content=text, metadata=chunk_metadata))
        print(f"✅ Semantic chunking: {len(documents)} docs → {len(all_chunks)} chunks")
        all_chunks = self.calculate_chunk_ids(all_chunks)
        self.preview_chunks(all_chunks)
        return all_chunks