from langchain_text_splitters import MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker as LangchainSemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from typing import List
from .base_chunker import BaseChunker


class DocumentAwareChunker(BaseChunker):
    """
    Hybrid chunker: document structure FIRST, semantic similarity SECOND.

    Pipeline per document:
        1. Structural split — cut at heading / tag / paragraph boundaries
           so each raw section is already semantically scoped.
        2. **Pre‑split** each section into embedding‑safe pieces (avoid context length error).
        3. Semantic split — run LangChain's SemanticChunker on each safe piece.
        4. Resulting chunks are returned for embedding and storage.

    Supports three modes (auto-detected or set explicitly via `doc_type`):
        "markdown"  – splits on # / ## / ### / #### headers
        "html"      – splits on <h1>…<h4> tags
        "text"      – paragraph-boundary fallback (double newlines)
    """

    USES_SIZE_OVERLAP = False

    MARKDOWN_HEADERS = [
        ("#",    "h1"),
        ("##",   "h2"),
        ("###",  "h3"),
        ("####", "h4"),
    ]

    HTML_HEADERS = [
        ("h1", "h1"),
        ("h2", "h2"),
        ("h3", "h3"),
        ("h4", "h4"),
    ]

    def __init__(
        self,
        breakpoint_threshold_type: str = "percentile",
        doc_type: str = "auto",
        max_chunk_size_for_embedding: int = 2000,   # NEW: safe size before semantic split
    ):
        # No size/overlap — don't call super().__init__
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.doc_type = doc_type.lower()
        self.max_chunk_size_for_embedding = max_chunk_size_for_embedding

        # ── Structural splitters ──────────────────────────────────────────────
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.MARKDOWN_HEADERS,
            strip_headers=False,
        )
        self.html_splitter = HTMLHeaderTextSplitter(
            headers_to_split_on=self.HTML_HEADERS,
        )

        # ── Pre‑splitter (to avoid embedding context length errors) ────────────
        self.pre_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size_for_embedding,
            chunk_overlap=0,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        # ── Semantic splitter (shared embeddings model) ───────────────────────
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.semantic_splitter = LangchainSemanticChunker(
            self.embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        all_chunks: List[Document] = []

        for doc in documents:
            mode = self._detect_mode(doc)
            structural_sections = self._structural_split(doc, mode)

            chunk_index = 0
            for section in structural_sections:
                # --- NEW: Pre‑split the section into safe pieces ---
                safe_pieces = self.pre_splitter.split_text(section.page_content)

                for piece in safe_pieces:
                    # Second stage: semantic split inside each safe piece
                    semantic_texts = self.semantic_splitter.split_text(piece)

                    for text in semantic_texts:
                        chunk_metadata = doc.metadata.copy()
                        chunk_metadata.update(section.metadata)
                        chunk_metadata["chunk_index"] = chunk_index
                        chunk_metadata["original_doc_id"] = doc.metadata.get("id", "unknown")
                        chunk_metadata["doc_type"] = mode
                        chunk_metadata["section_breadcrumb"] = self._build_breadcrumb(section.metadata)

                        all_chunks.append(Document(page_content=text, metadata=chunk_metadata))
                        chunk_index += 1

        print(
            f"✅ Document-aware + semantic chunking ({self.doc_type}): "
            f"{len(documents)} docs → {len(all_chunks)} chunks"
        )
        all_chunks = self.calculate_chunk_ids(all_chunks)
        self.preview_chunks(all_chunks)
        return all_chunks

    # ---------- Helper methods (unchanged from your original) ----------
    def _detect_mode(self, doc: Document) -> str:
        if self.doc_type != "auto":
            return self.doc_type
        source = doc.metadata.get("source", "")
        if source.endswith((".md", ".mdx")):
            return "markdown"
        if source.endswith((".html", ".htm")):
            return "html"
        content = doc.page_content.lstrip()
        if content.startswith("#"):
            return "markdown"
        if content.startswith("<"):
            return "html"
        return "text"

    def _structural_split(self, doc: Document, mode: str) -> List[Document]:
        if mode == "markdown":
            return self.markdown_splitter.split_text(doc.page_content)
        if mode == "html":
            return self.html_splitter.split_text(doc.page_content)
        return self._paragraph_split(doc.page_content)

    def _paragraph_split(self, text: str) -> List[Document]:
        MIN_SECTION_SIZE = 600
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        sections = []
        buffer = ""
        for para in paragraphs:
            candidate = (buffer + "\n\n" + para).strip() if buffer else para
            if buffer and len(candidate) >= MIN_SECTION_SIZE:
                sections.append(Document(page_content=buffer, metadata={}))
                buffer = para
            else:
                buffer = candidate
        if buffer:
            sections.append(Document(page_content=buffer, metadata={}))
        return sections

    def _build_breadcrumb(self, metadata: dict) -> str:
        parts = [metadata[key] for key in ("h1", "h2", "h3", "h4") if metadata.get(key)]
        return " > ".join(parts) if parts else "root"