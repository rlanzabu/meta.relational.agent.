import sys
from pathlib import Path
# Add the parent directory (modules/) to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from E00_embedding import ChromaDB
from H00_llm import LLM


class NoResultsFound(Exception):
    """Raised when no results are found for a query."""
    pass

class Retriever:
    def retriever(query_text: str):
        # 1. connect to the DB
        db = ChromaDB.connect_to_chroma_db()

        # 2. search for the most similar chunks
        results = db.similarity_search_with_score(query_text, k=3)

        if not results:
            raise NoResultsFound(f"No documents found for query: '{query_text}'")

        # 3. build context from the chunks
        context_text = "\n\n---\n\n".join(
            [doc.page_content for doc, _score in results]
        )
        return context_text, results

    def print_results(results, context_text, query_text):
        # 6. print results
        response = LLM.llm(context_text, query_text)
        sources = [doc.metadata.get("id", "unknown") for doc, _score in results]
        print(f"\n✅ Answer:\n{response.content}")
        print(f"\n📄 Sources:\n" + "\n".join(sources))