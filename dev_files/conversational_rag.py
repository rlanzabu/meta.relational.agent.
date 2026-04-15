from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import argparse
import json
from datetime import datetime
from pathlib import Path

CHROMA_PATH = "chroma"
HISTORY_FILE = "conversation_history.json"

# New prompt template that includes conversation history
CONVERSATIONAL_PROMPT = """
You are a helpful assistant. Use the conversation history and context to answer.

Conversation history:
{history}

Relevant context from documents:
{context}

Current question: {question}

Answer naturally, referring to previous questions if relevant.
"""


class ConversationalRAG:
    def __init__(self):
        """Initialize with database connection and empty history"""
        self.db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=OllamaEmbeddings(model="nomic-embed-text")
        )
        self.model = ChatOllama(model="llama3.2")
        self.history = []  # Stores conversation
        self.prompt_template = ChatPromptTemplate.from_template(CONVERSATIONAL_PROMPT)

        # Load previous history if exists
        self.load_history()

    def add_to_history(self, question: str, answer: str):
        """Store conversation in memory and file"""
        self.history.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
        # Keep only last 10 exchanges to avoid context overflow
        if len(self.history) > 10:
            self.history = self.history[-10:]
        self.save_history()

    def save_history(self):
        """Save history to file for persistence"""
        with open(HISTORY_FILE, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_history(self):
        """Load previous conversation history"""
        if Path(HISTORY_FILE).exists():
            with open(HISTORY_FILE, 'r') as f:
                self.history = json.load(f)
            print(f"📚 Loaded {len(self.history)} previous exchanges")

    def format_history(self) -> str:
        """Format conversation history for the prompt"""
        if not self.history:
            return "No previous conversation."

        formatted = []
        for exchange in self.history[-5:]:  # Last 5 exchanges
            formatted.append(f"User: {exchange['question']}")
            formatted.append(f"Assistant: {exchange['answer']}")
        return "\n".join(formatted)

    def query(self, question: str) -> dict:
        """Process a query with conversation context"""
        # 1. Search for relevant documents
        results = self.db.similarity_search_with_score(question, k=5)

        if not results:
            return {"answer": "No relevant documents found.", "sources": []}

        # 2. Build context
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

        # 3. Get conversation history
        history_text = self.format_history()

        # 4. Generate response with history
        prompt = self.prompt_template.format(
            history=history_text,
            context=context_text,
            question=question
        )
        response = self.model.invoke(prompt)

        # 5. Store in history
        self.add_to_history(question, response.content)

        # 6. Return results
        sources = [doc.metadata.get("id", "unknown") for doc, _ in results]
        return {
            "answer": response.content,
            "sources": sources
        }

    def clear_history(self):
        """Reset conversation memory"""
        self.history = []
        self.save_history()
        print("✨ Conversation history cleared")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--clear", action="store_true", help="Clear conversation history")
    args = parser.parse_args()

    rag = ConversationalRAG()

    if args.clear:
        rag.clear_history()

    result = rag.query(args.query_text)
    print(f"\n✅ Answer:\n{result['answer']}")
    print(f"\n📄 Sources:\n" + "\n".join(result['sources']))


if __name__ == "__main__":
    main()