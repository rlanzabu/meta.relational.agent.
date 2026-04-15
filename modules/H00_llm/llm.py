from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

class LLM:
    def llm(context_text, query_text):
        # 4. build the prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # 5. send to Ollama LLM
        model = ChatOllama(model="llama3.2")
        response = model.invoke(prompt)
        
        return response