# from langchain_chroma import Chroma
# from langchain_ollama import OllamaEmbeddings, ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# import argparse

# CHROMA_PATH = "chroma"

# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="The query text.")
#     args = parser.parse_args()
#     query_text = args.query_text
#     query_rag(query_text)

# def query_rag(query_text: str):
    # # 1. connect to the DB
    # db = Chroma(
    #     persist_directory=CHROMA_PATH,
    #     embedding_function=OllamaEmbeddings(model="nomic-embed-text")
    # )

    # # 2. search for the most similar chunks
    # results = db.similarity_search_with_score(query_text, k=5)

    # if not results:
    #     print("No results found.")
    #     return

    # # 3. build context from the chunks
    # context_text = "\n\n---\n\n".join(
    #     [doc.page_content for doc, _score in results]
    # )

    # # 4. build the prompt
    # prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # prompt = prompt_template.format(context=context_text, question=query_text)

    # # 5. send to Ollama LLM
    # model = ChatOllama(model="llama3.2")
    # response = model.invoke(prompt)

    # # 6. print results
    # sources = [doc.metadata.get("id", "unknown") for doc, _score in results]
    # print(f"\n✅ Answer:\n{response.content}")
    # print(f"\n📄 Sources:\n" + "\n".join(sources))

# if __name__ == "__main__":
#     main()