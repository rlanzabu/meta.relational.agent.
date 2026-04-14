from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class Chunking:
    def split_documents(documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Split into {len(chunks)} chunks")

        # Preview the first chunk
        print("\n--- FIRST CHUNK PREVIEW ---")
        print(f"Source: {chunks[30].metadata.get('source')}")
        print(f"Page:   {chunks[30].metadata.get('page')}")
        print(f"Text:\n{chunks[30].page_content}")
        print("---------------------------\n")

        return chunks

    def calculate_chunk_ids(chunks):
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id
            chunk.metadata["id"] = chunk_id

        return chunks