from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def create_vector_store(chunks, filename):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    metadatas = [
        {
            "page": chunk.metadata.get("page", "Unknown"),
            "source": chunk.metadata.get(
                "source",
                filename
            )
        }
        for chunk in chunks
    ]

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    return vectorstore