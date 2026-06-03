import streamlit as st
from dotenv import load_dotenv

from src.pdf_loader import load_and_split_pdf
from src.vector_store import create_vector_store
from src.rag_chain import generate_answer
from src.document_summarizer import summarize_document

load_dotenv()

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 RAG PDF Chatbot")
st.write("Upload one or more PDFs and ask questions about them.")

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    all_chunks = []
    document_stats = []

    st.subheader("📄 Document Summaries")

    for uploaded_file in uploaded_files:
        chunks = load_and_split_pdf(uploaded_file)
        all_chunks.extend(chunks)

        pages = set()

        for chunk in chunks:
            page = chunk.metadata.get("page")
            if page != "Unknown":
                pages.add(page)

        document_stats.append(
            {
                "name": uploaded_file.name,
                "chunks": len(chunks),
                "pages": len(pages)
            }
        )

        with st.expander(uploaded_file.name):
            summary = summarize_document(chunks)
            st.write(summary)

    with st.sidebar:
        st.header("📄 Uploaded Documents")

        for doc in document_stats:
            st.markdown(f"**{doc['name']}**")
            st.write(f"Pages: {doc['pages']}")
            st.write(f"Chunks: {doc['chunks']}")
            st.divider()

    st.success(f"{len(uploaded_files)} PDF(s) processed successfully!")

    vectorstore = create_vector_store(
        all_chunks,
        "multiple_documents"
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask a question about your PDF",
        key="main_chat_input"
    )

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        answer, docs = generate_answer(
            question,
            retriever
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

        with st.expander("View Sources"):
            for i, doc in enumerate(docs, start=1):
                page = doc.metadata.get("page", "Unknown")
                source = doc.metadata.get("source", "Unknown")

                st.markdown(
                    f"### Source {i} | Page {page} | {source}"
                )

                st.write(doc.page_content[:500])
                st.divider()
else:
    st.info("Upload one or more PDF files to begin.")