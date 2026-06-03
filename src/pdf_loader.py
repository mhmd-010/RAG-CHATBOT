from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_pdf(uploaded_file):
    pdf = PdfReader(uploaded_file)

    documents = []

    for page_number, page in enumerate(pdf.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            document = Document(
                page_content=page_text,
                metadata={
                    "page": page_number,
                    "source": uploaded_file.name
                }
            )

            documents.append(document)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        if "page" not in chunk.metadata or chunk.metadata["page"] is None:
            chunk.metadata["page"] = "Unknown"

        if "source" not in chunk.metadata:
            chunk.metadata["source"] = uploaded_file.name

    return chunks