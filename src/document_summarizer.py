from langchain_groq import ChatGroq


def summarize_document(chunks):
    if not chunks:
        return "No readable text was extracted from this document."

    content = "\n".join(
        [chunk.page_content for chunk in chunks[:10]]
    )

    if len(content.strip()) < 500:
        return "Not enough readable document content was extracted to generate a useful summary."

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    prompt = f"""
You are a document analyst.

Use ONLY the document content below.
Do not guess from the filename.
Do not say you cannot access the document.

If the content is unclear, say:
"The extracted text is not clear enough to summarize."

Provide:
1. Document type
2. Main topic
3. Key points
4. Important requirements or deadlines
5. One-sentence summary

Document content:
{content}
"""

    response = llm.invoke(prompt)

    return response.content