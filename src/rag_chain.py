from langchain_groq import ChatGroq


def generate_answer(question, retriever):
    docs = retriever.invoke(question)

    context_parts = []

    for doc in docs:
        page = doc.metadata.get("page", "Unknown")

        context_parts.append(
            f"""
Page: {page}

{doc.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    prompt = f"""
You are a helpful assistant.

Use ONLY the supplied context.

If the answer is not found in the context, say:
"I could not find the answer in the uploaded document."

When possible, mention the page number where the answer was found.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content, docs