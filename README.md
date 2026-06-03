# 📄 RAG PDF Chatbot

A multi-document Retrieval-Augmented Generation chatbot built with Python, Streamlit, LangChain, ChromaDB, HuggingFace embeddings, and Groq.

Users can upload one or more PDF files, generate document summaries, ask questions, and receive grounded answers based on retrieved document content.

## Features

- Upload one or multiple PDF documents
- Extract and split PDF text into chunks
- Generate embeddings using HuggingFace models
- Store document vectors in ChromaDB
- Ask questions using a Groq-powered LLM
- Retrieve relevant document chunks
- View source snippets
- Maintain chat history
- Display uploaded document statistics in the sidebar
- Generate automatic document summaries

## Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- HuggingFace Sentence Transformers
- Groq API
- pypdf
- python-dotenv

## RAG Pipeline

```text
PDF Upload
↓
Text Extraction
↓
Text Chunking
↓
Embedding Generation
↓
Vector Storage in ChromaDB
↓
Semantic Retrieval
↓
Groq LLM Response
↓
Answer with Sources