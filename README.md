# KB Assistant Backend

This backend is part of a document understanding system that act as an intelligent assistant to answer questions related to a set of documents.

## 🚀 Features

- Answering question related to documents
- Allow cross document information extraction
- Retrieve relevant document along with generated responses

## 🧱 Architecture Overview

1. **Ingestion Layer**
   - Uses LlamaParse for OCR + Vision + LLM processing
   - Outputs Markdown to preserve tables and formatting

2. **Cleaning Layer**
   - Regex applied to clean headers/footers

3. **Chunking Strategy**
   - Document-level chunks ensure contextual integrity

4. **Vectorization**
   - MistralAI Embedding Model (8K context, scalable, low-cost)

5. **Retrieval**
   - Hybrid search (semantic + keyword)
   - Postprocessor filters documents below relevancy threshold

6. **Response Generation**
    - Generate response
    - Attached relevant documents along with generated response

## 🔧 Technologies Used

- Python
- FastAPI 
- LLamaIndex and LlamaParse
- OpenAI
- MistralAI
- ChromaDB
- PostgreSQL
