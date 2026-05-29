# 📚 DocuQuery — RAG Document Analysis System

Upload any PDF and ask questions. Built with Retrieval-Augmented Generation (RAG).

## 🛠️ Tech Stack
- **Python** — Backend
- **Streamlit** — UI
- **Sentence Transformers** — Free embeddings
- **ChromaDB** — Vector database
- **PyPDF2** — PDF extraction

## 🚀 How It Works
1. Upload a PDF
2. Text is chunked into paragraphs
3. Embeddings are created (free, local)
4. Ask a question
5. System retrieves the 3 most relevant chunks
6. Results displayed with relevance scores

## ⚡ Run Locally
```bash
pip install streamlit PyPDF2 chromadb sentence-transformers
streamlit run app.py
