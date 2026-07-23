<div align="center">

# 🚀 Advanced Hybrid RAG System

### Production-Inspired Retrieval-Augmented Generation (RAG) System

Hybrid Search • BM25 • Dense Retrieval • Reciprocal Rank Fusion • Cross-Encoder Reranking • Multi-LLM Support • Conversation Memory • Streamlit

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-green.svg)
![BM25](https://img.shields.io/badge/Retrieval-BM25-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

</div>

---

# 📌 Overview

**Advanced Hybrid RAG System** is a production-inspired Retrieval-Augmented Generation application that allows users to upload documents, build a searchable knowledge base, and chat with them using modern LLMs.

Unlike basic RAG implementations, this project combines **semantic vector search**, **keyword retrieval**, **Reciprocal Rank Fusion**, and **Cross-Encoder Reranking** to significantly improve retrieval quality and reduce hallucinations.

The application supports multiple LLM providers including **OpenAI**, **Google Gemini**, and **Ollama**, while providing **citation-backed responses**, **conversation memory**, and an interactive **Streamlit** interface.

---

# ✨ Features

## 📄 Multi-format Document Ingestion

- PDF
- DOCX
- TXT
- CSV

---

## 🧠 Advanced Retrieval Pipeline

- Dense Retrieval (FAISS)
- BM25 Sparse Retrieval
- Hybrid Search
- Reciprocal Rank Fusion (RRF)
- Cross-Encoder Re-ranking
- Metadata-aware retrieval

---

## 🤖 Multiple LLM Providers

- OpenAI
- Google Gemini
- Ollama (Local Models)

---

## 💬 Conversational AI

- Context-aware chat
- Short-term memory
- Streaming responses
- Citation-backed answers
- JSON chat export

---

## 📚 Embeddings

- Sentence Transformers
- Persistent FAISS Index
- Local embedding generation

---

# 🏗️ System Architecture

```text
                     User
                       │
                       ▼
              Streamlit Interface
                       │
          Upload Documents / Ask Query
                       │
                       ▼
           Document Parsing Layer
       (PDF | DOCX | TXT | CSV)
                       │
                       ▼
               Chunking Engine
        Recursive Chunking + Metadata
                       │
                       ▼
            Embedding Generation
     SentenceTransformer Embeddings
                       │
                       ▼
                FAISS Vector Store
                       │
      ┌────────────────┴────────────────┐
      ▼                                 ▼
 Dense Semantic Search             BM25 Search
      │                                 │
      └──────────────┬──────────────────┘
                     ▼
         Reciprocal Rank Fusion
                     ▼
       Cross Encoder Re-ranking
                     ▼
          Top Relevant Chunks
                     ▼
            Prompt Construction
                     ▼
     OpenAI / Gemini / Ollama
                     ▼
       Citation-backed Response
```

---

# 🔍 Retrieval Pipeline

```text
User Query
     │
     ▼
Query Expansion
     │
     ▼
Dense Retrieval (FAISS)
     │
     ▼
Sparse Retrieval (BM25)
     │
     ▼
Reciprocal Rank Fusion
     │
     ▼
Cross Encoder Re-ranking
     │
     ▼
Context Compression
     │
     ▼
Prompt Builder
     │
     ▼
LLM
     │
     ▼
Answer + Citations
```

---

# 📂 Project Structure

```text
advanced-rag/

│
├── app.py
├── config.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── uploads/
│   ├── processed/
│   └── embeddings/
│
├── ingestion/
│   ├── pdf_loader.py
│   ├── docx_loader.py
│   ├── txt_loader.py
│   ├── csv_loader.py
│   └── parser.py
│
├── chunking/
│
├── embeddings/
│
├── retrieval/
│   ├── dense.py
│   ├── bm25.py
│   ├── hybrid.py
│   ├── query_expansion.py
│   └── reranker.py
│
├── llm/
│
├── memory/
│
└── tests/
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/raj5715567/advanced-hybrid-rag-system.git

cd advanced-hybrid-rag-system
```

---

## Create Virtual Environment

```bash
python -m venv .venv
```

Windows

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS

```bash
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment

```bash
cp .env.example .env
```

Add your API keys if needed.

```env
OPENAI_API_KEY=

GEMINI_API_KEY=

RAG_LLM_PROVIDER=gemini
```

---

## Run

```bash
streamlit run app.py
```

---


---


# 📸 Screenshots

## Home

> <img width="1917" height="1076" alt="image" src="https://github.com/user-attachments/assets/9655c683-f8ee-4306-9756-56d4565e492b" />


---

## Document Upload

> <img width="1917" height="1077" alt="image" src="https://github.com/user-attachments/assets/504a631c-0f9b-424d-b34b-c1e0598a522e" />


---

## Chat Interface

> <img width="1917" height="1078" alt="image" src="https://github.com/user-attachments/assets/020bb513-5272-43b7-bd42-e0d8292c0068" />


---

## Citation Display

><img width="1916" height="1078" alt="image" src="https://github.com/user-attachments/assets/d824ce66-17f0-4e27-b238-54805d0a273c" />


---

# 🔬 Technologies Used

| Category | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Language | Python |
| Embeddings | Sentence Transformers |
| Vector DB | FAISS |
| Sparse Retrieval | BM25 |
| Hybrid Retrieval | RRF |
| Re-ranking | Cross Encoder |
| LLM | OpenAI / Gemini / Ollama |

---

# 📈 Current Progress

| Module | Status |
|---------|--------|
| Document Parsing | ✅ |
| Chunking | ✅ |
| Embeddings | ✅ |
| FAISS | ✅ |
| Dense Retrieval | ✅ |
| BM25 | ✅ |
| Hybrid Search | ✅ |
| Reciprocal Rank Fusion | ✅ |
| Cross Encoder | ✅ |
| Conversation Memory | ✅ |
| Streaming Responses | ✅ |
| Source Citations | ✅ |
| Chat Export | ✅ |

---

# 🚀 Future Enhancements

- Parent-Child Retrieval
- Query Rewriting using LLMs
- Multi-Query Retrieval
- Contextual Compression Retriever
- GraphRAG
- Agentic RAG
- LangGraph Integration
- Knowledge Base Management
- Docker Deployment
- FastAPI Backend
- Authentication
- Cloud Deployment

---

# 🤝 Contributing

Contributions are welcome!

Feel free to fork the repository, open issues, or submit pull requests.

---

# 📜 License

Licensed under the MIT License.

---

# 👨‍💻 Author

**Raj Kumar**

GitHub

https://github.com/raj5715567
