# 🚀 AI RAG Knowledge Assistant (CLI)

An **AI-powered Retrieval-Augmented Generation (RAG) system** that enables intelligent question answering over PDF documents using semantic search and a local Large Language Model.

---

## 🧠 Overview

This project implements a **complete end-to-end RAG pipeline** that combines:

* 📄 Document ingestion (PDF)
* ✂️ Intelligent chunking
* 🔎 Vector similarity search
* 🤖 Local LLM inference
* 📚 Source-aware responses

The system runs entirely **locally**, ensuring:

* 🔒 Privacy (no external APIs required)
* ⚡ Fast response time
* 💰 Zero cost inference

RAG systems enhance LLM accuracy by retrieving relevant context before generating answers, improving factual correctness. ([Medium][1])

---

## ✨ Features

* 📂 Multi-PDF document support
* ✂️ Recursive chunking with overlap
* 🔎 Semantic search using FAISS
* 🧠 Embeddings via Sentence Transformers
* 🤖 Local LLM using Ollama (Llama3)
* 💬 CLI-based conversational interface
* 🧠 Conversation memory
* 📚 Source citation (document tracking)
* 💾 Persistent vector database (FAISS save/load)

---

## 🏗️ Architecture

```
User Query
    ↓
Embedding Model
    ↓
FAISS Vector Search
    ↓
Top-K Relevant Chunks
    ↓
Prompt + Context
    ↓
LLM (Llama3 via Ollama)
    ↓
Final Answer + Sources
```

---

## 🧰 Tech Stack

* Python
* FAISS (Vector Database)
* Sentence Transformers (Embeddings)
* Ollama (Local LLM Runtime)
* Llama3 (Language Model)
* LangChain (Document Processing)

---

## 📁 Project Structure

```
project1/
│
├── app/
│   ├── ingestion.py        # Load PDFs
│   ├── embeddings.py       # Generate embeddings
│   ├── retriever.py        # FAISS logic
│   ├── rag_pipeline.py     # Main pipeline
│
├── cli/
│   └── chat_cli.py         # CLI interface
│
├── data/
│   └── documents/          # PDF files
│
├── vectorstore/            # Saved FAISS index
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/abdullaabdulraoof/rag-knowledge-assistant-cli.git
cd rag-knowledge-assistant-cli
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python -m cli.chat_cli
```

---

## 💬 Example Usage

```
Ask: What is Artificial Intelligence?

Answer:
Artificial Intelligence is a branch of computer science focused on creating systems capable of performing tasks that require human intelligence.

Sources:
- data/documents/ai.pdf
```

---

## 🧠 Key Concepts Implemented

* Retrieval-Augmented Generation (RAG)
* Vector Embeddings
* Semantic Search
* Top-K Retrieval
* Context Injection
* Prompt Engineering

---

## 🚀 Future Improvements

* 🌐 Web UI (Next.js)
* ⚡ FastAPI backend
* 🔍 Hybrid search (BM25 + Vector)
* 📊 RAG evaluation metrics
* 📡 Streaming responses
* 📁 Multi-format support (DOCX, TXT, HTML)

---

## 📊 Why This Project Matters

RAG systems are widely used in:

* AI assistants
* Enterprise knowledge systems
* ChatGPT-like applications
* Document search engines

They solve a key limitation of LLMs:
👉 **Hallucination by grounding responses in real data**

---

## 👨‍💻 Author

**Abdulla Abdul Raoof**

* AI Engineer | Computer Vision | RAG Systems
* Focused on building real-world AI applications

---

## ⭐ If you found this useful

Give this repo a ⭐ and connect with me!

[1]: https://medium.com/%40sainathmitalakar/building-a-devops-rag-assistant-bridging-ai-and-real-time-knowledge-access-0a231717918b?utm_source=chatgpt.com "Building a DevOps RAG Assistant: Bridging AI and Real-Time Knowledge Access | by Sainath Shivaji Mitalakar | Medium"
