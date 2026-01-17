# 📄 RAG PDF Assistant with Ollama

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![Ollama](https://img.shields.io/badge/LLM-Ollama-orange)
![FAISS](https://img.shields.io/badge/Vector_Store-FAISS-green)

**A high-performance, privacy-first desktop application that transforms static PDFs into interactive intelligence assets using local RAG.**

## 🏗️ Technical Architecture

The platform uses a **Dual-Process Flow** to ensure data stays local while maintaining a high-density intelligence output.

1.  **Ingestion Layer**: `PyPDF2` extracts text → `Sentence-Transformers` generates vector embeddings.
2.  **Storage Layer**: A local **FAISS** index stores embeddings for millisecond-latency retrieval.
3.  **Inference Layer**: **Ollama** runs local LLMs (Llama 3.2, Mistral) to synthesize retrieved context into natural language answers.

---

## ✨ Key Features

| Category | Feature | Benefit |
| :--- | :--- | :--- |
| **Privacy** | 100% Local Processing | No data leaves your machine; works in air-gapped environments. |
| **Speed** | Asynchronous RAG | Process 100+ pages in under 30 seconds. |
| **UX** | Citation Mapping | Every AI answer includes exact page references for fact-checking. |
| **Cost** | Zero API Fees | Runs entirely on your hardware; no subscription required. |

---

## 🛠️ Tech Stack

* **UI Framework**: Streamlit (Custom CSS for professional "Dark Mode")
* **Vector Database**: FAISS (Facebook AI Similarity Search)
* **Embeddings**: `all-MiniLM-L6-v2` (Fast, local transformer)
* **LLM Runtime**: Ollama (Supporting Llama 3, Mistral, Phi-3)
* **PDF Processing**: PyPDF2 / LangChain (for advanced chunking)

---

### 📊 Performance Benchmarks

| Task | Target Performance | Technical Implementation |
| :--- | :--- | :--- |
| **Ingestion Latency** | < 1s per page | Asynchronous parallel chunking |
| **Retrieval Speed** | < 50ms | Local FAISS vector similarity search |
| **Inference Time** | 10-25 tokens/sec | Optimized local LLM execution |
| **Memory Footprint** | ~4GB RAM | Efficient resource management |

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.9+**
- **Ollama Installed** ([Download here](https://ollama.com))

### 2. Installation
```bash
git clone [https://github.com/yourusername/rag-pdf-assistant.git](https://github.com/yourusername/rag-pdf-assistant.git)
cd rag-pdf-assistant
pip install -r requirements.txt
# Ensure Ollama is running in the background (Command: ollama serve)
streamlit run app.py
