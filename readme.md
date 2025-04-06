# 📚 AI-Powered Study Assistant using RAG

This is a project developed as a practical exercise for the **Agents course by Hugging Face**.

The main goal is to create an **AI agent** capable of supporting the learning process through **Retrieval-Augmented Generation (RAG)**, leveraging **LlamaIndex** to ingest, index, and query personal study documents.

---

## 🎯 Project Goals

- ✅ Practice concepts from the Hugging Face *Agents* course.
- 📄 Build an intelligent assistant that can answer questions based on personal documents.
- 🤖 Integrate **RAG**, **LlamaIndex**, **ChromaDB**, and **Ollama** into a seamless workflow.
- 📚 Enhance study effectiveness by turning static notes into an interactive tool.

---

## 🛠️ Tech Stack

- [LlamaIndex](https://www.llamaindex.ai/) – document ingestion, indexing, and retrieval
- [ChromaDB](https://www.trychroma.com/) – persistent vector store
- [Ollama](https://ollama.ai/) – local LLMs for fast, private responses
- Hugging Face Embeddings (BAAI/bge-small-en-v1.5)
- Python 3.10+

---

## 🚀 How It Works

1. Documents are loaded from a specified folder.
2. They are split into sentences and converted into embeddings.
3. The embeddings are stored in ChromaDB.
4. You can ask questions in natural language and get context-aware responses powered by a local LLM.
