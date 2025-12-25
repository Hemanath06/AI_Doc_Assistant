<p align="center">
  <h1 align="center">🤖 AI Doc Assistant</h1>
  <p align="center">
    <strong>Intelligent Document Q&A powered by RAG and LLM</strong>
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="#configuration">Configuration</a> •
    <a href="#architecture">Architecture</a>
  </p>
</p>

---

## 📖 Overview

**AI Doc Assistant** is an intelligent document assistant that leverages Retrieval-Augmented Generation (RAG) to provide accurate, context-aware answers from your documents. Built with Streamlit for an elegant UI and powered by LLaMA 3.3 70B via Groq API, it delivers fast, reliable responses while maintaining conversation memory.

### Why AI Doc Assistant?

- 🎯 **Accurate Answers** - Retrieves relevant document chunks and generates precise responses
- ⚡ **Fast Processing** - Incremental document processing with intelligent caching
- 💬 **Contextual Memory** - Remembers conversation context within sessions
- 🎨 **Modern UI** - Elegant dark theme with responsive design

---

## ✨ Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **RAG-Powered Q&A** | Semantic search with FAISS vector database for accurate retrieval |
| **Multi-Format Support** | Process PDF, DOCX, CSV, XLSX, and TXT files |
| **Conversation Memory** | Context-aware responses with session-based memory |
| **Incremental Processing** | Only processes new/changed documents, saving time |
| **Debug Mode** | View retrieved chunks for transparency and debugging |
| **Strict Relevance Mode** | Filter out low-relevance answers for accuracy |

### Smart Document Processing

```
📂 sop_documents/
├── doc1.pdf (unchanged) → ⚡ Load from cache
├── doc2.docx (unchanged) → ⚡ Load from cache  
├── doc3.pdf (NEW)        → 🔄 Process only this
└── subfolder/
    └── doc4.xlsx (MODIFIED) → 🔄 Process only this
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.9 or higher
- [Groq API Key](https://console.groq.com/) (free tier available)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-doc-assistant.git
   cd ai-doc-assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API key**
   
   Update `config.json` with your Groq API key:
   ```json
   {
       "model": "llama-3.3-70b-versatile",
       "grok_api_key": "your_groq_api_key_here",
       "documents_folder": "sop_documents",
       "session_memory_file": "session_memories.json"
   }
   ```

5. **Add your documents**
   
   Place your documents in the `sop_documents/` folder

6. **Run the application**
   ```bash
   streamlit run main.py
   ```

7. **Open your browser**
   
   Navigate to `http://localhost:8501`

---

## 🚀 Usage

### Adding Documents

1. **Backend Upload**: Place files directly in `sop_documents/` folder
2. **UI Upload**: Use the sidebar uploader for quick additions

### Asking Questions

Simply type your question in the chat input. The AI will:
1. Search your documents for relevant information
2. Generate a contextual response
3. Remember the conversation for follow-up questions

### Debug Mode

Enable **Debug Mode** in the sidebar to:
- View retrieved document chunks
- See chunk metadata (source, type, size)
- Analyze retrieval quality

### Strict Relevance Mode

Enable **Strict Relevance Mode** to:
- Only show answers when highly relevant chunks are found
- Reduce hallucination and inaccurate responses

---

## ⚙️ Configuration

### config.json

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | LLM model to use | `llama-3.3-70b-versatile` |
| `grok_api_key` | Your Groq API key | Required |
| `documents_folder` | Document storage path | `sop_documents` |
| `session_memory_file` | Memory storage file | `session_memories.json` |

### Supported Document Types

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF | `.pdf` | Text extraction with PyMuPDF |
| Word | `.docx` | Full document parsing |
| Excel | `.xlsx` | Tabular data extraction |
| CSV | `.csv` | Structured data support |
| Text | `.txt` | Plain text files |

---

## �️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                              │
│                    (main.py - User Interface)                    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    Analysis Layer                                │
│              (analysis.py - RAG Chain & Query)                   │
│  • Query Enhancement  • Chain Creation  • Relevance Validation  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    Utilities Layer                               │
│           (utils.py - Document Processing & Memory)              │
│  • Document Loading  • Chunking  • Vector Store  • Caching      │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    External Services                             │
│         Groq API (LLaMA 3.3)  •  FAISS Vector DB                │
└─────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
ai-doc-assistant/
├── main.py                    # Streamlit application entry point
├── analysis.py                # RAG chain and query processing
├── utils.py                   # Document processing utilities
├── config.json                # Application configuration
├── requirements.txt           # Python dependencies
├── sop_documents/             # Document storage directory
│   └── (your documents)
├── vectorstore/               # FAISS vector database (auto-generated)
├── processed_files_info.json  # Document tracking cache (auto-generated)
└── README.md                  # This file
```

