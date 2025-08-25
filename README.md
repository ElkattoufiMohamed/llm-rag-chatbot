# 🧠 RAG Chatbot: Question-Answering with a Custom Knowledge Base

This project is a hands-on implementation of **Retrieval-Augmented Generation (RAG)**. It demonstrates how to build a chatbot that can answer user questions based on a **custom document** (e.g. product manuals, Wikipedia articles, research papers), going beyond the pre-trained knowledge of the language model.

The project starts with `.txt` files as the knowledge base, and extends to **PDF support**, making it applicable to real-world use cases.

---

## 🚀 Project Goal

The goal of this project is to create a **question-answering chatbot** that:

- Retrieves relevant information from your document(s).
- Passes it as context to a language model.
- Generates **accurate, context-specific answers** instead of relying only on pre-training.

This is a practical, beginner-friendly application of **RAG** — a technique at the core of modern LLM applications.

---

## 🛠️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/ElkattoufiMohamed/llm-rag-chatbot.git
cd llm-rag-chatbot

```

### 2. Install dependencies

```bash
pip install -r requirements.txt

```

Or manually:

```bash
pip install transformers langchain faiss-cpu sentence-transformers pandas numpy jupyter torch pymupdf

```

---

## 📂 Project Structure

```
├── data/
│   ├── knowledge_base.txt   # Example TXT knowledge base
│   ├── knowledge_base.pdf   # Example PDF knowledge base
├── llm-rag-chatbot.ipynb    # Main Jupyter notebook
├── llm-rag-chatbot-pdf.ipynb 
├── README.md                # Project documentation
└── requirements.txt         # Dependencies

```

---

## 💻 How It Works

1. **Load the Knowledge Base**
    - Supports `.txt` (simple string read).
    - Supports `.pdf` (via PyMuPDF to extract page-wise text).
2. **Split into Chunks**
    - Uses `RecursiveCharacterTextSplitter` to break documents into manageable chunks (1000 chars with 200 overlap).
3. **Create Embeddings**
    - Each chunk is converted into a dense vector using `sentence-transformers/all-MiniLM-L6-v2`.
4. **Build Vector Store**
    - Store embeddings in **FAISS** for efficient similarity search.
5. **Query Processing**
    - User asks a question → retrieve top-k relevant chunks.
    - Combine chunks + question → pass to a **Small LLM** (Flan-T5).
6. **Answer Generation**
    - The LLM generates a context-aware answer.
    - (Optional) Cites the source page if PDF is used.

---

## 📑 Handling PDF Files

Unlike `.txt`, PDFs need a parser to extract text.

We use **PyMuPDF** (`fitz`) to read each page and store both the text and its page number.

```python
import fitz

def load_pdf(path):
    pages = []
    doc = fitz.open(path)
    for i in range(len(doc)):
        text = doc[i].get_text("text")
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages

```

After extraction, the text is split into chunks (just like `.txt`) while keeping **page metadata**, so the chatbot can reference the original page in its answer.

---


## 📈 Next Steps

- Support multiple PDFs as knowledge base.
- Use larger LLMs (e.g. Flan-T5-large, LLaMA) for improved answer quality.
- Deploy as a web app with **Streamlit** or **Gradio**.

---

## 🙌 Acknowledgments

- LangChain for simplifying RAG workflows.
- Hugging Face Transformers for models and pipelines.
- Sentence-Transformers for embeddings.
- [FAISS](https://github.com/facebookresearch/faiss) for fast similarity search.
- PyMuPDF for PDF parsing.

## 📜 License

This repository contains code for educational/research purposes. Please refer to the licenses of individual models and libraries for their respective terms.

## Contact

If you have any questions or feedback, feel free to reach out:

- GitHub: [ElkattoufiMohamed](https://github.com/ElkattoufiMohamed)
- Email: contact@mohamedelkattoufi.com