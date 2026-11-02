# 🤖 AI-Powered RAG Chatbot

An intelligent **Retrieval-Augmented Generation (RAG) Chatbot** that allows users to upload PDF documents and ask questions based on their content. It uses semantic search, vector embeddings, FAISS, and a lightweight Hugging Face language model to retrieve relevant information and generate answers with source references.

## 🚀 Project Overview

The AI RAG Chatbot makes it easier to interact with PDF documents. Instead of manually searching through lengthy files, users can upload their PDFs, ask questions in natural language, and receive AI-generated answers based on the document content.

The application uses a Retrieval-Augmented Generation architecture to retrieve relevant document chunks before generating responses.

## ✨ Features

- 📄 **PDF Upload:** Upload one or multiple PDF documents.
- 📚 **Document Management:** Add and remove uploaded documents.
- 🧠 **Text Embeddings:** Convert text into numerical vector representations.
- 🔍 **Semantic Search:** Retrieve relevant document content based on meaning.
- ⚡ **FAISS Vector Database:** Efficient similarity search over document embeddings.
- 🤖 **AI-Powered Answers:** Generate answers using Hugging Face FLAN-T5 Small.
- 📌 **Source References:** Display document names and page references.
- 💬 **Interactive Chat Interface:** Ask questions through a Streamlit chat interface.
- 📊 **Document Statistics:** View uploaded PDFs, processed documents, and text chunks.
- 🆓 **Free and Local-Friendly:** Uses open-source models and tools.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web interface |
| LangChain | Document and text processing |
| Hugging Face Transformers | Language model integration |
| FLAN-T5 Small | Answer generation |
| Sentence Transformers | Text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| FAISS | Vector similarity search |
| PyPDF | PDF text extraction |

## 🏗️ Project Architecture

1. **PDF Upload:** Users upload PDF documents.
2. **Text Extraction:** Text is extracted from PDF files.
3. **Text Chunking:** Extracted text is divided into smaller chunks.
4. **Embedding Generation:** Text chunks are converted into vector embeddings.
5. **Vector Storage:** Embeddings are stored in a FAISS vector database.
6. **Semantic Retrieval:** Relevant chunks are retrieved based on the user's question.
7. **Answer Generation:** FLAN-T5 Small generates an answer using the retrieved document content.
8. **Source Display:** Relevant document references are displayed with the answer.

## 📁 Project Structure

```text
AI-RAG-Chatbot/
│
├── app.py
│
├── src/
│   ├── pdf_processor.py
│   ├── text_chunker.py
│   ├── vector_store.py
│   └── llm_engine.py
│
├── data/
├── vectorstore/
├── assets/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/maria3t378-ai/AI-RAG-Chatbot.git
```

### 2. Navigate to the Project Folder

```bash
cd AI-RAG-Chatbot
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```powershell
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 💻 How to Use

1. Launch the application.
2. Upload one or more PDF documents.
3. Click **Process All Documents**.
4. Wait for the embeddings and FAISS vector database to be created.
5. Enter a question related to your uploaded documents.
6. View the AI-generated answer and its source references.

## 🧪 Example Question

**Question:**

What are embeddings?

**Example Answer:**

Embeddings convert text, images, or other information into numerical vectors that help computers compare meaning and similarity.

## 🎯 Project Goals

- Explore Retrieval-Augmented Generation (RAG).
- Understand semantic search and vector embeddings.
- Build practical applications using open-source AI models.
- Develop an interactive PDF question-answering system.
- Improve document retrieval and AI-powered information access.

## 👩‍💻 Developer

**Maria Siddique**

AI & Machine Learning Developer

- GitHub: [@maria3t378-ai](https://github.com/maria3t378-ai)
- LinkedIn: [Maria Siddique](https://www.linkedin.com/in/maria-siddique-b18b51439/)

---

⭐ If you find this project interesting, consider giving it a star!
