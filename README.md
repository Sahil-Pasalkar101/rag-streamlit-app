
#  Retrieval-Augmented Generation (RAG) Application
Link: https://rag-app-main.streamlit.app/
A Retrieval-Augmented Generation (RAG) application that allows users to upload documents and ask questions about their content. The system retrieves relevant information from the uploaded documents and uses a Large Language Model (LLM) to generate context-aware responses.

The project combines **LangChain** for building the RAG pipeline, **Hugging Face** for text embeddings, and the **Groq API** for fast LLM inference.

## 🚀 Features

- **Document Upload:** Upload PDF and TXT files for analysis.
- **Text Processing:** Split documents into smaller chunks for efficient retrieval.
- **Semantic Search:** Generate embeddings using Hugging Face embedding models.
- **Vector Search:** Store and retrieve relevant document chunks using FAISS.
- **Context-Aware Answers:** Use retrieved document content to generate relevant responses.
- **LLM Integration:** Generate responses using the Groq API.
- **Interactive Interface:** Ask questions and explore document content through a Streamlit application.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application development |
| LangChain | RAG pipeline, document processing, and LLM integration |
| Hugging Face | Text embedding generation |
| FAISS | Vector storage and similarity search |
| Groq API | Fast inference using a supported language model |
| Streamlit | Interactive web interface |

## 🧠 How It Works

The application follows a Retrieval-Augmented Generation workflow:

1. **Document Upload:** The user uploads a PDF or TXT file.
2. **Text Extraction:** The application extracts text from the uploaded document.
3. **Text Chunking:** LangChain splits the text into smaller, manageable chunks.
4. **Embedding Generation:** Hugging Face converts the text chunks into vector embeddings.
5. **Vector Indexing:** FAISS indexes the embeddings to enable similarity-based retrieval.
6. **Query Processing:** The user's question is converted into an embedding and matched against relevant document chunks.
7. **Response Generation:** The retrieved context is passed to a language model through the Groq API.
8. **Answer Display:** The application displays the generated response through Streamlit.

## 🏗️ Architecture

```text
             User
              |
              v
      Streamlit Interface
              |
              v
       Upload PDF / TXT
              |
              v
        Text Extraction
              |
              v
       Text Chunking
         (LangChain)
              |
              v
   Hugging Face Embeddings
              |
              v
        FAISS Vector Store
              |
              v
         User Question
              |
              v
       Similarity Search
              |
              v
      Retrieved Context
              |
              v
       Groq API + LLM
              |
              v
      Contextual Answer
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

### 3. Install dependencies

If a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

Otherwise, install the main dependencies used by the project:

```bash
pip install streamlit langchain langchain-community langchain-text-splitters langchain-huggingface langchain-groq faiss-cpu sentence-transformers pypdf
```

*Note: The exact dependencies may vary depending on your implementation and installed LangChain versions.*

### 4. Configure the Groq API key

Create an API key through the [Groq Console](https://console.groq.com/).

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Load the key in Python using `python-dotenv`, or use Streamlit secrets if your application is deployed.

**Important:** Never commit API keys, passwords, or other secrets to GitHub.

Add the following to your `.gitignore` file:

```gitignore
.env
venv/
__pycache__/
.streamlit/secrets.toml
```

### 5. Run the application

If your Streamlit entry point is `main.py`, run:

```bash
streamlit run main.py
```

Open the local URL displayed in your terminal to interact with the application.

## 💡 Example Use Case

Imagine uploading a research paper, textbook, or technical document.

**User question:**
> What are the main findings discussed in this document?

**Application workflow:**
- Retrieves the most relevant text chunks from the uploaded document.
- Supplies the retrieved context to the language model.
- Generates a response based on the available document information.

This approach helps users explore lengthy documents without manually searching through every page.


## 🔮 Future Improvements

- Add conversational memory for multi-turn question answering.
- Support additional document formats.
- Display source references and page numbers alongside answers.
- Improve retrieval using metadata filtering and reranking.
- Add persistent vector storage for previously uploaded documents.
- Deploy the application to a cloud hosting platform.
- Add automated tests, logging, and error handling.


---

**Key Technologies:** Python · LangChain · Hugging Face · FAISS · Groq API · Streamlit
