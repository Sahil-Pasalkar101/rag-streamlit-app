
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory


def get_pdfs_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            text += page.extract_text() or ""

    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    return text_splitter.split_text(raw_text)


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    return vectorstore


def get_session_history(session_id: str):
    if "chat_store" not in st.session_state:
        st.session_state.chat_store = {}

    if session_id not in st.session_state.chat_store:
        st.session_state.chat_store[session_id] = (
            InMemoryChatMessageHistory()
        )

    return st.session_state.chat_store[session_id]


def get_conversation_chain(vector_store):
    llm = ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0.3
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful AI assistant that answers questions
based on the provided document context.

Use the context below to answer the user's question.

If the answer cannot be found in the provided context,
clearly say that you don't know based on the provided documents.

Context:

{context}
"""
        ),
        MessagesPlaceholder(
            variable_name="chat_history"
        ),
        (
            "human",
            "{input}"
        )
    ])

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    def rag_function(inputs):
        docs = retriever.invoke(
            inputs["input"]
        )

        formatted_context = format_docs(docs)

        messages = prompt.invoke({
            "context": formatted_context,
            "chat_history": inputs.get(
                "chat_history",
                []
            ),
            "input": inputs["input"]
        })

        response = llm.invoke(messages)

        return {
            "answer": response.content,
            "context": docs
        }

    rag_chain = RunnableLambda(
        rag_function
    )

    conversation_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return conversation_chain


def main():
    load_dotenv()

    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="📚"
    )

    st.title("📚 RAG Chatbot")
    st.caption("Ask questions about your PDF documents using Groq")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input(
        "Ask a question about your documents..."
    )

    if user_question:
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.markdown(user_question)

        if st.session_state.conversation:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.conversation.invoke(
                        {"input": user_question},
                        config={
                            "configurable": {
                                "session_id": "default_user"
                            }
                        }
                    )

                    answer = response["answer"]

                    st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        else:
            with st.chat_message("assistant"):
                st.warning(
                    "Please upload and process documents first."
                )

    with st.sidebar:
        st.header("Documents")

        pdf_docs = st.file_uploader(
            "Upload your PDF documents",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button(
            "Process Documents",
            use_container_width=True
        ):
            if not pdf_docs:
                st.warning(
                    "Please upload at least one PDF."
                )
                return

            with st.spinner("Processing documents..."):
                raw_text = get_pdfs_text(
                    pdf_docs
                )

                text_chunks = get_text_chunks(
                    raw_text
                )

                vector_store = get_vectorstore(
                    text_chunks
                )

                st.session_state.conversation = (
                    get_conversation_chain(
                        vector_store
                    )
                )

                st.session_state.messages = []

                st.success(
                    "Documents processed successfully!"
                )


if __name__ == "__main__":
    main()

