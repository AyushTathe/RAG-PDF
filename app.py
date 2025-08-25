# File: app.py

import io
import os
import time
import streamlit as st
from dotenv import load_dotenv

from config import *
from utils.helpers import (
    compute_file_sha1,
    persist_uploaded_pdf,
    read_pdf_pages_text,
    chunk_documents,
    get_vectorstore_cached,
    create_conversational_chain,
    build_memory,
)

# -----------------------------
# App bootstrap
# -----------------------------
load_dotenv()  # Load .env if present

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📄",
    layout="wide",
)

# -----------------------------
# Initialize session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for key in ["vectorstore", "chain", "memory", "pdf_hash", "pdf_filename", "hf_token"]:
    if key not in st.session_state:
        st.session_state[key] = None


# -----------------------------
# Sidebar: API key & settings
# -----------------------------
with st.sidebar:
    st.title(SIDEBAR_TITLE)

    hf_token_input = st.text_input(
        "Hugging Face Access Token",
        type="password",
        placeholder="(will override .env if provided)",
        help="Paste your Hugging Face access token here to authenticate with private models.",
    )

    if hf_token_input:
        st.session_state.hf_token = hf_token_input
        os.environ["HF_TOKEN"] = hf_token_input
    elif os.environ.get(ENV_HF_TOKEN):
        st.session_state.hf_token = os.environ[ENV_HF_TOKEN]

    st.divider()

    st.subheader("Model & Chunking Settings")
    st.session_state.llm_model_name = st.selectbox(
        "LLM Model for Generation",
        options=[DEFAULT_LLM_MODEL, "distilgpt2", "gpt2-medium"],
        help="Select a Hugging Face model for text generation.",
    )

    st.session_state.embedding_model_name = st.text_input(
        "Embedding Model",
        value=DEFAULT_MODEL_NAME,
        help="The Hugging Face model used to create document embeddings.",
    )
    st.session_state.temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_TEMPERATURE,
        step=0.1,
        help="Adjusts the randomness of the LLM's output. Higher values are more creative.",
    )
    st.session_state.top_k = st.slider(
        "Top K documents",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K,
        step=1,
        help="The number of relevant document chunks to retrieve for the LLM.",
    )
    st.session_state.chunk_size = st.number_input(
        "Chunk Size",
        value=CHUNK_SIZE,
        min_value=100,
        step=50,
        help="The number of characters per document chunk.",
    )
    st.session_state.chunk_overlap = st.number_input(
        "Chunk Overlap",
        value=CHUNK_OVERLAP,
        min_value=0,
        step=50,
        help="The number of overlapping characters between chunks.",
    )
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type="pdf",
        help="Upload a PDF file to enable Q&A.",
    )

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        file_hash = compute_file_sha1(file_bytes)

        if file_hash != st.session_state.get("pdf_hash"):
            with st.spinner("Processing PDF..."):
                st.session_state.pdf_hash = file_hash
                st.session_state.pdf_filename = uploaded_file.name

                # Process PDF and create vector store
                pages_text = read_pdf_pages_text(io.BytesIO(file_bytes))
                docs = chunk_documents(
                    pages_text=pages_text,
                    source_filename=st.session_state.pdf_filename,
                    chunk_size=st.session_state.chunk_size,
                    chunk_overlap=st.session_state.chunk_overlap,
                )
                
                # Use a unique collection name for the vector store
                collection_name = f"rag-{file_hash}"
                persist_path = os.path.join(CHROMA_PARENT_DIR, collection_name)

                st.session_state.vectorstore = get_vectorstore_cached(
                    docs=docs,
                    collection_name=collection_name,
                    persist_directory=persist_path,
                )

                # Get the actual number of documents from the vector store
                doc_count = st.session_state.vectorstore.get()['ids']
                effective_top_k = min(st.session_state.top_k, len(doc_count))
                
                st.session_state.chain = create_conversational_chain(
                    st.session_state.vectorstore,
                    embedding_model_name=st.session_state.embedding_model_name,
                    llm_model_name=st.session_state.llm_model_name,
                    temperature=st.session_state.temperature,
                    top_k=effective_top_k,  # Pass the adjusted top_k value
                    hf_token=st.session_state.hf_token,
                )
                
                st.session_state.messages = []
                st.success("PDF processed successfully! You can now ask questions.")
        
# -----------------------------
# Main content area
# -----------------------------
st.title(APP_TITLE)
st.write(f"Document: {st.session_state.pdf_filename}" if st.session_state.pdf_filename else "Please upload a PDF in the sidebar to begin.")

# Display chat messages
chat_container = st.container()
for message in st.session_state.messages:
    with chat_container.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user question
question = st.chat_input("Ask a question about the document...")
if question:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    with chat_container.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with chat_container.chat_message("assistant"):
        placeholder = st.empty()
        try:
            start = time.time()
            result = st.session_state.chain.invoke({"question": question})
            answer = result.get("answer") or result.get("result") or "(No answer returned.)"

            # Show sources
            source_docs = result.get("source_documents", [])
            citations = []
            for i, d in enumerate(source_docs, start=1):
                src = d.metadata.get("source", "document")
                page = d.metadata.get("page", "?")
                citations.append(f"[{i}] {os.path.basename(src)} (p.{page})")
            citation_text = "\n\n**Sources**\n" + "\n".join(citations) if citations else ""

            elapsed = time.time() - start
            placeholder.markdown(answer + citation_text)
            st.caption(f"Answered in {elapsed:.2f}s")

            st.session_state.messages.append({"role": "assistant", "content": answer + citation_text})

        except Exception as e:
            placeholder.error(f"Error generating answer: {e}")

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption(
    "This system uses Retrieval-Augmented Generation (RAG) with ChromaDB and HuggingFace embeddings "
    "to provide a conversational interface for your PDF documents."
)
