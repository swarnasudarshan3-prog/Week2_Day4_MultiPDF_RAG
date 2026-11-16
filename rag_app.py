"""Week 2 Day 4: Multi-PDF retrieval-augmented generation with source pages.

Run: python -m streamlit run rag_app.py
Requires OPENAI_API_KEY in the current process environment.
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


st.set_page_config(page_title="Multi-PDF RAG | Week 2 Day 4", page_icon="📚", layout="wide")
st.title("📚 Multi-PDF RAG Assistant")
st.caption("Ask questions across several PDFs and see the supporting file and page for each answer.")

if "store" not in st.session_state:
    st.session_state.store = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


with st.sidebar:
    st.header("1. Upload PDFs")
    uploads = st.file_uploader(
        "Select two or more PDFs (or one for testing)",
        type=["pdf"],
        accept_multiple_files=True,
    )
    build = st.button("Build / Rebuild Index", type="primary", use_container_width=True)
    if st.button("Clear Index", use_container_width=True):
        st.session_state.store = None
        st.session_state.chunk_count = 0
        st.session_state.indexed_files = []
        st.success("Index cleared.")
    st.divider()
    st.write("**Embedding model:** text-embedding-3-small")
    st.write("**Answer model:** gpt-4o-mini")
    st.write("**Chunk size:** 1000 characters; overlap: 200")
    st.caption("Uploads are indexed in memory for this Streamlit session. API requests send document text to OpenAI.")


if build:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is missing. Set it in the same terminal before starting Streamlit.")
    elif not uploads:
        st.warning("Upload at least one PDF before building an index.")
    else:
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = []
            loaded_names = []
            skipped_files = []
            with st.spinner("Reading PDFs, splitting pages, and creating one FAISS index..."):
                for index, uploaded in enumerate(uploads):
                    # Temp files allow PyPDFLoader to read Streamlit uploads.
                    # Prefix is unique, but file metadata is set to the original upload name.
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        temp_path = tmp.name
                        tmp.write(uploaded.getvalue())
                    try:
                        pages = PyPDFLoader(temp_path).load()
                        for page in pages:
                            page.metadata["source"] = uploaded.name
                            page.metadata["page_number"] = int(page.metadata.get("page", 0)) + 1
                            page.metadata["upload_index"] = index
                        file_chunks = [c for c in splitter.split_documents(pages) if c.page_content.strip()]
                        if file_chunks:
                            chunks.extend(file_chunks)
                            loaded_names.append(uploaded.name)
                        else:
                            skipped_files.append(f"{uploaded.name}: no extractable text (possibly scanned)")
                    except Exception as exc:
                        skipped_files.append(f"{uploaded.name}: {exc}")
                    finally:
                        Path(temp_path).unlink(missing_ok=True)
                if not chunks:
                    raise ValueError("None of the PDFs contained extractable text. Scanned PDFs may need OCR.")
                # One shared store for chunks from ALL uploaded PDFs.
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                new_store = FAISS.from_documents(chunks, embeddings)
                # Replace the old index only AFTER a successful rebuild.
                st.session_state.store = new_store
                st.session_state.chunk_count = len(chunks)
                st.session_state.indexed_files = loaded_names
            st.success(f"Indexed {len(chunks)} chunks from {len(loaded_names)} PDF(s) in ONE FAISS database.")
            if skipped_files:
                st.warning("Skipped: " + " | ".join(skipped_files))
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")


if st.session_state.store is not None:
    st.info(
        f"Ready: **{st.session_state.chunk_count} chunks** from "
        + ", ".join(f"`{name}`" for name in st.session_state.indexed_files)
    )
else:
    st.info("Upload PDFs and click **Build / Rebuild Index** to begin.")

st.header("2. Ask a question")
question = st.text_input("Question about the uploaded documents", placeholder="What do these documents say about ...?")
ask = st.button("Get Answer", disabled=st.session_state.store is None or not question.strip())

if ask:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is missing.")
    else:
        try:
            with st.spinner("Searching across all uploaded PDFs and drafting an answer..."):
                retriever = st.session_state.store.as_retriever(search_kwargs={"k": 6})
                docs = retriever.invoke(question.strip())
                if not docs:
                    st.warning("No matching text was retrieved.")
                    st.stop()
                context_parts = []
                for i, doc in enumerate(docs, start=1):
                    source = doc.metadata.get("source", "Unknown PDF")
                    page = doc.metadata.get("page_number", int(doc.metadata.get("page", 0)) + 1)
                    context_parts.append(f"[{i}] File: {source} | Page: {page}\n{doc.page_content}")
                context = "\n\n".join(context_parts)
                messages = [
                    (
                        "system",
                        "You are a careful document question-answering assistant. "
                        "Use ONLY the retrieved excerpts below. Do not use outside knowledge. "
                        "If there is insufficient evidence, say 'I cannot find that in the uploaded PDFs.' "
                        "If sources conflict, explicitly explain the disagreement and cite both. "
                        "Cite claims inline using excerpt IDs such as [1] or [2]. "
                        "Do not cite excerpts that do not support the claim.\n\n"
                        "Retrieved excerpts:\n" + context,
                    ),
                    ("human", question.strip()),
                ]
                response = ChatOpenAI(model="gpt-4o-mini", temperature=0).invoke(messages)
            st.subheader("Answer")
            st.write(response.content)
            st.subheader("Sources retrieved")
            st.caption("These are the retrieved excerpts; refer to the [number] citations in the answer for supporting evidence.")
            for i, doc in enumerate(docs, start=1):
                source = doc.metadata.get("source", "Unknown PDF")
                page = doc.metadata.get("page_number", int(doc.metadata.get("page", 0)) + 1)
                with st.expander(f"[{i}] {source} — page {page}"):
                    st.write(doc.page_content)
        except Exception as exc:
            st.error(f"Question answering failed: {exc}")

st.caption("Note: PDF page numbers refer to physical PDF pages (starting at 1), not printed page labels.")
