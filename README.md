
# Week 2 Day 4: Multi-PDF RAG Application

## Project Overview

A Retrieval-Augmented Generation (RAG) application built using Python, LangChain, Streamlit, FAISS, and OpenAI.

Users can upload multiple PDF documents, create one searchable vector database, ask questions, and receive AI-generated answers with document filenames and page-number citations.

## Technologies Used

- Python
- Streamlit
- LangChain
- OpenAI GPT-4o-mini
- OpenAI text-embedding-3-small
- FAISS
- PyPDF

## Key Features

- Upload multiple PDF documents simultaneously.
- Split PDF content into searchable chunks.
- Store embeddings in one FAISS vector database.
- Display the number of indexed chunks.
- Ask questions across uploaded documents.
- Generate answers using retrieved context.
- Display source filenames and page numbers.
- Maintain the vector database in Streamlit session state.

## Installation

Create a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements_rag.txt
```

## API Configuration

Set your OpenAI API key:

```powershell
$env:OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

Never upload API keys to GitHub.

## Run the Application

```powershell
python -m streamlit run rag_app.py
```

Open the local URL displayed in the terminal.

## Application Workflow

1. Upload two or more PDF files.
2. Click Build / Rebuild Index.
3. Wait for FAISS indexing to complete.
4. Enter a question.
5. Click Get Answer.
6. Review the answer and source references.

## Testing Results

Tested with two assignment PDFs:

- Week2_Day3_LangChain_LangSmith.pdf
- Week2_Day4_MultiPDF_RAG_App.pdf

**Indexing results:**

- PDFs processed: 2
- Chunks indexed: 8
- Vector databases: 1

**Verification:** Successfully answered questions from both PDF documents with source filenames and page numbers.

## Author

Mohan Ramalingam

Week 2 Day 4 Assignment
