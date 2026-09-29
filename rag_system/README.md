# Document-Grounded RAG System

A Retrieval-Augmented Generation (RAG) system that:
1. Ingests real PDF/Markdown/TXT documents.
2. Splits them into overlapping chunks.
3. Creates vector embeddings.
4. Stores the chunks in a local Chroma vector database.
5. Retrieves relevant chunks for a question.
6. Injects the retrieved chunks into an LLM prompt.
7. Produces a grounded answer with source filenames.

## Requirements
- Python 3.10+ recommended
- Ollama installed and running
- A real 10–20 page document set placed in `data/`

## Windows setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation, use:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install an Ollama model:
```powershell
ollama pull llama3.2:3b
```

Then:
```powershell
python ingest.py
python ask.py
```

## Architecture

Documents -> Text extraction -> Chunking -> Embeddings -> ChromaDB
                                                     |
User question -> Query embedding -> Similarity retrieval
                                                     |
Retrieved chunks -> Prompt -> LLM -> Grounded answer

## Assignment evidence to demonstrate

Use a question whose answer appears only in your uploaded documents.
Show:
- the retrieved source chunks,
- the final grounded answer,
- the source filename(s).

Then ask an unrelated/private-document question. The system should state that the answer was not found instead of inventing it.
