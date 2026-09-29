"""
Ingest PDFs / Markdown / TXT files into ChromaDB.

Run:
    python ingest.py
"""

from pathlib import Path
import hashlib
import re

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import DATA_DIR, DB_DIR, COLLECTION_NAME, CHUNK_WORDS, OVERLAP_WORDS


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)

    if path.suffix.lower() in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    return ""


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str):
    words = text.split()
    chunks = []
    step = max(1, CHUNK_WORDS - OVERLAP_WORDS)

    for start in range(0, len(words), step):
        chunk = words[start:start + CHUNK_WORDS]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if start + CHUNK_WORDS >= len(words):
            break

    return chunks


def main():
    data_path = Path(DATA_DIR)
    supported = {".pdf", ".md", ".txt"}
    source_files = [p for p in data_path.rglob("*") if p.is_file() and p.suffix.lower() in supported]

    if not source_files:
        print("No PDF/MD/TXT files found in the data folder.")
        print("Put your 10–20 pages of real documents inside ./data and run again.")
        return

    print(f"Found {len(source_files)} document(s).")

    print("Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "User document chunks for RAG"}
    )

    ids, documents, metadatas = [], [], []

    for path in source_files:
        text = clean_text(read_file(path))
        if not text:
            print(f"Skipping empty/unreadable file: {path}")
            continue

        chunks = chunk_text(text)
        print(f"{path.name}: {len(text.split())} words -> {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            raw_id = f"{path.resolve()}::{i}"
            chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()

            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "source": path.name,
                "chunk": i
            })

    if not documents:
        print("No usable text was extracted.")
        return

    print(f"Creating embeddings for {len(documents)} chunks...")
    embeddings = embedder.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True
    ).tolist()

    # Re-running ingestion replaces the same chunk IDs instead of duplicating them.
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("\nIngestion complete.")
    print(f"Stored chunks: {collection.count()}")
    print(f"Vector database: {Path(DB_DIR).resolve()}")


if __name__ == "__main__":
    main()
