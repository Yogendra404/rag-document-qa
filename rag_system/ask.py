"""
Ask questions against the indexed documents.

Run:
    python ask.py
"""

import requests
import chromadb
from sentence_transformers import SentenceTransformer

from config import DB_DIR, COLLECTION_NAME, TOP_K, OLLAMA_URL, OLLAMA_MODEL


def retrieve(question, collection, embedder):
    q_embedding = embedder.encode(
        [question],
        normalize_embeddings=True
    )[0].tolist()

    result = collection.query(
        query_embeddings=[q_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    return result


def build_prompt(question, result):
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]

    context_parts = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        context_parts.append(
            f"[Source {i}: {meta['source']}, chunk {meta['chunk']}]\n{doc}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document-grounded assistant.

Answer the user's question using ONLY the supplied document context.
If the context does not contain the answer, say:
"I could not find that information in the provided documents."

Do not invent facts. At the end, list the source filenames you used.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""
    return prompt


def generate_answer(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"]


def main():
    print("Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        print("Vector database is empty. Run: python ingest.py")
        return

    print(f"Indexed chunks: {collection.count()}")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        result = retrieve(question, collection, embedder)

        print("\nRetrieved chunks:")
        for meta, distance in zip(result["metadatas"][0], result["distances"][0]):
            print(f"- {meta['source']} | chunk {meta['chunk']} | distance={distance:.4f}")

        prompt = build_prompt(question, result)

        try:
            answer = generate_answer(prompt)
        except requests.exceptions.ConnectionError:
            print("\nOllama is not running. Start Ollama and make sure the model is installed.")
            print("Example: ollama pull llama3.2:3b")
            continue
        except Exception as e:
            print(f"\nLLM error: {e}")
            continue

        print("\nAssistant:")
        print(answer)
        print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    main()
