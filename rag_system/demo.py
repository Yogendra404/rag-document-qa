"""
Simple demonstration for the assignment:
1. A question answerable from the documents.
2. A question that should be unknown to a plain/general LLM.

Run:
    python demo.py
"""

import ask


def run_demo(question):
    client = __import__("chromadb").PersistentClient(path=ask.DB_DIR)
    collection = client.get_or_create_collection(name=ask.COLLECTION_NAME)
    embedder = ask.SentenceTransformer("all-MiniLM-L6-v2")

    result = ask.retrieve(question, collection, embedder)

    print("\nQUESTION:", question)
    print("TOP RETRIEVED SOURCES:")
    for meta, distance in zip(result["metadatas"][0], result["distances"][0]):
        print(f"  {meta['source']} | chunk {meta['chunk']} | distance={distance:.4f}")

    prompt = ask.build_prompt(question, result)
    print("\nRAG ANSWER:")
    print(ask.generate_answer(prompt))


if __name__ == "__main__":
    run_demo("What specific information does my document set provide about the topic?")
    run_demo("What is the secret code word written in my private documents?")
