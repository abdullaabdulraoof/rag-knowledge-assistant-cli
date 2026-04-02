import numpy as np
import ollama

from app.ingestion import load_and_chunk_documents
from app.embeddings import create_embeddings, model
from app.retriever import create_or_load_vector_store

chat_history=[]
# Load pipeline once
chunks = load_and_chunk_documents()

embeddings = create_embeddings(chunks)

index = create_or_load_vector_store(embeddings)

def ask_question(query):

    global chat_history

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    D, I = index.search(query_embedding, k=3)

    retrieved_chunks = []

    for i in I[0]:
        if i != -1:
            retrieved_chunks.append(chunks[i])

    context = "\n".join([c.page_content for c in retrieved_chunks])

    sources = [c.metadata["source"] for c in retrieved_chunks]

    history_text = ""

    for q, a in chat_history:
        history_text += f"User: {q}\nAssistant: {a}\n"

    prompt = f"""
You are a helpful AI assistant.

Conversation History:
{history_text}

Context:
{context}

Question:
{query}

Answer clearly.
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]

    chat_history.append((query, answer))

    return answer, list(set(sources))
