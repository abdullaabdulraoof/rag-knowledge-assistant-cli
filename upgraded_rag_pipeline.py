from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
import os

docs = []

for file in os.listdir("documents"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(f"documents/{file}")
        docs.extend(loader.load())

print("Total Pages Loaded:", len(docs))

# Chunking

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

print("Total Chunks:", len(chunks))

from sentence_transformers import SentenceTransformer
# -----------------------------
# Create Embeddings
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk.page_content for chunk in chunks]

embeddings = model.encode(texts)

print("Embedding shape:", embeddings.shape)

import faiss
import numpy as np
# conver to numpy array and float32 for faiss

embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
# create faiss index

index = faiss.IndexFlatL2(dimension)
# add embeddings to index

index.add(embeddings)

print("Stored vectors:", index.ntotal)

# query = "Who created Python?"

# query_embedding = model.encode([query])
# query_embedding = np.array(query_embedding).astype("float32")

# D, I = index.search(query_embedding, k=3)

# print("Top results index:", I)
# print("\nRetrieved Chunks:\n")

# for i in I[0]:
#     print(chunks[i].page_content)
#     print("----------------------")

# Now we have the retrieved chunks, we can use them as context for an LLM to answer the question.
import ollama

query = "Who created Python?"

query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

D, I = index.search(query_embedding, k=3)

# Collect retrieved chunks
retrieved_chunks = []

for i in I[0]:
    if i != -1:
        retrieved_chunks.append(chunks[i])

context = "\n".join([c.page_content for c in retrieved_chunks])

sources = [c.metadata["source"] for c in retrieved_chunks]

print("\nRetrieved Context:\n", context)
# Create prompt for LLM

prompt = f"""
You are a helpful AI assistant.

Answer the question using the context below.

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

print("\nAnswer:\n")
print(answer)

print("\nSources:")
for s in set(sources):
    print("-", s)