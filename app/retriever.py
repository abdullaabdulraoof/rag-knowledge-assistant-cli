import faiss
import os

VECTOR_PATH = "vectorstore/faiss_index.bin"


def create_or_load_vector_store(embeddings):

    dimension = embeddings.shape[1]

    # If index already exists → load it
    if os.path.exists(VECTOR_PATH):

        print("Loading existing FAISS index...")
        index = faiss.read_index(VECTOR_PATH)

    else:

        print("Creating new FAISS index...")
        index = faiss.IndexFlatL2(dimension)

        index.add(embeddings)

        os.makedirs("vectorstore", exist_ok=True)

        faiss.write_index(index, VECTOR_PATH)

        print("FAISS index saved.")

    print("Stored vectors:", index.ntotal)

    return index