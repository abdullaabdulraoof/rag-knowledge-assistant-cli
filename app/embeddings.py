from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(chunks):

    texts = [chunk.page_content for chunk in chunks]

    embeddings = model.encode(texts)

    embeddings = np.array(embeddings).astype("float32")

    print("Embedding shape:", embeddings.shape)

    return embeddings