from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=650, overlap=75):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def create_vector_store(chunks):
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings

def retrieve_relevant_chunks(question, index, chunks, k=5):
    q_embedding = embedder.encode([question])
    D, I = index.search(np.array(q_embedding), k)
    return " ".join([chunks[i] for i in I[0]])

