import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# FAISS Index
dimension = 384  # Model output size
faiss_index = faiss.IndexFlatL2(dimension)
sentences = []  # Store sentences from PDF


def add_text_to_faiss(text: str):
    """Splits text into sentences, generates embeddings, and stores in FAISS."""
    global sentences
    sentences = text.split(". ")  # Simple sentence segmentation
    embeddings = embedding_model.encode(sentences)
    faiss_index.add(np.array(embeddings))


def get_similar_sentences(query: str, top_k: int = 3):
    """Retrieves top-k most relevant sentences based on similarity."""
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(np.array(query_embedding), top_k)
    return [sentences[idx] for idx in indices[0] if idx < len(sentences)]
