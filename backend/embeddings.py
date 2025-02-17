import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


dimension = 384  
index = faiss.IndexFlatL2(dimension)

stored_sentences = []

def index_document(text: str):
    """Indexes a document into FAISS."""
    global stored_sentences
    sentences = text.split(". ")
    stored_sentences.extend(sentences)
    
    embeddings = embedding_model.encode(sentences, convert_to_numpy=True)
    index.add(embeddings)

def search_faiss(query: str, top_k: int = 3):
    """Retrieves top-k relevant sentences from FAISS."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    
    return [stored_sentences[i] for i in indices[0] if i < len(stored_sentences)]
