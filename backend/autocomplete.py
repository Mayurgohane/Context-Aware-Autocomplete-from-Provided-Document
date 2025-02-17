from backend.embeddings import search_faiss
from backend.llm import enhance_with_llm

def autocomplete_suggestions(query: str):
    """Fetch FAISS and LLaMA-2 based suggestions."""
    faiss_results = search_faiss(query)
    llm_completion = enhance_with_llm(query)
    
    return faiss_results, llm_completion
