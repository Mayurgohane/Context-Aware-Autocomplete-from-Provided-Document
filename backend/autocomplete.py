from backend.embeddings import get_similar_sentences

def autocomplete_suggestions(query: str, top_k: int = 3):
    """Returns top-k autocomplete suggestions for a given query."""
    return get_similar_sentences(query, top_k)
