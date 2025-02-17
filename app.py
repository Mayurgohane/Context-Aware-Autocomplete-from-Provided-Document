import streamlit as st
from backend.pdf_loader import extract_text_from_pdf
from backend.embeddings import index_document
from backend.autocomplete import autocomplete_suggestions

st.set_page_config(page_title="Context-Aware Autocomplete", layout="wide")

st.title("📄 Context-Aware Autocomplete with FAISS & LLaMA-2")

uploaded_file = st.file_uploader("Upload a PDF Document", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    text = extract_text_from_pdf("temp.pdf")
    index_document(text)
    st.success("✅ Document Indexed Successfully!")

user_query = st.text_input("Start typing your sentence...")

if user_query:
    faiss_suggestions, llm_completion = autocomplete_suggestions(user_query)

    st.subheader("🔍 FAISS-based Suggestions:")
    for suggestion in faiss_suggestions:
        st.write(f"- {suggestion}")

    st.subheader("🤖 LLaMA-2 Completion (via Groq API):")
    st.write(llm_completion)
