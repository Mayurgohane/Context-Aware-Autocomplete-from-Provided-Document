import streamlit as st
from backend.pdf_processing import extract_text_from_pdf
from backend.embeddings import add_text_to_faiss
from backend.autocomplete import autocomplete_suggestions
from backend.llm import enhance_with_llm

st.title("📄 Context-Aware Autocomplete")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
if uploaded_file is not None:
    pdf_path = "temp.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract and index text
    text = extract_text_from_pdf(pdf_path)
    add_text_to_faiss(text)

    st.success("PDF uploaded and processed successfully! ✅")

# Text input for autocomplete
user_input = st.text_input("Start typing...")

if user_input:
    suggestions = autocomplete_suggestions(user_input, top_k=3)
    
    st.subheader("📌 Autocomplete Suggestions")
    for suggestion in suggestions:
        st.write(f"- {suggestion}")

    # LLaMA-enhanced suggestion
    enhanced_suggestion = enhance_with_llm(user_input)
    st.subheader("🤖 AI-Enhanced Suggestion")
    st.write(enhanced_suggestion)
