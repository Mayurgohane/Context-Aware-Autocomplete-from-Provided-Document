import streamlit as st
from backend.pdf_loader import extract_text_from_pdf
from backend.embeddings import index_document
from backend.autocomplete import autocomplete_suggestions  

# Streamlit setup
st.set_page_config(page_title="Context-Aware Autocomplete", layout="wide")

st.title("Context-Aware Autocomplete Using Vector DB & LLMs")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF Document", type="pdf")

if uploaded_file:
    # Save the uploaded PDF file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text from the PDF
    text = extract_text_from_pdf("temp.pdf")
    
    # Index the document
    index_document(text)
    st.success("Document Indexed Successfully!")

# User input for autocomplete query
user_query = st.text_input("Start typing your sentence...")

if user_query:
    # Get autocomplete suggestions from the vector DB and LLM
    faiss_suggestions, llm_completion = autocomplete_suggestions(user_query)

    # Display suggestions based on vector embeddings (FAISS)
    st.subheader("Suggestions based on Embeddings:")
    for suggestion in faiss_suggestions:
        st.write(f"- {suggestion}")

    # Display suggestions based on LLM (OpenAI)
    st.subheader("Suggestions using LLM:")
    st.write(llm_completion)
