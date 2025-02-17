from langchain.document_loaders import PyPDFLoader

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a given PDF file using PyPDFLoader."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return "\n".join([page.page_content for page in pages])
