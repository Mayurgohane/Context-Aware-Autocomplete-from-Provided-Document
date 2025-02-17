import PyPDF2

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a given PDF file."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text
