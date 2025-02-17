import PyPDF2
import pypdf

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a given PDF file."""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"
