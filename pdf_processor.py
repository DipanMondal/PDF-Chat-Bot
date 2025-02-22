from PyPDF2 import PdfReader


def get_pdfs_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text


def get_pdf_text(pdf_doc):
    text = ""
    reader = PdfReader(pdf_doc)
    for page in reader.pages:
        text += page.extract_text()
    return text
