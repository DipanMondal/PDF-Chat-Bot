from PyPDF2 import PdfReader
import io
import fitz


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


def pdf_extract(pdf_doc):
    doc = fitz.open(filetype="pdf", stream=pdf_doc.read())  # Read from BytesIO
    text_data = []
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]

        # Extract text
        text = page.get_text("text")
        if text.strip():
            text_data.append({"page": page_num + 1, "content": text})

        # Extract images
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]  # Extract image as bytes
            images.append({"page": page_num + 1, "image_data": img_bytes})

    return text_data, images


if __name__ == '__main__':
    pass

