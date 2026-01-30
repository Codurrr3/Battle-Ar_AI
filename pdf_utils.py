import fitz
import requests
import os
import uuid

def extract_text_from_pdf(pdf_url: str) -> str:
    filename = f"temp_{uuid.uuid4().hex}.pdf"

    r = requests.get(pdf_url)
    with open(filename, "wb") as f:
        f.write(r.content)

    text = ""
    with fitz.open(filename) as doc:
        for page in doc:
            text += page.get_text()

    os.remove(filename)
    return text
