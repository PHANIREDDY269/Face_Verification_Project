# extract_details.py
import fitz, re

def extract_resume_details(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    email = emails[0] if emails else "Not found"
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    name = lines[0] if lines else "Not found"
    return {"name": name, "email": email}
