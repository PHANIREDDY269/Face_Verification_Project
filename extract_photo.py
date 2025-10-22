# extract_photo.py
import fitz  # PyMuPDF
from PIL import Image
import io
import os

def extract_photo_from_pdf(pdf_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        images = doc.get_page_images(page_index)
        for img in images:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image.save(output_path, format="JPEG")
            print(f"Extracted image to {output_path}")
            return True
    return False
