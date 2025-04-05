
## Goals:
1. Extract header fields (patient info, ICD codes, total charges)

2. Extract tabular data: CPT codes, descriptions, charges from claim form

3. Use OCR with Tesseract and OpenCV for structured parsing

4. Provide a Streamlit app for drag-and-drop, preview, and CSV download

# !pip install pdf2image
# !pip install pytesseract
# !pip install cv2
# !pip install opencv-python

import streamlit as st
from pdf2image import convert_from_bytes
import pytesseract, cv2
import numpy as np
import pandas as pd
import re
import os
from io import BytesIO
from tempfile import TemporaryDirectory

def preprocess_image(pil_img):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

def extract_text(img):
    return pytesseract.image_to_string(img)

def extract_header(text):
    fields = {
        "Patient Name": re.search(r"Patient Name[:\-]?\s*(.+)", text),
        "Patient ID": re.search(r"Patient ID[:\-]?\s*(\w+)", text),
        "ICD Code": re.search(r"ICD-10[:\-]?\s*([A-Z]\d{2}\.?\d*)", text),
        "Total Amount": re.search(r"Total Billed[:\-]?\s*\$?(\d+\.?\d*)", text)
    }
    return {k: (v.group(1).strip() if v else "") for k, v in fields.items()}

def extract_table_lines(text):
    # Capture lines with CPT codes and surrounding details
    lines = text.split("\n")
    cpt_data = []
    for line in lines:
        match = re.search(r"\b(\d{5})\b.*?\$?(\d+\.?\d*)", line)
        if match:
            code = match.group(1)
            amount = match.group(2)
            desc = re.sub(r"\b(\d{5})\b|\$?\d+\.?\d*", "", line).strip()
            cpt_data.append({"CPT Code": code, "Description": desc, "Charge": amount})
    return pd.DataFrame(cpt_data)

def process_pdf(uploaded_file):
    images = convert_from_bytes(uploaded_file.read())
    img = preprocess_image(images[0])
    text = extract_text(img)
    header = extract_header(text)
    table = extract_table_lines(text)
    return header, table

# Configure Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Streamlit UI setup
st.set_page_config(page_title="CMS-1500 Claim OCR", layout="centered")
st.title("📄 CMS-1500 Healthcare Claim OCR")

# Upload multiple claim PDFs
uploaded_files = st.file_uploader("Upload one or more CMS-1500 claim PDFs", type=["pdf"], accept_multiple_files=True)

# Process uploaded files
if uploaded_files:
    headers, procedures = [], []

    with TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # OCR processing
            image = convert_from_bytes(open(file_path, "rb").read(), dpi=300)[0]
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            blur = cv2.bilateralFilter(gray, 9, 75, 75)
            proc = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            text = pytesseract.image_to_string(proc, config='--oem 3 --psm 6')

            # Regex helper functions
            def get(g): return g.group(1).strip() if g else ""
            def match(p): return re.search(p, text)

            # Extract claim header fields
            header = {
                "File": uploaded_file.name,
                "Patient Name": get(match(r"(?:2\.?\s*)?Patient(?:’s)? Name[:\-\s]*([A-Za-z ,.'-]+)")),
                "Patient ID": get(match(r"(?:1a\.?\s*)?Insured ID Number[:\-\s]*(\w+)")),
                "ICD Code": get(match(r"(?:21\.)?ICD(?:-10)?[:\-\s]*([A-Z]\d{2}\.?\d*)")),
                "Total Amount": get(match(r"(?:28\.)?Total Charge[s]?:?\s*\$?(\d+\.?\d*)")),
                "Date of Service": get(match(r"(?:24A)?\s*Date[s]? of Service[:\-\s]*(\d{2}/\d{2}/\d{4})")),
                "Provider Name": get(match(r"(?:33\.)?Provider Name[:\-\s]*([A-Za-z ,.'-]+)")),
                "Insurance": get(match(r"Insurance Plan Name[:\-\s]*([A-Za-z0-9 ,.'-]+)")),
                "Chief Complaint": get(match(r"Chief Complaint[:\-\s]*(.+)"))
            }
            headers.append(header)

            # Extract CPT procedure table lines
            for line in text.split("\n"):
                m = re.search(r"\b(\d{5})\b.*?(\d{2,}\.\d{2})", line)
                if m:
                    code, amt = m.group(1), m.group(2)
                    desc = re.sub(r"\b\d{5}\b|\$?\d+\.\d*", "", line).strip()
                    procedures.append({
                        "File": uploaded_file.name,
                        "CPT Code": code,
                        "Description": desc,
                        "Charge": amt
                    })

    # Display and export header results
    st.subheader("🧾 Extracted Claim Headers")
    header_df = pd.DataFrame(headers)
    st.dataframe(header_df)
    st.download_button("📥 Download All Headers CSV", header_df.to_csv(index=False), "all_claim_summaries.csv", "text/csv")

    # Display and export CPT table results
    st.subheader("📋 Extracted CPT Procedure Table")
    if procedures:
        proc_df = pd.DataFrame(procedures)
        st.dataframe(proc_df)
        st.download_button("📥 Download All CPT Tables CSV", proc_df.to_csv(index=False), "all_cpt_tables.csv", "text/csv")
    else:
        st.warning("No CPT codes found in the uploaded documents.")