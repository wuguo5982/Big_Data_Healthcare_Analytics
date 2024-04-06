# This project aims to classify patients' diabetes types based on initial descriptions by comparing them 
# with standard diabetes features (pdf format), and subsequently propose suitable treatment strategies.
# It maybe greatly helpful for virtual medical doctor in the future.

# Original medical symptom from git clone https://huggingface.co/datasets/BI55/MedText

# 1). PDF to image (pdf2image)
# 2). Format of pdf_parts (format of jpg, encode to base64)
# 3). Model of Google Gemini Pro
# 4). Prompts instruction
# 5). Streamlit App


import os
import base64
import streamlit as st
import io
from PIL import Image 
import pdf2image
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = "XXX"
os.environ["GOOGLE_API_KEY"] = "XXX"
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def medical_gemini(input, pdf_cotent, prompt):
    model=genai.GenerativeModel('gemini-pro-vision')
    response=model.generate_content([input, stardard_diebetes[0],prompt])  # stardard_diebetes is pdf (format)
    return response.text

def stardard_diseases_pdf(uploaded_file):
    if uploaded_file is not None:                                    
        images=pdf2image.convert_from_bytes(uploaded_file.read())          #  Convert pdf to image using pdf2image
        first_page=images[0]
        
        img_byte_arr = io.BytesIO()                                        # Convert to bytes
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "mime_type": "image/jpeg",                                # format of jpg
                "data": base64.b64encode(img_byte_arr).decode()           # encode to base64 (format)
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")

## Streamlit App

st.set_page_config(page_title="Virtual Medical Doctor")
st.header("Classify the Type of Diebetes")
input_text=st.text_area("Symptom Description: ", key="input")
uploaded_file=st.file_uploader("Upload your resume(PDF)...", type=["pdf"])


if uploaded_file is not None:
    st.write("PDF Uploaded Successfully")


submit1 = st.button("Tell me about the Medical Symptom of this Patient")
submit2 = st.button("Classify the Type of Diebetes Based on Patients' Symptoms")
submit3 = st.button("Recommend Treatment Strategy for this Patient")

prompt_sample = """
As an experienced medical doctor, your role is to review the patient's medical symptom description and recommend a treatment strategy. 
Please provide your professional evaluation on whether the patient's symptoms align with standard symptom profiles. 
Additionally, recommend potential treatment methods based on the specified symptoms.
"""


if uploaded_file is not None:
    stardard_diebetes=stardard_diseases_pdf(uploaded_file)
    response=medical_gemini(prompt_sample, stardard_diebetes, input_text)
    st.subheader("The Answer is")
    st.write(response)
else:
    st.write("Please upload the medical symptom description")



# Symptom: 
# A 24-year-old woman with Type 1 Diabetes presents with persistent nausea, vomiting, and abdominal pain for the past day. 
# She also mentions she has been feeling thirsty and urinating a lot. On examination, her breath has a fruity odor. 
# Her blood glucose level is 450 mg/dL, and urine ketones are positive. What is the potential diagnosis, and what are the next steps?",
# "This woman's presentation is consistent with diabetic ketoacidosis (DKA), a serious complication of Type 1 Diabetes that occurs 
# when the body starts breaking down fat at a fast rate due to a lack of insulin, leading to an accumulation of ketones in the blood and urine. 
# Immediate hospital admission is required. Treatment includes intravenous fluids, insulin therapy, and electrolyte replacement. 
# Additionally, identifying and treating any underlying trigger (such as infection) is important."



# The Answer is:
# 1). Tell Me About the Summarized Symptom within 100 words:
# The patient's symptoms align with standard symptom profiles for Type 1 Diabetes. The excessive thirst and urination are classic symptoms of high blood sugar levels, 
# as the body attempts to eliminate the excess sugar through urine. Fatigue and weight loss can also occur due to the body's inability to use glucose for energy, 
# leading to a breakdown of muscle and fat tissue. Blurred vision is another common symptom, as high blood sugar levels can affect the blood vessels in the retina, 
# leading to changes in the lens's shape and function. Tingling or numbness in the hands and feet is a sign of nerve damage caused by chronic high blood sugar levels..


# 2). Classify the Type of Diebetes Based on Patients' Symptoms:
# The patient's symptoms align with standard symptom profiles for type 1 diabetes. 

# 3). Recommend Treatment Strategy for this Patient:
# The recommended treatment methods include insulin therapy, dietary changes, and regular exercise.

## References:
# 1). Huggingface. 
# 2). Contribution from Krish naik.
# Medical symptom from git clone https://huggingface.co/datasets/BI55/MedText
