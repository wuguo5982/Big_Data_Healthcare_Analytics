## List the symptoms of medical diseases, it will be beneficial for medical diagnosis and prevention.
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

# Function

def LLama_result(input_text, nums_of_words, types_of_diabetes):

    # # LLama2 model
    llm=CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.1})
    
    # Prompt Template

    template="""
        Write a medical description of {types_of_diabetes} for disease symptom {input_text}
        within {nums of words} words.
            """
    
    prompt=PromptTemplate(input_variables=["types_of_diabetes","input_text",'nums_of_words'],
                          template=template)
    
    # Format
    result=llm(prompt.format(types_of_diabetes=types_of_diabetes,input_text=input_text,nums_of_words=nums_of_words))
    print(result)
    return result



st.set_page_config(page_title="Typical Symptoms Assciated with Common Medical Diseases",
                    page_icon="🚑",
                    layout='centered',
                    initial_sidebar_state='auto')

st.header("Descript the Symptom of Diabetes Diseases 🚑")

input_text=st.text_input("Enter the Topic of Diebetes Diseases")

# Lists of columns related to the fields.

col1, col2=st.columns([4,4])

with col1:
    nums_of_words=st.text_input('nums_of_words')
with col2:
    types_of_diabetes=st.selectbox('Types of Medical Diabetes Diseases',
                            ('Type 1 Diabetes','Type 2 Diabetes','Other Types of Diabetes Diseases'), index=0)
    
submit=st.button("Generate")

# Answer
if submit:
    st.write(LLama_result(input_text, nums_of_words, types_of_diabetes))
    