
# # Streamlit app
#%%
import json
import streamlit as st
import os

import datetime

# LLM imports
from langchain.document_loaders import PyPDFLoader

#%%
user_info_dict = {
    "fname": "Dylan",
    "lname": "Loader",
    "resume_path": {"/Users/dloader/Documents/GitHub/lazy-cv/references/Dylan-Loader-Resume-October-2023.pdf"},
    "signature_path": None,
    "manager_name": None,
}

#%%
# Streamlit header 
st.title("Lazy CV LLM Cover Letter Generator")
with st.form("my_form"):
    # User inputs 
    applicant_fname = st.text_input("First Name", value=user_info_dict["fname"])
    applicant_lname = st.text_input("Last Name", value=user_info_dict["lname"])
    application_date = st.date_input("Application Date", value=None)
    company_name = st.text_input("Company Name")
    company_description = st.text_area("Company Description or Mission Statement", "Optional")
    job_description = st.text_area("Job Description")
    submit_button = st.form_submit_button("Generate Cover Letter")   
if submit_button:
    # Prompt generation
    preprompt = "You are an assistant designed to generate cover letter here is the information required to write a compelling and persuasive cover letter:"
    # Prompts from form
    prompt = f"The candidate's name is {applicant_fname} {applicant_lname}. "
    prompt += f"The date the application is being submitted is: {application_date}. "
    prompt += f"The job description is: {job_description}. "
    prompt += f"The company's name is {company_name}. "
    if company_description!="Optional":
        prompt += f"The company provided this information about themselves: {company_description}. "
    # prompt += f"1"
    # prompt += f"2"
    # prompt += f"3"
    # prompt += f"4"
    
    # prompt = f"The user's resume is {vector reference}" 
    # text = st.text_area("Enter your text")
    
    # Call the LLM on the prompt
    
    # TODO Add signature
    # if signature_path != None:
        # current_signature = open(signature_path, "r").read()
    # Write the cover letter to a file
    # TODO figure out how to write pdf or docx
    
    # print(prompt)
