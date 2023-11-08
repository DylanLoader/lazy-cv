# # Streamlit app
#%%
import json
import streamlit as st
import os

import datetime

# LLM imports
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA

#%%
os.getcwd()

#%%
# Streamlit header 
st.title("Lazy CV LLM Cover Letter Generator")
with st.form("my_form"):
    # User inputs
    user_info = json.load(open("user_info.json", encoding='utf-8'))['user_info']
    tog = st.toggle('Save User Information')
    # Applicant Information
    st.subheader("Applicant Information")
    application_date = st.date_input("Application Date", value=None)
    applicant_fname = st.text_input("First Name", value=user_info["fname"], placeholder="Required")
    applicant_lname = st.text_input("Last Name", value=user_info["lname"], placeholder="Required")
    # applicant_email = st.text_input("Email", value=None, placeholder="Required")
    applicant_experience = st.text_area("Experience", value=None, placeholder="Required", help="Place your work experience here, usually providing it in bullet points is best.")
    # Application Information
    st.subheader("Company Information")
    company_name = st.text_input("Company Name", placeholder="Required")
    job_title = st.text_input("Job Title", placeholder="Required")
    company_description = st.text_area("Company Description or Mission Statement", placeholder="Optional")
    job_description = st.text_area("Job Description", placeholder="Required")
    submit_button = st.form_submit_button("Generate Cover Letter")   
    
with st.spinner(text="Generating Cover Letter"):
    if submit_button:
        # save submitted user data. 
        # if tog: 
        #     # Save the user data to the json
        #     pass
        # Prompt generation
        pre_prompt = "You are an AI cover letter writing assistant and your task to write a persuasive cover letter using the following information: \n"
        # Prompts from form
        # Candidate Information
        prompt = pre_prompt + f"The candidate's name is {applicant_fname} {applicant_lname}. \n"
        prompt += f"The candidate's experience is: {applicant_experience} \n"
        # Job Specific Information
        prompt += f"The date the application is being submitted is: {application_date}. \n"
        prompt += f"The job title is: {job_title}. \n"
        prompt += f"The job description is: {job_description}. \n"
        prompt += f"The company's name is {company_name}. \n"
        if company_description!="Optional":
            prompt += f"The company provided this information about themselves: {company_description}. \n"

        prompt += "Generate a cover letter at most 1000 words long using the above information."
        #############################################
        #TODO Get the experience from the resume pdf
        # prompt = f"The user's resume is {vector reference}" 
        # Create the vector reference for the resume
        # loader = PyPDFLoader(
        #     file_path=user_info["resume_path"]
        #     )
        # resume_data = loader.load()
        # text_chunk = RecursiveCharacterTextSplitter(chunk_size=10000, 
        #                                         chunk_overlap=20).split_documents(resume_data)
        # # Setup the sentence transformer emb
        # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # # Create an embedding store
        # vector_store = FAISS.from_documents(text_chunk, embedding=embeddings)
        #############################################
        model_info = json.load(open("user_info.json", encoding='utf-8'))['model_info']
        model_path = model_info["LLM_model_PATH"]
        llm = LlamaCpp(
            streaming=False,
            model_path=model_path,
            temperature=0.3,
            top_p=1,
            n_batch=128,
            n_ctx=2000,
            verbose=True
            )
        # Write prompt to file
        with open('prompt.txt', 'w', encoding='utf-8') as f:
            f.write(prompt)
        prompt_template = PromptTemplate(
            input_variables=["prompt"],
            template="[INST] {prompt}[/INST]"
            )
        from langchain.chains import LLMChain
        import langchain
        # langchain.debug = True
        llm = LLMChain(
            llm=llm,
            prompt=prompt_template
            )
        llm_response = llm.run({"prompt":prompt, "max_tokens":200})
        # llm_reply = llm_response({"prompt":prompt})
        with open('response.txt', 'w', encoding='utf-8') as f:
            f.write(llm_response)
        st.subheader("Cover Letter:")
        # st.write(qa_response)
        st.write(llm_response)
        # TODO Add signature
        # if signature_path != None:
            # current_signature = open(signature_path, "r").read()
        # Write the cover letter to a file
        # TODO figure out how to write pdf or docx
        # print(prompt)
        # TODO add dump to file button, so we can save the cover letter to a file to be used as 'good' examples for rag

        st.success('Done!')
# %%
