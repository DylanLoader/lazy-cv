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

#%%
# Streamlit header 
st.title("Lazy CV LLM Cover Letter Generator")
with st.form("my_form"):
    # User inputs
    user_info = json.load(open("user_info.json"))['user_info']
    tog = st.toggle('Save User Information')
    applicant_fname = st.text_input("First Name", value=user_info["fname"])
    applicant_lname = st.text_input("Last Name", value=user_info["lname"])
    application_date = st.date_input("Application Date", value=None)
    company_name = st.text_input("Company Name")
    job_title = st.text_input("Job Title")
    company_description = st.text_area("Company Description or Mission Statement", "Optional")
    job_description = st.text_area("Job Description")
    submit_button = st.form_submit_button("Generate Cover Letter")   
with st.spinner(text="Generating Cover Letter"):
    if submit_button:
        
        # save submitted user data. 
        if tog: 
            # Save the user data to the json
            pass
        # Prompt generation
        pre_prompt = "You are an AI cover letter writing assistant and your task to write a persuasive cover letter using the following information: "
        # Prompts from form
        prompt = pre_prompt + f"The candidate's name is {applicant_fname} {applicant_lname}. "
        prompt += f"The date the application is being submitted is: {application_date}. "
        prompt += f"The job title is: {job_title}. "
        prompt += f"The job description  is: {job_description}. "
        prompt += f"The job description is: {job_description}. "
        prompt += f"The company's name is {company_name}. "
        if company_description!="Optional":
            prompt += f"The company provided this information about themselves: {company_description}. "
        # prompt += f"1"
        # prompt += f"2"
        # prompt += f"3"
        # prompt += "[/INST]"
        
        # prompt = f"The user's resume is {vector reference}" 
        # text = st.text_area("Enter your text")
        
        # Create the vector reference for the resume
        loader = PyPDFLoader(
            file_path=user_info["resume_path"]
            )
        resume_data = loader.load()
        text_chunk = RecursiveCharacterTextSplitter(chunk_size=10000, 
                                                chunk_overlap=20).split_documents(resume_data)
        # Setup the sentence transformer emb
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Create an embedding store
        vector_store = FAISS.from_documents(text_chunk, embedding=embeddings)
        
        model_path = "/Users/dloader/Documents/LLM-models/TheBloke/mistral-11b-omnimix-bf16.Q5_K_M.gguf/mistral-11b-omnimix-bf16.Q5_K_M.gguf"
        llm = LlamaCpp(
            streaming=True,
            model_path=model_path,
            temperature=0.7,
            top_p=1,
            n_ctx=4096,
            verbose=False,
            )
        from langchain.chains import RetrievalQA
        prompt_template = PromptTemplate(
            input_variables=["prompt"],
            template="[INST] {prompt}[/INST]"
            )
        # chain = LLMChain(
        #     llm=llm,
        #     prompt=prompt_template,
        #     retriever=vector_store.as_retriever(search_kwargs={"k": 1})
        #     )
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={'k': 1}),
            chain_type="stuff",   
        )
        qa_response = qa.run(prompt)
        st.subheader("Cover Letter:")
        st.write(qa_response)
        # TODO Add signature
        # if signature_path != None:
            # current_signature = open(signature_path, "r").read()
        # Write the cover letter to a file
        # TODO figure out how to write pdf or docx
        
        # print(prompt)
st.success('Done!')
# %%
