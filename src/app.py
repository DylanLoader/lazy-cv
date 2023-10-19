
# # Streamlit app
#%%
import json
import streamlit as st
import os

#%%
user_info_dict = {
    "name": "Dylan Loader",
    "signature_path": None,
    "manager_name": None,
}

#%%
# Streamlit header 
st.title("Cover Letter Generator")
with st.form("my_form"):
    # User inputs 
    
    
    # Prompt generation
    preprompt = "You are an assistant designed to generate cover letters, you will only reply with the cover letter text"
    
    # Prompts from form
    prompt = f"Job description: "
    # prompt = f"The user's resume is {vector reference}" 
    # text = st.text_area("Enter your text")
    submit_button = st.form_submit_button("Generate Cover Letter")
