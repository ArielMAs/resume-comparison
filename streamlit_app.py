import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from PyPDF2 import PdfReader

st.set_page_config(page_title='Resume vs. Job Description Analyzer', layout='wide')

def get_pdf_text(pdf):
    text=""
    #for pdf in pdf_docs:
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_conversational_chain(user_api,resume_text,jd_text):
    # Step 1: Create a prompt template
    template = """
You are a career advisor and job market expert. Compare the following resume and job description.

Your tasks:
1. Provide a concise summary at the top including:
   - Overall Fit: High / Moderate / Low
   - Key Gaps (short list)
2. Identify detailed GAPS — missing skills, qualifications, or experiences based on the job description.
3. Suggest RESUME IMPROVEMENTS — what to emphasize or rephrase to better match the role.

Format your response like this:

Summary:
- Overall Fit: <High | Moderate | Low>
- Key Gaps: <comma-separated short list>

Gaps:
- ...

Resume Improvement Suggestions:
- ...

---
RESUME:  
{resume}

---
JOB DESCRIPTION:  
{job_description}
"""

    #Step 2: load model
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=user_api)
    prompt = PromptTemplate(template=template,input_variables = ["resume", "job_description"])
    # Step 3: Create the chain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Step 4: Run the chain with your inputs
    response = chain.run({
        "resume": resume_text,
        "job_description": jd_text
    })
    st.write("Reply: ",response)





# Show title and description.
st.title("🔍 Resume vs. Job Description Analyzer")
st.markdown("""
Welcome to the **Resume vs. JD Analyzer** — powered by a large language model!

Paste a resume and a job description below to:
- ✅ **Identify missing skills, experience, or qualifications**
- 💡 **Get personalized suggestions to improve your resume**
- 🎯 **Optimize your chances of landing the job**

This tool is perfect for job seekers, recruiters, and career coaches.  
_All processing is done securely using AI. No data is stored._

---
            """)
st.write(
    "Upload your resume and job description  – Google Generative AI will answer! "
    "You'll need a Google API key for the chatbot to access Google's Generative AI models. Obtain your [API key](https://makersuite.google.com/app/apikey) "
)

google_api_key = st.text_input("Enter your Google API Key:", type="password")

if not google_api_key:
    st.info("Please add your Google API key to continue.", icon="🗝️")
else:

    # Let the user upload a file via `st.file_uploader`.
    uploaded_resume = st.file_uploader(
        "Upload your resume (.pdf)",type='pdf'
    )
    
    
    # Let the user paste job description via `st.text_input`.
    job_description = st.text_input(
        "Paste job Description"
    )

    if st.button("Submit & process"):
        with st.spinner("processing..."):
            raw_text = get_pdf_text(uploaded_resume)
            st.success("Documents processed successfully!")
            with st.spinner("Generating results..."):
                get_conversational_chain(user_api=google_api_key,resume_text=raw_text,jd_text=job_description)
            

        
