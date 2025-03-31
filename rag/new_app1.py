import streamlit as st
# Move set_page_config to the very top!
st.set_page_config(page_title="Multi-Page App", layout="wide")

import sqlite3
import io
import re
import PyPDF2
import docx
import os
import json
import speech_recognition as sr
from dotenv import load_dotenv

# Interview-related libraries
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from database_helpers import load_conversation, save_conversation, clear_conversation
from report_generation import generate_report
from query_pinecone import query_pinecone
from llm_setup import get_prompt_template, generate_user_info

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# -------------------------------
# Database Functions (for resume uploader)
# -------------------------------
def init_db():
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            job_description TEXT,
            resume_filename TEXT,
            name TEXT,
            skills TEXT,
            work_experience TEXT,
            projects TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

def create_user(username, password):
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()
    return bool(user)

def get_user_id(username):
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        return result[0]
    return None

# -------------------------------
# File Extraction Functions
# -------------------------------
def extract_text_from_file(resume_file):
    file_extension = resume_file.name.split(".")[-1].lower()
    if file_extension == "txt":
        try:
            return resume_file.read().decode("utf-8", errors="ignore")
        except Exception as e:
            st.error("Error reading TXT file: " + str(e))
            return ""
    elif file_extension == "pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(resume_file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text
        except Exception as e:
            st.error("Error processing PDF file: " + str(e))
            return ""
    elif file_extension == "docx":
        try:
            doc = docx.Document(io.BytesIO(resume_file.read()))
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            st.error("Error processing DOCX file: " + str(e))
            return ""
    else:
        st.error("Unsupported file format!")
        return ""

# -------------------------------
# Resume Extraction Function (Dynamic & Case-Insensitive)
# -------------------------------
def extract_resume_details(text):
    details = {
        "NAME": "",
        "SKILLS": "",
        "WORK EXPERIENCE": "",
        "PROJECT": ""
    }
    fields = ["name", "skills", "work experience", "project"]
    current_field = None
    for line in text.splitlines():
        line_stripped = line.strip()
        found_field = False
        for field in fields:
            pattern = r'^' + re.escape(field) + r'\s*:?\s*(.*)$'
            match = re.match(pattern, line_stripped, re.IGNORECASE)
            if match:
                current_field = field.upper()
                details[current_field] = match.group(1).strip()
                found_field = True
                break
        if not found_field and current_field:
            details[current_field] += "\n" + line_stripped
    for key in details:
        details[key] = details[key].strip()
    return details

# -------------------------------
# Interview Helper Functions
# -------------------------------
def initialize_llm(model):
    """Initialize the LLM with the selected model and API key."""
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    return ChatGroq(
        model=model,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )


def get_question_answer_context(user_info):
    """Fetches relevant interview questions from Pinecone."""
    query_results = query_pinecone(query=user_info, top_k=4)
    return [
        {"question": match["metadata"]["question"], "answer": match["metadata"]["answer"]}
        for match in query_results["matches"]
    ]

# -------------------------------
# Resume Uploader Page
# -------------------------------
def resume_uploader():
    st.title("Resume Uploader App")
    init_db()
    
    # Track login status
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Authentication
    if not st.session_state.logged_in:
        auth_mode = st.sidebar.selectbox("Login/Sign Up", ["Login", "Sign Up"])
        
        if auth_mode == "Login":
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if login_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("Logged in successfully!")
                else:
                    st.error("Invalid username or password")
        else:
            st.subheader("Sign Up")
            new_username = st.text_input("Choose a Username", key="signup_username")
            new_password = st.text_input("Choose a Password", type="password", key="signup_password")
            if st.button("Sign Up"):
                if create_user(new_username, new_password):
                    st.success("User created successfully! Please log in.")
                else:
                    st.error("Username already exists. Please choose a different username.")
    else:
        st.sidebar.write(f"Logged in as {st.session_state.username}")
        st.subheader("Job Description and Resume Upload")
        job_description = st.text_area("Enter Job Description")
        resume_file = st.file_uploader("Upload Resume (PDF, DOCX, or TXT)", type=["txt", "pdf", "docx"])
        
        if st.button("Submit"):
            if not job_description:
                st.error("Please enter the job description.")
            elif resume_file is None:
                st.error("Please upload your resume.")
            else:
                resume_text = extract_text_from_file(resume_file)
                if not resume_text.strip():
                    st.warning("No text found in the resume. It might be image-based or empty.")
                else:
                    details = extract_resume_details(resume_text)
                    st.write("### Extracted Resume Details:")
                    st.write("**NAME:**", details.get("NAME", ""))
                    st.write("**SKILLS:**", details.get("SKILLS", ""))
                    st.write("**WORK EXPERIENCE:**", details.get("WORK EXPERIENCE", ""))
                    st.write("**PROJECT:**", details.get("PROJECT", ""))

                    # Generate user info dynamically
                    st.session_state.user_info = generate_user_info(resume_text, job_description)
                    st.success("User info generated successfully!")
                            
                    user_id = get_user_id(st.session_state.username)
                    conn = sqlite3.connect("app.db")
                    c = conn.cursor()
                    c.execute(
                        "INSERT INTO resumes (user_id, job_description, resume_filename, name, skills, work_experience, projects) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (user_id, job_description, resume_file.name,
                         details.get("NAME", ""),
                         details.get("SKILLS", ""),
                         details.get("WORK EXPERIENCE", ""),
                         details.get("PROJECT", ""))
                    )
                    conn.commit()
                    conn.close()
                    st.success("Data stored successfully!")
        
        # Navigate to Interview page
        if st.button("Take Interview"):
            st.experimental_set_query_params(page="Interview")
            st.rerun()

# -------------------------------
# Interview Page
# -------------------------------
def interview_page():
    st.title("🤖 AI Mock Interview")
    st.sidebar.title("🔧 Settings")
    
    if "last_question" not in st.session_state:
        st.session_state.last_question = None
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "report_ready" not in st.session_state:
        st.session_state.report_ready = False

    st.session_state.selected_model = st.sidebar.selectbox("Select Model", ["qwen-2.5-32b", "llama-2-13b", "gpt-4"])
    
    if st.sidebar.button("Start Interview"):
        if not GROQ_API_KEY or not PINECONE_API_KEY:
            st.sidebar.warning("Missing API keys! Check your .env file.")
        elif "user_info" not in st.session_state or not st.session_state.user_info:
            st.sidebar.error("User info is missing! Please upload your resume first.")
        else:
            st.session_state.llm = initialize_llm(st.session_state.selected_model)
            st.session_state.prompt = get_prompt_template()
            st.session_state.parser = StrOutputParser()
            st.session_state.conversation_history = load_conversation()

            # Fetch questions using generated user_info
            st.session_state.question_answer_list = get_question_answer_context(st.session_state.user_info)
            st.session_state.chat_active = True
            st.session_state.messages = []
            st.session_state.report_ready = False
            st.success("Interview started! Type below to respond.")

    if st.sidebar.button("End Interview"):
        st.session_state.chat_active = False
        st.sidebar.info("Interview ended! You can generate a report now.")

    if st.sidebar.button("Generate Report"):
        with st.spinner("Generating report... Please wait."):
            report = generate_report()
            st.session_state.report_ready = True
            clear_conversation()

    chat_container = st.container()
    
    if st.session_state.get("chat_active", False):
        user_input = st.chat_input("Type your response...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.conversation_history.append({"question": st.session_state.last_question, "answer": user_input})
            save_conversation(st.session_state.conversation_history)
            chain = RunnableSequence(st.session_state.prompt, st.session_state.llm, st.session_state.parser)
            response = chain.invoke({
                "conversation_history": st.session_state.conversation_history,
                "user_info": st.session_state.user_info,
                "question_answer_list": st.session_state.question_answer_list
            })
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.last_question = response
        with chat_container:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(msg["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(msg["content"])
    
    if st.session_state.get("report_ready", False):
        st.subheader("Interview Report")
        report_file_path = "report.txt"
        if os.path.exists(report_file_path):
            with open(report_file_path, "rb") as file:
                report_bytes = file.read()
            st.download_button(
                label="📥 Click to Download Report",
                data=report_bytes,
                file_name="Interview_Report.txt",
                mime="text/plain"
            )
            os.remove(report_file_path)
            st.session_state.report_ready = False
            st.sidebar.success("Report deleted from system after download.")
    
    if st.button("Back to Resume Uploader"):
        st.experimental_set_query_params(page="Resume")
        st.rerun()

# -------------------------------
# Main: Page Navigation Using Query Parameters
# -------------------------------
def main():
    params = st.experimental_get_query_params()
    page = params.get("page", ["Resume"])[0]
    if page == "Interview":
        interview_page()
    else:
        resume_uploader()

if __name__ == "__main__":
    main()
