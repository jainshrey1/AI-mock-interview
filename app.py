import streamlit as st
import re
from io import BytesIO
import os

# For PDF processing
import PyPDF2
# For DOCX processing
import docx2txt

# ------------------------
# Helper functions
# ------------------------

def extract_text_from_pdf(file: BytesIO) -> str:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def extract_text_from_docx(file: BytesIO) -> str:
    # Write to a temporary file to use docx2txt (which requires a file path)
    with open("temp.docx", "wb") as f:
        f.write(file.getbuffer())
    text = docx2txt.process("temp.docx")
    os.remove("temp.docx")
    return text

def parse_resume(text: str) -> dict:
    details = {}
    # Extract email
    emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    details["Email"] = emails[0] if emails else "Not found"
    # Extract phone number (10 digits as a simple pattern)
    phones = re.findall(r'\b\d{10}\b', text)
    details["Phone"] = phones[0] if phones else "Not found"
    # Extract LinkedIn URL (basic pattern)
    linkedin = re.findall(r'(https?://(?:www\.)?linkedin\.com/in/\S+)', text)
    details["LinkedIn"] = linkedin[0] if linkedin else "Not found"
    # Extract GitHub URL (basic pattern)
    github = re.findall(r'(https?://(?:www\.)?github\.com/\S+)', text)
    details["GitHub"] = github[0] if github else "Not found"
    # Assume the first line is the candidate's name (this is a simplification)
    lines = text.strip().splitlines()
    details["Name"] = lines[0] if lines else "Not found"
    # For demonstration, set other fields as "Not extracted"
    details["Skills"] = "Not extracted"
    details["Experience"] = "Not extracted"
    details["Projects"] = "Not extracted"
    details["Achievements"] = "Not extracted"
    return details

# ------------------------
# Page Functions
# ------------------------

def login_page():
    st.title("User Login")
    st.write("Please enter your credentials:")
    
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        # Check if the user is registered
        if "users" in st.session_state and email in st.session_state.users:
            if st.session_state.users[email]["password"] == password:
                st.success("Login successful!")
                st.session_state.logged_in = True
                st.session_state.current_user = st.session_state.users[email]
                st.session_state.page = "dashboard"
            else:
                st.error("Incorrect password. Please try again.")
        else:
            st.error("User not found. Please register first.")
            
    st.write("Don't have an account?")
    if st.button("Go to Registration"):
        st.session_state.page = "register"

def register_page():
    st.title("User Registration")
    st.write("Fill in your details to create an account:")
    
    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if st.button("Register"):
        if not all([first_name, last_name, email, password]):
            st.error("Please fill in all fields.")
        else:
            # Store the user details in session_state; in production, this would be a database
            if "users" not in st.session_state:
                st.session_state.users = {}
            st.session_state.users[email] = {
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "password": password
            }
            st.success("Account created successfully!")
            st.session_state.logged_in = True
            st.session_state.current_user = st.session_state.users[email]
            st.session_state.page = "dashboard"
    
    if st.button("Back to Login"):
        st.session_state.page = "login"

def dashboard_page():
    st.title("Dashboard")
    st.write(f"Welcome, {st.session_state.current_user['first_name']}!")
    
    st.subheader("Resume Upload and Parsing")
    st.write("Upload your resume (PDF or DOCX) to extract details.")
    
    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])
    if uploaded_file is not None:
        file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type}
        st.write(file_details)
        
        # Extract text based on file type
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            text = extract_text_from_docx(uploaded_file)
        else:
            st.error("Unsupported file type.")
            text = ""
        
        if text:
            st.success("Resume uploaded and processed successfully!")
            st.write("**Extracted Resume Text (sample):**")
            st.text_area("", text, height=200)
            
            # Parse resume details
            details = parse_resume(text)
            st.subheader("Parsed Resume Details")
            st.write(details)
    
    st.subheader("Job Description")
    job_description = st.text_area("Enter the job description here:")
    
    if st.button("Take Interview"):
        if uploaded_file is None:
            st.error("Please upload your resume before taking the interview.")
        elif not job_description.strip():
            st.error("Please enter the job description.")
        else:
            st.success("Starting interview process...")
            # Here you would implement the interview functionality
            st.info("Interview functionality coming soon!")
    
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.session_state.page = "login"
        st.success("Logged out successfully!")

# ------------------------
# Main App Navigation
# ------------------------

def main():
    st.set_page_config(page_title="Streamlit Interview App", layout="centered")
    
    # Initialize session state variables if they do not exist
    if "page" not in st.session_state:
        st.session_state.page = "login"
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "users" not in st.session_state:
        st.session_state.users = {}
    if "current_user" not in st.session_state:
        st.session_state.current_user = None

    # Navigation among pages
    if st.session_state.page == "login":
        login_page()
    elif st.session_state.page == "register":
        register_page()
    elif st.session_state.page == "dashboard":
        if st.session_state.logged_in:
            dashboard_page()
        else:
            st.error("You must be logged in to view the dashboard.")
            st.session_state.page = "login"
            login_page()
    else:
        st.error("Page not found.")

if __name__ == '__main__':
    main()
