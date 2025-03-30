import streamlit as st
import os
import json
import speech_recognition as sr
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv
from database_helpers import load_conversation, save_conversation, clear_conversation
from report_generation import generate_report
from query_pinecone import query_pinecone
from llm_setup import get_prompt_template

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

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

# Initialize session state variables
if "last_question" not in st.session_state:
    st.session_state.last_question = None

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "report_ready" not in st.session_state:
    st.session_state.report_ready = False  # Track if the report is generated

# Streamlit UI
st.set_page_config(page_title="AI Mock Interview", layout="wide")
st.sidebar.title("🔧 Settings")

# User selects the model
st.session_state.selected_model = st.sidebar.selectbox("Select Model", ["qwen-2.5-32b", "llama-2-13b", "gpt-4"])

# Sidebar buttons
if st.sidebar.button("Start Interview"):
    if not GROQ_API_KEY or not PINECONE_API_KEY:
        st.sidebar.warning("Missing API keys! Check your .env file.")
    else:
        st.session_state.llm = initialize_llm(st.session_state.selected_model)
        st.session_state.prompt = get_prompt_template()
        st.session_state.parser = StrOutputParser()
        st.session_state.conversation_history = load_conversation()
        st.session_state.user_info = "Data Analytics, SQL, Tableau, Power BI experience."
        st.session_state.question_answer_list = get_question_answer_context(st.session_state.user_info)
        st.session_state.chat_active = True
        st.session_state.messages = []
        st.session_state.report_ready = False  # Reset report state
        st.success("Interview started! Type below to respond.")

# End Interview Button in Sidebar
if st.sidebar.button("End Interview"):
    """Stop the interview but DO NOT clear database.json"""
    st.session_state.chat_active = False
    st.sidebar.info("Interview ended! You can generate a report now.")

# Generate Report Button in Sidebar
if st.sidebar.button("Generate Report"):
    """Generate the report with a loading spinner, provide a download option, and delete the report file after download."""
    with st.spinner("Generating report... Please wait."):
        report = generate_report()
        st.session_state.report_ready = True  # Mark that the report is ready
        clear_conversation()  # Clear database.json AFTER report generation

# Chat Interface
st.title("🤖 AI Mock Interview")
chat_container = st.container()

# 🎤 VOICE INPUT FUNCTION
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening... Speak now!")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            st.success("Processing voice input...")
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            st.error("Sorry, could not understand the audio.")
            return ""
        except sr.RequestError:
            st.error("Speech recognition service is unavailable.")
            return ""

if "chat_active" in st.session_state and st.session_state.chat_active:
    st.write("🎤 **You can type or use voice input!**")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        voice_button = st.button("🎙️ Use Voice Input")
    
    with col2:
        user_input = st.chat_input("Type your response...")

    if voice_button:
        user_input = recognize_speech()

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.conversation_history.append({"question": st.session_state.last_question, "answer": user_input})
        save_conversation(st.session_state.conversation_history)
        
        # Generate AI response
        chain = RunnableSequence(st.session_state.prompt, st.session_state.llm, st.session_state.parser)
        response = chain.invoke({
            "conversation_history": st.session_state.conversation_history,
            "user_info": st.session_state.user_info,
            "question_answer_list": st.session_state.question_answer_list
        })
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.last_question = response
        
    # Display conversation
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])

# Report Download Section
if st.session_state.report_ready:
    st.subheader("Interview Report")
    
    report_file_path = "report.txt"
    if os.path.exists(report_file_path):
        with open(report_file_path, "rb") as file:
            report_bytes = file.read()
        
        # Download button
        st.download_button(
            label="📥 Click to Download Report",
            data=report_bytes,
            file_name="Interview_Report.txt",
            mime="text/plain"
        )
        
        # Delete report file after download
        os.remove(report_file_path)
        st.session_state.report_ready = False  # Reset state after download
        st.sidebar.success("Report deleted from system after download.")