import streamlit as st
import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv
from database_helpers import load_conversation, save_conversation, clear_conversation
from query_pinecone import query_pinecone
# Streamlit UI
st.set_page_config(page_title="AI Mock Interview", layout="wide")
st.sidebar.title("🔧 Settings")

# User inputs for API keys and model selection
st.session_state.groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
st.session_state.pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password")
st.session_state.selected_model = st.sidebar.selectbox("Select Model", ["qwen-2.5-32b", "llama-2-13b", "gpt-4"])

if st.sidebar.button("Start Interview"):
    if not st.session_state.groq_api_key or not st.session_state.pinecone_api_key:
        st.sidebar.warning("Please enter API keys to proceed!")
    else:
        st.session_state.llm = initialize_llm(st.session_state.selected_model)
        st.session_state.prompt = get_prompt_template()
        st.session_state.parser = StrOutputParser()
        st.session_state.conversation_history = load_conversation()
        st.session_state.user_info = "Data Analytics, SQL, Tableau, Power BI experience."
        st.session_state.question_answer_list = get_question_answer_context(st.session_state.user_info)
        st.session_state.chat_active = True
        st.session_state.messages = []
        st.success("Interview started! Type below to respond.")

# Chat Interface
st.title("🤖 AI Mock Interview")
chat_container = st.container()

if "chat_active" in st.session_state and st.session_state.chat_active:
    user_input = st.chat_input("Type your response...")
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
                st.chat_message("👤", msg["content"])
            else:
                st.chat_message("🤖", msg["content"])
