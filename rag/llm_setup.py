from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
import getpass
import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

llm = ChatGroq(
    model="qwen-2.5-32b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

messages = [
    (
        "system",
        "You are a expert in taking technical interviews in Data Analytics domain. Ask user a question like you are conducting a real interview. Only one question.",
    ),
    ("human", "Hi, I am a Data Analyst with 3 years of experience. I am proficient in SQL, Python, and Tableau. I have worked on various projects involving data cleaning, visualization, and analysis. I am looking for a new opportunity to grow my career. Can you ask me a question related to my experience?"),
]


chain = RunnableSequence(prompt1, llm, parser)
print(chain.invoke({'topic':'AI'}))
#ai_msg = llm.invoke(messages)
#print(type(ai_msg))
#print(ai_msg.content)