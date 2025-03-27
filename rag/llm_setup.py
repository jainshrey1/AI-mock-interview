from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
import getpass
from query_pinecone import query_pinecone
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





prompt = PromptTemplate(
    template="""You are a expert in taking technical interviews in Data Analytics domain.
            You have to ask technical questions to the user from the below set of Question- Answer list, but remember you can ask subsequent question from your end also if the user is giving answer for the asked question, but if you think that you need to ask a new question then only pick the new question from the given Question-Answer list only.
            Ask only relevant question based on the provided user info.
            User-Info: {user_info}.
            Question-Answer List: {question_answer_list}.
            Keep your question under 40 words and be funny sometimes. Only one question at a time.""",
    input_variables=['user_info', 'question_answer_list']
)

parser = StrOutputParser()

messages = [
    (
        "system",
        "You are a expert in taking technical interviews in Data Analytics domain. Ask user a question like you are conducting a real interview. Only one question.",
    ),
    ("human", "Hi, I am a Data Analyst with 3 years of experience. I am proficient in SQL, Python, and Tableau. I have worked on various projects involving data cleaning, visualization, and analysis. I am looking for a new opportunity to grow my career. Can you ask me a question related to my experience?"),
]

if __name__ == "__main__":
    user_info= """
    Collect, clean, and analyze large datasets to provide business insights.
    - Develop and maintain interactive dashboards using Power BI and Tableau.
    - Perform data modeling and ensure efficient database management.
    - Utilize SQL to query databases and extract relevant information.
    - Conduct statistical analysis and machine learning techniques to uncover patterns and trends.
    - Collaborate with cross-functional teams to understand business requirements and deliver data-driven solutions.
    - Ensure data security and compliance with privacy regulations.
    - Optimize the performance of data visualization and reporting tools.
"""
    question_answer_list= query_pinecone(query= user_info, top_k=4)
    print(question_answer_list)
    chain = RunnableSequence(prompt, llm, parser)
    print(chain.invoke({'user_info':user_info, 'question_answer_list':question_answer_list}))
#ai_msg = llm.invoke(messages)
#print(type(ai_msg))
#print(ai_msg.content)