from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
import getpass
from query_pinecone import query_pinecone
import os
import json

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
    template="""
    You are an expert interviewer in the Data Analytics domain, conducting a structured mock interview.

    - Start by asking the first question **from the provided Question-Answer list**.
    - If the user responds, analyze their answer and ask a **relevant follow-up question** to dig deeper into their experience.
    - If the response requires shifting topics, pick the **next question only from the provided Question-Answer list**.
    - Never generate a new question outside the provided list.
    - Maintain a natural conversation flow and keep each question concise (under 40 words).
    - Occasionally, add a touch of humor to keep the interview engaging.

    **Previous Conversation:** {conversation_history}
    **User Information:** {user_info}  
    **Question-Answer List:** {question_answer_list}""",
    input_variables=['conversation_history', 'user_info', 'question_answer_list']
    )

parser = StrOutputParser()

messages = [
    (
        "system",
        "You are a expert in taking technical interviews in Data Analytics domain. Ask user a question like you are conducting a real interview. Only one question.",
    ),
    ("human", "Hi, I am a Data Analyst with 3 years of experience. I am proficient in SQL, Python, and Tableau. I have worked on various projects involving data cleaning, visualization, and analysis. I am looking for a new opportunity to grow my career. Can you ask me a question related to my experience?"),
]

# File to store conversation history
DATABASE_FILE = "database.json"

# Function to load conversation history
def load_conversation():
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, "r") as file:
            return json.load(file)
    return []

# Function to save conversation history
def save_conversation(history):
    with open(DATABASE_FILE, "w") as file:
        json.dump(history, file, indent=4)

# Function to clear the conversation history (reset database)
def clear_conversation():
    if os.path.exists(DATABASE_FILE):
        os.remove(DATABASE_FILE)
        print("\n✅ Interview session ended. Conversation history cleared.")


   

if __name__ == "__main__":
    
    os.environ["TOKENIZERS_PARALLELISM"]=  os.getenv("TOKENIZERS_PARALLELISM")

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
    query_results= query_pinecone(query= user_info, top_k=4)
    # Display results
    combine_result= ''
    print("\n🔹 Top Matching Results:")
    for match in query_results["matches"]:
        #print(f"\n📌 Score: {match['score']}")
        #print(f"💡 Question: {match['metadata']['question']}")
        #print(f"✅ Answer: {match['metadata']['answer']}")
        combine_result= combine_result + f"\n📌 Score: {match['score']}" + f"💡 Question: {match['metadata']['question']}" + f"✅ Answer: {match['metadata']['answer']}"
        #question_answer_list.append(combine_result)

    print(combine_result)
    print("-" * 50)

    # Load previous conversation
    conversation_history = load_conversation()
    
    # Construct the LLM chain
    chain = RunnableSequence(prompt, llm, parser)


    while True:
        # Generate the next interview question
        response = chain.invoke({
            "conversation_history": conversation_history,
            "user_info": user_info,
            "question_answer_list": combine_result
        })

        # Print the question
        print("\n🤖 LLM Interviewer:", response)

        # Get the user's answer
        user_answer = input("\n👤 Your Answer (type 'exit' to end interview): ")

        if user_answer.lower() == "exit":
            clear_conversation()
            break

        # Save the conversation history
        conversation_history.append({"question": response, "answer": user_answer})
        save_conversation(conversation_history)
    print("\n🎤 Interview session has ended. Thank you!")