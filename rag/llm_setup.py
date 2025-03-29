from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
import getpass
from database_helpers import load_conversation, save_conversation, clear_conversation
from query_pinecone import query_pinecone
import os

from dotenv import load_dotenv


def get_llm():
    """
    Initialize the LLM with the Groq API key and other parameters.
    """
    # Set the environment variable for the Groq API key
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    # Initialize the LLM with the Groq API key and other parameters
    llm = ChatGroq(
        model="qwen-2.5-32b",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )
    return llm

def get_prompt_template():
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
    return prompt

def get_user_info():
    """
    Get user information for the interview.
    """
    user_info = """
    Collect, clean, and analyze large datasets to provide business insights.
    - Develop and maintain interactive dashboards using Power BI and Tableau.
    - Perform data modeling and ensure efficient database management.
    - Utilize SQL to query databases and extract relevant information.
    - Conduct statistical analysis and machine learning techniques to uncover patterns and trends.
    - Collaborate with cross-functional teams to understand business requirements and deliver data-driven solutions.
    - Ensure data security and compliance with privacy regulations.
    - Optimize the performance of data visualization and reporting tools.
"""
    return user_info


def get_question_answer_context():
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
    return combine_result


prompt= get_prompt_template()
llm= get_llm()
user_info= get_user_info()
combine_result= get_question_answer_context()

parser = StrOutputParser()

# Load previous conversation
conversation_history = load_conversation()

def chat_with_llm():
    """
    Get the response from the LLM based on the user message.
    """
    # Load the conversation history
    conversation_history = load_conversation()
    
    # Create a chain with the prompt, LLM, and parser
    chain = RunnableSequence(prompt, llm, parser)
    while True:
        # Invoke the chain with the conversation history and user information
        response = chain.invoke({
                    "conversation_history": conversation_history,
                    "user_info": user_info,
                    "question_answer_list": combine_result
                })
        # Print the response
        print("\n🤖 LLM Interviewer:", response)
        # Get the user's answer
        user_answer = input("\n👤 Your Answer (type 'exit' to end interview): ")
        if user_answer.lower() == "exit":
            #clear_conversation()
            break
        # Save the conversation history
        conversation_history.append({"question": response, "answer": user_answer})
        save_conversation(conversation_history)
    print("\n🎤 Interview session has ended. Thank you!")


   

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    if "GROQ_API_KEY" not in os.environ:
        os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
    os.environ["TOKENIZERS_PARALLELISM"]=  os.getenv("TOKENIZERS_PARALLELISM")
    chat_with_llm()