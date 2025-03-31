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
    You are an expert interviewer in the Data Analytics domain, conducting a structured yet dynamic mock interview. Your goal is to assess the candidate’s expertise across multiple areas while maintaining a natural and engaging conversation.

    Interview Guidelines:
    Start by asking the first question from the provided Question-Answer list.

    Analyze the user’s response and ask insightful follow-up questions to assess depth and practical experience.

    Rotate between different topics within the user’s expertise rather than focusing too much on one area. Ensure a mix of:

    Technical skills (SQL, Python, data engineering, etc.)

    Cloud & DevOps (AWS, Kubernetes, CI/CD, etc.)

    Business intelligence (Tableau, Power BI, dashboarding, etc.)

    Problem-solving & scenario-based questions

    If all the questions from the Question-Answer list have been covered, generate new questions but only within the scope of the user_info.

    Ensure questions remain diverse—don’t over-focus on one area unless the user’s responses indicate deep expertise in that field.

    Occasionally add a touch of humor or encouragement to keep the conversation engaging.

    **Previous Conversation:** {conversation_history}
    **User Information:** {user_info}  
    **Question-Answer List:** {question_answer_list}""",
    input_variables=['conversation_history', 'user_info', 'question_answer_list']
    )
    return prompt

def generate_user_info(resume_text, job_description):
    """
    Generates user information summary from resume and job description.
    """
    llm = get_llm()  # Initialize LLM
    prompt_template = PromptTemplate(
        template="""
        Based on the given resume and job description, generate a structured summary 
        that highlights the user's key skills, experience, and relevant qualifications.

        **Resume Content:** {resume_text}
        **Job Description:** {job_description}

        Output should be formatted as:
        - Key Skills: [...]
        - Work Experience: [...]
        - Projects: [...]
        """,
        input_variables=["resume_text", "job_description"]
    )

    chain = RunnableSequence(prompt_template, llm, StrOutputParser())
    user_info_summary = chain.invoke({"resume_text": resume_text, "job_description": job_description})

    return user_info_summary




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



def chat_with_llm(prompt, llm, parser):
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
    
    resume_text= """
    Summary
    Results-driven Data Scientist with 3.5+ years of experience in machine learning, NLP, and data-driven decision-making. Expertise in Python, SQL, LLMs, and MLOps with hands-on experience in deploying ML models on AWS. Passionate about Generative AI and AI-driven automation.

    Skills
    Programming & Tools: Python, SQL, Pandas, NumPy, Scikit-learn, TensorFlow, Hugging Face, LangChain, PyTorch, Streamlit, Docker, AWS (SageMaker, Lambda), Git
    """
    job_description= """
    *Job Responsibilities:*
    - Collect, clean, and analyze large datasets to provide business insights.
    - Develop and maintain interactive dashboards using Power BI and Tableau.
    - Perform data modeling and ensure efficient database management.
    - Utilize SQL to query databases and extract relevant information.
    - Conduct statistical analysis and machine learning techniques to uncover patterns and trends.
    - Collaborate with cross-functional teams to understand business requirements and deliver data-driven solutions.
    - Ensure data security and compliance with privacy regulations.
    - Optimize the performance of data visualization and reporting tools.

    *Required Skills & Qualifications:*
    - Bachelor's degree in Data Science, Computer Science, Statistics, or a related field.
    - Proven experience with Power BI and Tableau for data visualization.
    - Strong SQL skills for data extraction and manipulation.
    - Knowledge of Python or R for statistical analysis and machine learning.
    - Understanding of data modeling techniques and database management.
    - Familiarity with Excel for data analysis and reporting.
    - Ability to interpret and present findings to both technical and non-technical stakeholders.
    - Experience with cloud platforms such as AWS, Azure, or Google Cloud is a plus.

    *Preferred Qualifications:*
    - Experience with data warehousing and ETL processes.
    - Knowledge of advanced statistical methods and predictive analytics.
    - Proficiency in business intelligence tools and automation frameworks.
    """

    
    llm = get_llm()
    prompt= get_prompt_template()
    parser = StrOutputParser()
    user_info= generate_user_info(resume_text, job_description)
    combine_result= get_question_answer_context()
    chat_with_llm(prompt, llm, parser)
