import json
from query_pinecone import query_pinecone
from database_helpers import load_conversation
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser  # Import the parser

parser = StrOutputParser()  # Initialize the parser


def fetch_ground_truth(question):
    """Fetches the ground truth answer from Pinecone or LLM if Pinecone fails."""
    if not question:
        return None  # Skip if question is null
    
    query_results = query_pinecone(query=question, top_k=1)
    
    if query_results["matches"]:
        return query_results["matches"][0]["metadata"]["answer"]
    
    # If Pinecone fails, use LLM to generate an answer
    llm = ChatGroq(model="qwen-2.5-32b", temperature=0)
    response = llm.invoke(question)  # Get raw response
    return parser.invoke(response)  # Extract text using parser

def generate_report():
    """Generates an interview performance report by comparing user answers to ground truth."""
    with open("database.json", "r") as file:
        conversation = json.load(file)

    enriched_data = []
    
    for entry in conversation:
        if not entry["question"]:
            continue  # Skip null questions
        
        ground_truth = fetch_ground_truth(entry["question"])
        entry["ground_truth"] = ground_truth
        enriched_data.append(entry)
    
    with open("database.json", "w") as file:
        json.dump(enriched_data, file, indent=4)

    # Generate feedback using LLM
    llm = ChatGroq(model="qwen-2.5-32b", temperature=0)
    report_prompt = f"Analyze the following interview data and provide feedback on accuracy and suggestions:\n\n{json.dumps(enriched_data, indent=2)}"
    
    response = llm.invoke(report_prompt)  # Get raw response
    report = parser.invoke(response)  # Extract text using parser

    with open("report.txt", "w") as file:
        file.write(report)

    return report

if __name__ == "__main__":
    # Example usage
    report = generate_report()
    print("Report generated successfully!")
    print(report)