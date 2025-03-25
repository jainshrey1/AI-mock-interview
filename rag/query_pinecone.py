import os
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone index name (Ensure it follows naming rules)
INDEX_NAME = "interview"

def create_embedding(query):
    """
    Generate an embedding for the query using HuggingFace's MiniLM model.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embedding = embedding_model.embed_query(query)
    return embedding

def query_pinecone(query, top_k=2):
    """
    Query Pinecone vector database with an embedded query.
    """
    try:
        print(f"🔹 Generating embedding for query: {query}")
        xq = create_embedding(query)  # Convert query to embedding

        print("🔹 Connecting to Pinecone index...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(name=INDEX_NAME)

        print("🔹 Querying Pinecone...")
        query_results = index.query(vector=xq, top_k=top_k, include_metadata=True)
        # Display results
        print("\n🔹 Top Matching Results:")
        for match in query_results["matches"]:
            print(f"\n📌 Score: {match['score']}")
            print(f"💡 Question: {match['metadata']['question']}")
            print(f"✅ Answer: {match['metadata']['answer']}")
            print("-" * 50)

        return query_results
    except Exception as e:
        raise Exception(f"❌ Error querying Pinecone: {e}")

# Example query
query = "What are some common challenges you have faced while working with Power BI, and how did you overcome them?"
query_results = query_pinecone(query, top_k=3)
