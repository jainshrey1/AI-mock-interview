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


def get_interview_data():
    """
    Get the interview data from the CSV file.
    """
    try:
        file_path = "/Users/sachinmishra/Desktop/AI-mock-interview/Dataset/cleaned_dataset.csv"
        interview_dataframe = pd.read_csv(file_path)
        print("✅ Interview data loaded successfully!")

        interview_dataframe = interview_dataframe.drop_duplicates(keep="first")
        print("✅ Dropped duplicate questions!")

        # Removing prefix numbers from the Question column
        interview_dataframe["Question"] = interview_dataframe["Question"].str.replace(r"^\d+\.\s*", "", regex=True)
        print("✅ Removed numeric prefixes from questions!")

        return interview_dataframe
    except Exception as e:
        raise Exception(f"❌ Error reading interview data: {e}")
    
def create_pinecone_index():
    """
    Create and return a Pinecone index.
    """
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        if INDEX_NAME not in pc.list_indexes().names():
            print(f"🔹 Creating Pinecone index: {INDEX_NAME}...")
            pc.create_index(
                name=INDEX_NAME,  
                dimension=384,  # Fixed from 786 to correct dimension (MiniLM is 384)
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print("✅ Pinecone index created successfully!")

        print("🔹 Connecting to existing Pinecone index...")
        index = pc.Index(name=INDEX_NAME)
        return index
    except Exception as e:
        raise Exception(f"❌ Error in creating or connecting to Pinecone index: {e}")

def chunk_data(data, batch_size=100):
    """
    Splits data into smaller chunks for batch upserting.
    """
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]
    

def generate_embeddings():
    """
    Generate embeddings for interview data and upsert into Pinecone.
    """
    try:
        print("🔹 Initializing embedding model...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        interview_dataframe = get_interview_data()
        print(f"✅ Loaded {len(interview_dataframe)} interview questions.")

        # Initialize Pinecone index
        index = create_pinecone_index()

        # Convert rows to vectors
        vectors = []
        for i, row in interview_dataframe.iterrows():
            text = f"Skill: {row['Skill']}\nQuestion: {row['Question']}\nAnswer: {row['Answer']}"
            embedding = embedding_model.embed_query(text)  # Generate embedding
            vectors.append((str(i), embedding, {"skill": row["Skill"], "question": row["Question"], "answer": row["Answer"]}))

        print(f"🔹 Preparing to upsert {len(vectors)} vectors into Pinecone...")

        # Upsert data in batches
        for batch in chunk_data(vectors, batch_size=100):
            index.upsert(batch)
            print(f"✅ Upserted {len(batch)} vectors successfully!")

        print("🎉 All data successfully stored in Pinecone!")
    except Exception as e:
        raise Exception(f"❌ Error in generating embeddings: {e}")

if __name__ == "__main__":
    generate_embeddings()