import os
import json

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