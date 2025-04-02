import os
import json
import sqlite3

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

# Database Functions (for resume uploader)
# -------------------------------
def init_db():
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            job_description TEXT,
            resume_filename TEXT,
            name TEXT,
            skills TEXT,
            work_experience TEXT,
            projects TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

def create_user(username, password):
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()
    return bool(user)

def get_user_id(username):
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        return result[0]
    return None