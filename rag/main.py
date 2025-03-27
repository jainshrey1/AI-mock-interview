from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import uvicorn
import sys
import json
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response

app = FastAPI()

origins = [
    "http://localhost:5174",
    "http://localhost:5173",
    "http://localhost:9000",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "System health is good"}

@app.post("/tal_to_llm")
async def tal_to_llm():
    pass

def get_chat_response(user_message):
    messages = load_messages()


def load_messages():
    messages = []
    file = 'database.json'

    empty = os.stat(file).st_size == 0

    if not empty:
        with open(file) as db_file:
            data = json.load(db_file)
            for item in data:
                messages.append(item)
    else:
        messages.append(
            {role": "system", "content": "You are a expert in taking technical interviews in Data Analytics domain.
            You have to ask technical questions to the user from the below set of Question- Answer list, but remember you can ask subsequent question from your end also if the user is giving answer for the asked question, but if you think that you need to ask a new question then only pick the new question from the given Question-Answer list only.
            Ask only relevant question based on the provided user info.
            User-Info: {user_info}.
            Question-Answer List: {question_answer_list}.
            Keep your question under 40 words and be funny sometimes."""}
        )
    return messages


@app.get("/clear")
async def clear_history():
    file = 'database.json'
    open(file, 'w')
    return {"message": "Chat history has been cleared"}

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)