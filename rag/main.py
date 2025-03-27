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





@app.get("/clear")
async def clear_history():
    file = 'database.json'
    open(file, 'w')
    return {"message": "Chat history has been cleared"}

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)