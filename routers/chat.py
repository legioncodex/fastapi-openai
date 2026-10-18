from fastapi import FastAPI,APIRouter  
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
router = APIRouter()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

from pydantic import BaseModel

class ChatRequest(BaseModel):
    message:str

conversation = []

instructions = """
you are a FastAPI Tutor
    - FastAPI
    - API Development
    - Pydantic
if the user asks something unrelated to these topics, 
response: I can only help with FastAPI
"""

@router.post("/chat")
def ask_ai(request: ChatRequest):
    conversation.append({
        "role": "user",
        "content": request.message
    })

    messages_payload = [{
        "role": "system", 
        "content": instructions
    }] + conversation

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages_payload
    )

    answer = response.choices[0].message.content

    conversation.append({
        "role":"assistant",
        "content":answer
    })

    return {
        "res": answer
    }