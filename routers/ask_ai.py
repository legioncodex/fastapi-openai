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

class QuestionRequest(BaseModel):
    question:str

@router.post("/ask-ai")
def ask_ai(request: QuestionRequest):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "user", "content": request.question}
        ]
    )

    return {
        "answer": response.choices[0].message.content
    }
