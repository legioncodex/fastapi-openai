# main.py
from fastapi import FastAPI
from routers.ask_ai import router as ask_ai
from routers.chat import router as chat
from routers.upload import router as upload

app = FastAPI(title="Production RAG API")

app.include_router(ask_ai, prefix="", tags=["Ask AI"])
app.include_router(chat, prefix="", tags=["Conversational Chat"])
app.include_router(upload, prefix="/upload", tags=["PDF Processing"])

@app.get("/")
def read_root():
    return {"status": "Server running smoothly"}


        