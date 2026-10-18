from fastapi import APIRouter, UploadFile, File
from groq import Groq
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import chromadb 
from pypdf import PdfReader
import uuid

load_dotenv()

router = APIRouter()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="document"
)

UPLOAD_DIR="upload"
os.makedirs(UPLOAD_DIR,exist_ok=True)

def extract(file_path: str):
    reader = PdfReader(file_path)
    text=""

    for page in reader.pages:
        page_text=page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text:str,chunk_size:int=1000,chunk_overlap:int=200):
    chunks = []

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

def text_embedding(text:str):
    response = client.embeddings.create(
        model="nomic-embed-text-v1.5",
        input=text
    )
    return response.data[0].embedding

@router.post("/upload")
def upload_pdf(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR,file.filename)

    content = file.file.read()

    with open(file_path,"wb") as f:
        f.write(content)

    text = extract(file_path)
    chunks = chunk_text(text)

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"filename": file.filename} for _ in chunks]

    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas
    )

    return {
        "message": "upload Successfully",
        "chunks": len(chunks)
    }

class QuestionRequest(BaseModel):
    question: str

@router.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question 

    results = collection.query(
        query_texts=[question],n_results=3
    )

    documents = results["documents"][0]
    
    context = "\n\n".join(documents)
    
    prompt = f"""
    You are a helpful PDF assistant.
    
    Answer the users questions using only the information provided in the context 

    if the answer is not available in the context
    say that the information is not available in the document
    Context: {context}
    Question: {question}
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content

    return {
        "question":question,
        "answer": answer
    }