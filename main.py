from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agent import answer

app = FastAPI(title="Titanic Dataset Chat Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo/assignment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Titanic Chat Agent Running"}


@app.post("/ask")
def ask(req: AskRequest):
    q = (req.question or "").strip()
    if not q:
        return {"type": "text", "content": "Please type a question."}

    try:
        return answer(q)
    except Exception as e:
        return {"type": "text", "content": f"⚠️ Server error: {e}"}