from fastapi import FastAPI
from pydantic import BaseModel
from pdf_utils import extract_text_from_pdf
from embeddings import chunk_text, create_vector_store, retrieve_relevant_chunks
from ai_engine import answer_question, summarize_text
from topic_detector import extract_topics
from difficulty import estimate_difficulty

app = FastAPI()

class PaperRequest(BaseModel):
    pdf_url: str
    questions: list[str] = []

@app.post("/analyze-paper")
def analyze_paper(request: PaperRequest):
    text = extract_text_from_pdf(request.pdf_url)
    chunks = chunk_text(text)
    index, embeddings = create_vector_store(chunks)

    summary = summarize_text(text)
    topics = extract_topics(text)
    level = estimate_difficulty(text)

    answers = []
    for q in request.questions:
        context = retrieve_relevant_chunks(q, index, chunks)
        ans = answer_question(context, q)
        answers.append(ans)

    return {
        "summary": summary,
        "topics": topics,
        "difficulty_level": level,
        "answers": answers
    }
