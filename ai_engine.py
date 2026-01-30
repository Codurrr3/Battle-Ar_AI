from transformers import pipeline

qa_generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1
)

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1
)

def answer_question(context, question):
    prompt = f"""
    You are answering questions from a research paper.
Use ONLY the provided context to answer.
Search carefully — the answer may be paraphrased.
If the answer is partially present, give the closest correct answer.
Only say "Not found in document" if the information truly does not exist.

Context:
{context}

Question: {question}
If not found, say: Not found in document.
"""
    result = qa_generator(prompt, max_new_tokens=80, do_sample=False)
    return result[0]["generated_text"].strip()

def summarize_text(text):
    summary = summarizer(text[:2000], max_length=150, min_length=60, do_sample=False)
    return summary[0]["summary_text"]
