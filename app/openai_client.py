from openai import OpenAI
from typing import List
from app.config import OPENAI_API_KEY, EMBEDDING_MODEL, CHAT_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

def get_embeddings(texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [data.embedding for data in response.data]

def ask_question(question: str, context_chunks: List[str]) -> str:
    context = "\n\n".join(context_chunks)
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions based only on the provided context. If the answer cannot be found in the context, say so."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]
    
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.1
    )
    
    return response.choices[0].message.content