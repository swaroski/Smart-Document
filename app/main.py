from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
import logging

from app.document_loader import process_document
from app.vector_store import vector_store
from app.openai_client import ask_question
from app.config import TOP_K

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Document Search API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = TOP_K
    search_type: Optional[str] = "hybrid"  # "semantic", "keyword", "hybrid"

class SearchResponse(BaseModel):
    results: List[dict]
    total_results: int

class AskRequest(BaseModel):
    question: str
    k: Optional[int] = TOP_K

class AskResponse(BaseModel):
    answer: str
    sources: List[dict]

@app.on_event("startup")
async def startup_event():
    vector_store.load_index()
    logger.info("Vector store loaded successfully")

@app.get("/")
async def root():
    return {"message": "Smart Document Search API", "status": "running"}

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    try:
        uploaded_files = []
        all_chunks = []
        
        for file in files:
            if not file.filename:
                continue
                
            content = await file.read()
            
            try:
                chunks = process_document(file.filename, content)
                all_chunks.extend(chunks)
                uploaded_files.append({
                    "filename": file.filename,
                    "chunks_count": len(chunks),
                    "size": len(content)
                })
                logger.info(f"Processed {file.filename}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error processing {file.filename}: {str(e)}")
        
        if all_chunks:
            vector_store.add_documents(all_chunks)
            vector_store.save_index()
            logger.info(f"Added {len(all_chunks)} chunks to vector store")
        
        return {
            "message": f"Successfully uploaded {len(uploaded_files)} files",
            "files": uploaded_files,
            "total_chunks": len(all_chunks)
        }
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    try:
        if request.search_type == "semantic":
            results = vector_store.search(request.query, request.k)
        elif request.search_type == "keyword":
            results = vector_store.keyword_search(request.query, request.k)
        else:  # hybrid
            results = vector_store.hybrid_search(request.query, request.k)
        
        return SearchResponse(
            results=results,
            total_results=len(results)
        )
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/ask", response_model=AskResponse)
async def ask_question_endpoint(request: AskRequest):
    try:
        # Get relevant chunks
        relevant_chunks = vector_store.hybrid_search(request.question, request.k)
        
        if not relevant_chunks:
            return AskResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[]
            )
        
        # Extract text from chunks for context
        context_texts = [chunk['text'] for chunk in relevant_chunks]
        
        # Ask OpenAI
        answer = ask_question(request.question, context_texts)
        
        # Prepare sources
        sources = []
        for chunk in relevant_chunks:
            sources.append({
                "filename": chunk.get("filename", "Unknown"),
                "text_preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                "score": chunk.get("score", 0)
            })
        
        return AskResponse(
            answer=answer,
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Ask error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")

@app.get("/status")
async def get_status():
    return {
        "status": "running",
        "documents_count": len(vector_store.chunks),
        "index_ready": vector_store.index is not None
    }

@app.delete("/clear")
async def clear_documents():
    try:
        vector_store.chunks = []
        vector_store.index = None
        vector_store.save_index()
        
        return {"message": "All documents cleared successfully"}
        
    except Exception as e:
        logger.error(f"Clear error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)