import io
from typing import List, Dict
import PyPDF2
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

def extract_text_from_pdf(file_content: bytes) -> str:
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_txt(file_content: bytes) -> str:
    return file_content.decode('utf-8')

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict[str, str]]:
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        chunks.append({
            "text": chunk_text,
            "start_index": i,
            "end_index": min(i + chunk_size, len(words))
        })
        
        if i + chunk_size >= len(words):
            break
    
    return chunks

def process_document(filename: str, file_content: bytes) -> List[Dict[str, str]]:
    if filename.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_content)
    elif filename.lower().endswith('.txt'):
        text = extract_text_from_txt(file_content)
    else:
        raise ValueError(f"Unsupported file type: {filename}")
    
    chunks = chunk_text(text)
    
    for i, chunk in enumerate(chunks):
        chunk['id'] = f"{filename}_chunk_{i}"
        chunk['filename'] = filename
    
    return chunks