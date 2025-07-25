import streamlit as st
import os
import io
import tempfile
import pickle
from typing import List, Dict
import numpy as np
import faiss
import PyPDF2
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration - try Streamlit secrets first, then environment variables
try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    EMBEDDING_MODEL = st.secrets.get("EMBEDDING_MODEL", "gemini-embedding-001")
    CHAT_MODEL = st.secrets.get("CHAT_MODEL", "gemini-1.5-flash")
    CHUNK_SIZE = int(st.secrets.get("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(st.secrets.get("CHUNK_OVERLAP", "50"))
    TOP_K = int(st.secrets.get("TOP_K", "5"))
except:
    # Fallback to environment variables if secrets not available
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-1.5-flash")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    TOP_K = int(os.getenv("TOP_K", "5"))

# Initialize Google AI client
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    client_configured = True
else:
    client_configured = False

# Page config
st.set_page_config(
    page_title="Smart Document Search",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = []

class SimpleVectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.dimension = None  # Will be set dynamically based on first embedding
    
    def add_documents(self, chunks: List[Dict[str, str]]):
        if not chunks:
            return
        
        if not client_configured:
            st.error("Google API key not found. Please set GOOGLE_API_KEY in your environment.")
            return
            
        # Get embeddings
        texts = [chunk['text'] for chunk in chunks]
        
        try:
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
        except Exception as e:
            st.error(f"Error getting embeddings: {str(e)}")
            return
        
        # Set dimension from first embedding if not set
        if self.dimension is None:
            self.dimension = len(embeddings[0])
        
        # Initialize or update index
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.chunks = []
        
        self.chunks.extend(chunks)
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)
        
        self.index.add(embeddings_array)
    
    def search(self, query: str, k: int = TOP_K) -> List[Dict]:
        if self.index is None or len(self.chunks) == 0:
            return []
        
        if not client_configured:
            return []
        
        try:
            # Get query embedding
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=query,
                task_type="retrieval_query"
            )
            query_embedding = result['embedding']
            
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)
            
            scores, indices = self.index.search(query_vector, min(k, len(self.chunks)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx].copy()
                    chunk['score'] = float(score)
                    results.append(chunk)
            
            return results
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []
    
    def keyword_search(self, query: str, k: int = TOP_K) -> List[Dict]:
        if not self.chunks:
            return []
        
        query_lower = query.lower()
        results = []
        
        for chunk in self.chunks:
            text_lower = chunk['text'].lower()
            if query_lower in text_lower:
                score = text_lower.count(query_lower) / len(text_lower.split())
                chunk_copy = chunk.copy()
                chunk_copy['score'] = score
                results.append(chunk_copy)
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]

def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

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

def process_file(filename: str, file_content: bytes) -> List[Dict[str, str]]:
    if filename.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_content)
    elif filename.lower().endswith('.txt'):
        text = file_content.decode('utf-8')
    else:
        st.error(f"Unsupported file type: {filename}")
        return []
    
    if not text.strip():
        st.error(f"No text extracted from {filename}")
        return []
    
    chunks = chunk_text(text)
    
    for i, chunk in enumerate(chunks):
        chunk['id'] = f"{filename}_chunk_{i}"
        chunk['filename'] = filename
    
    return chunks

def ask_question(question: str, context_chunks: List[str]) -> str:
    if not client_configured:
        return "Google API key not configured."
    
    if not context_chunks:
        return "No relevant context found to answer the question."
    
    context = "\n\n".join(context_chunks)
    
    prompt = f"""You are a helpful assistant that answers questions based only on the provided context. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
    
    try:
        model = genai.GenerativeModel(CHAT_MODEL)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error getting answer: {str(e)}"

def main():
    st.title("📚 Smart Document Search")
    
    # Check API key
    if not GOOGLE_API_KEY:
        st.error("🔴 Google API key not found. Please set GOOGLE_API_KEY in your environment variables.")
        st.info("Create a `.env` file with: `GOOGLE_API_KEY=your_api_key_here`")
        return
    
    st.success("🟢 Google AI API key loaded")
    
    # Initialize vector store
    if st.session_state.vector_store is None:
        st.session_state.vector_store = SimpleVectorStore()
    
    # Sidebar info
    st.sidebar.metric("Documents Loaded", len(st.session_state.chunks))
    if st.sidebar.button("🗑️ Clear All Documents"):
        st.session_state.vector_store = SimpleVectorStore()
        st.session_state.chunks = []
        st.rerun()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "🔍 Search", "❓ Ask"])
    
    with tab1:
        st.header("Upload Documents")
        st.write("Upload PDF or text files to add them to the search index.")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'txt'],
            help="Select one or more PDF or text files"
        )
        
        if uploaded_files:
            if st.button("Process Files", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                all_chunks = []
                processed_files = []
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    file_content = file.read()
                    chunks = process_file(file.name, file_content)
                    
                    if chunks:
                        all_chunks.extend(chunks)
                        processed_files.append({
                            "filename": file.name,
                            "chunks": len(chunks),
                            "size": len(file_content)
                        })
                
                if all_chunks:
                    status_text.text("Creating embeddings...")
                    st.session_state.vector_store.add_documents(all_chunks)
                    st.session_state.chunks.extend(all_chunks)
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Processing complete!")
                    
                    st.success(f"Processed {len(processed_files)} files with {len(all_chunks)} chunks total")
                    
                    # Show summary
                    for file_info in processed_files:
                        st.write(f"📄 {file_info['filename']}: {file_info['chunks']} chunks ({file_info['size']:,} bytes)")
                else:
                    st.error("No content could be extracted from the uploaded files.")
    
    with tab2:
        st.header("Search Documents")
        
        if not st.session_state.chunks:
            st.info("Upload some documents first to enable search.")
            return
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            search_query = st.text_input("Enter your search query", placeholder="e.g., machine learning algorithms")
        
        with col2:
            search_type = st.selectbox("Search Type", ["semantic", "keyword"])
        
        with col3:
            top_k = st.number_input("Results", min_value=1, max_value=20, value=5)
        
        if search_query:
            with st.spinner("Searching..."):
                if search_type == "semantic":
                    results = st.session_state.vector_store.search(search_query, top_k)
                else:
                    results = st.session_state.vector_store.keyword_search(search_query, top_k)
            
            if results:
                st.success(f"Found {len(results)} results")
                
                for i, result in enumerate(results):
                    with st.expander(f"📄 {result.get('filename', 'Unknown')} (Score: {result.get('score', 0):.3f})"):
                        st.write(result["text"])
                        st.caption(f"Chunk: {result.get('id', 'N/A')}")
            else:
                st.warning("No results found.")
    
    with tab3:
        st.header("Ask Questions")
        
        if not st.session_state.chunks:
            st.info("Upload some documents first to enable Q&A.")
            return
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            question = st.text_input("Ask a question about your documents", placeholder="e.g., What are the main points discussed?")
        
        with col2:
            context_k = st.number_input("Context", min_value=1, max_value=10, value=3, help="Number of relevant chunks to use as context")
        
        if question:
            with st.spinner("Finding relevant information and generating answer..."):
                # Get relevant chunks
                relevant_chunks = st.session_state.vector_store.search(question, context_k)
                
                if relevant_chunks:
                    context_texts = [chunk['text'] for chunk in relevant_chunks]
                    answer = ask_question(question, context_texts)
                    
                    st.subheader("🤖 Answer:")
                    st.write(answer)
                    
                    st.subheader("📚 Sources:")
                    for i, chunk in enumerate(relevant_chunks):
                        with st.expander(f"Source {i+1}: {chunk.get('filename', 'Unknown')} (Score: {chunk.get('score', 0):.3f})"):
                            preview = chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text']
                            st.write(preview)
                else:
                    st.warning("No relevant information found to answer your question.")

if __name__ == "__main__":
    main()