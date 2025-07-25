import streamlit as st
import requests
import json
from typing import List, Dict

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Smart Document Search",
    page_icon="📚",
    layout="wide"
)

def check_api_status():
    try:
        response = requests.get(f"{API_BASE_URL}/status")
        return response.status_code == 200
    except:
        return False

def upload_files(files):
    files_data = []
    for file in files:
        files_data.append(("files", (file.name, file.getvalue(), file.type)))
    
    try:
        response = requests.post(f"{API_BASE_URL}/upload", files=files_data)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Upload failed: {str(e)}")
        return None

def search_documents(query: str, search_type: str = "hybrid", k: int = 5):
    try:
        response = requests.post(f"{API_BASE_URL}/search", json={
            "query": query,
            "search_type": search_type,
            "k": k
        })
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return None

def ask_question(question: str, k: int = 5):
    try:
        response = requests.post(f"{API_BASE_URL}/ask", json={
            "question": question,
            "k": k
        })
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Question failed: {str(e)}")
        return None

def clear_documents():
    try:
        response = requests.delete(f"{API_BASE_URL}/clear")
        return response.status_code == 200
    except Exception as e:
        st.error(f"Clear failed: {str(e)}")
        return False

def main():
    st.title("📚 Smart Document Search")
    
    # Check API status
    if not check_api_status():
        st.error("🔴 API server is not running. Please start the FastAPI server first.")
        st.code("python -m uvicorn app.main:app --reload")
        return
    
    st.success("🟢 API server is running")
    
    # Get current status
    try:
        status_response = requests.get(f"{API_BASE_URL}/status")
        if status_response.status_code == 200:
            status = status_response.json()
            st.sidebar.metric("Documents Loaded", status.get("documents_count", 0))
            st.sidebar.metric("Index Ready", "✅" if status.get("index_ready") else "❌")
    except:
        pass
    
    # Clear button in sidebar
    if st.sidebar.button("🗑️ Clear All Documents", type="secondary"):
        if clear_documents():
            st.sidebar.success("Documents cleared!")
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "🔍 Search", "❓ Ask"])
    
    with tab1:
        st.header("Upload Documents")
        st.write("Upload PDF or text files to add them to the search index.")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'txt']
        )
        
        if uploaded_files:
            if st.button("Upload Files", type="primary"):
                with st.spinner("Processing files..."):
                    result = upload_files(uploaded_files)
                    
                if result:
                    st.success(f"✅ {result['message']}")
                    st.json(result)
                else:
                    st.error("Upload failed. Please check the API logs.")
    
    with tab2:
        st.header("Search Documents")
        st.write("Search through your uploaded documents using keyword, semantic, or hybrid search.")
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            search_query = st.text_input("Enter your search query", placeholder="e.g., machine learning algorithms")
        
        with col2:
            search_type = st.selectbox("Search Type", ["hybrid", "semantic", "keyword"])
        
        with col3:
            top_k = st.number_input("Top K Results", min_value=1, max_value=20, value=5)
        
        if search_query and st.button("Search", type="primary"):
            with st.spinner("Searching..."):
                results = search_documents(search_query, search_type, top_k)
            
            if results and results.get("results"):
                st.success(f"Found {results['total_results']} results")
                
                for i, result in enumerate(results["results"]):
                    with st.expander(f"Result {i+1} - {result.get('filename', 'Unknown')} (Score: {result.get('score', 0):.3f})"):
                        st.write(result["text"])
                        st.caption(f"Chunk ID: {result.get('id', 'N/A')}")
            else:
                st.warning("No results found.")
    
    with tab3:
        st.header("Ask Questions")
        st.write("Ask questions about your documents and get AI-generated answers with sources.")
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            question = st.text_input("Enter your question", placeholder="e.g., What are the main benefits of machine learning?")
        
        with col2:
            context_k = st.number_input("Context Chunks", min_value=1, max_value=10, value=5)
        
        if question and st.button("Ask Question", type="primary"):
            with st.spinner("Thinking..."):
                result = ask_question(question, context_k)
            
            if result:
                st.subheader("Answer:")
                st.write(result["answer"])
                
                if result.get("sources"):
                    st.subheader("Sources:")
                    for i, source in enumerate(result["sources"]):
                        with st.expander(f"Source {i+1} - {source.get('filename', 'Unknown')} (Score: {source.get('score', 0):.3f})"):
                            st.write(source.get("text_preview", "No preview available"))
            else:
                st.error("Failed to get an answer. Please try again.")

if __name__ == "__main__":
    main()