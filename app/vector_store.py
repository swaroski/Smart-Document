import faiss
import numpy as np
import pickle
import os
import tempfile
from typing import List, Dict, Tuple
from app.config import FAISS_INDEX_PATH, TOP_K
from app.openai_client import get_embeddings

class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.dimension = 1536  # text-embedding-3-small dimension
        
    def create_index(self, chunks: List[Dict[str, str]]):
        if not chunks:
            return
            
        texts = [chunk['text'] for chunk in chunks]
        embeddings = get_embeddings(texts)
        
        self.chunks = chunks
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)  # Normalize for cosine similarity
        
        self.index.add(embeddings_array)
    
    def add_documents(self, new_chunks: List[Dict[str, str]]):
        if not new_chunks:
            return
            
        if self.index is None:
            self.create_index(new_chunks)
            return
            
        texts = [chunk['text'] for chunk in new_chunks]
        embeddings = get_embeddings(texts)
        
        self.chunks.extend(new_chunks)
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)
        
        self.index.add(embeddings_array)
    
    def search(self, query: str, k: int = TOP_K) -> List[Dict[str, any]]:
        if self.index is None or len(self.chunks) == 0:
            return []
            
        query_embedding = get_embeddings([query])[0]
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
    
    def keyword_search(self, query: str, k: int = TOP_K) -> List[Dict[str, any]]:
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
    
    def hybrid_search(self, query: str, k: int = TOP_K, semantic_weight: float = 0.7) -> List[Dict[str, any]]:
        semantic_results = self.search(query, k * 2)
        keyword_results = self.keyword_search(query, k * 2)
        
        combined_scores = {}
        
        for result in semantic_results:
            chunk_id = result['id']
            combined_scores[chunk_id] = {
                'chunk': result,
                'semantic_score': result['score'],
                'keyword_score': 0
            }
        
        for result in keyword_results:
            chunk_id = result['id']
            if chunk_id in combined_scores:
                combined_scores[chunk_id]['keyword_score'] = result['score']
            else:
                combined_scores[chunk_id] = {
                    'chunk': result,
                    'semantic_score': 0,
                    'keyword_score': result['score']
                }
        
        final_results = []
        for chunk_id, scores in combined_scores.items():
            final_score = (semantic_weight * scores['semantic_score'] + 
                          (1 - semantic_weight) * scores['keyword_score'])
            chunk = scores['chunk'].copy()
            chunk['score'] = final_score
            final_results.append(chunk)
        
        final_results.sort(key=lambda x: x['score'], reverse=True)
        return final_results[:k]
    
    def save_index(self, path: str = None):
        if self.index is None:
            return
        
        # For serverless environments, use temp directory
        if path is None:
            path = os.getenv("FAISS_INDEX_PATH", tempfile.gettempdir())
            
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
    
    def load_index(self, path: str = None):
        if path is None:
            path = os.getenv("FAISS_INDEX_PATH", tempfile.gettempdir())
            
        index_file = os.path.join(path, "index.faiss")
        chunks_file = os.path.join(path, "chunks.pkl")
        
        if os.path.exists(index_file) and os.path.exists(chunks_file):
            self.index = faiss.read_index(index_file)
            
            with open(chunks_file, "rb") as f:
                self.chunks = pickle.load(f)
                
            return True
        return False

# Global instance - will be recreated for each serverless function call
vector_store = VectorStore()