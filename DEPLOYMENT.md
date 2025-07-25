# Deployment Guide

## Option 1: Split Deployment (Recommended)

### Backend: Vercel (FastAPI)
```bash
# 1. Install Vercel CLI
npm i -g vercel

# 2. Deploy backend
vercel --prod

# 3. Set environment variables in Vercel dashboard:
# - OPENAI_API_KEY
# - EMBEDDING_MODEL
# - CHAT_MODEL
```

### Frontend: Streamlit Cloud
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository, set main file: `ui/streamlit_app.py`
4. Update `API_BASE_URL` in `ui/streamlit_app.py` to your Vercel URL
5. Add environment variables (not needed for frontend)

## Option 2: Single VPS Deployment

### Digital Ocean/AWS/GCP
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-key"

# Run with PM2 or systemd
pm2 start "uvicorn app.main:app --host 0.0.0.0 --port 8000"
pm2 start "streamlit run ui/streamlit_app.py --server.port 8501"
```

## Option 3: Docker Deployment

### Create Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  smartdoc:
    build: .
    ports:
      - "8000:8000"
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./faiss_index:/app/faiss_index
```

## Important Notes for Vercel

⚠️ **Serverless Limitations:**
- FAISS index won't persist between function calls
- Each request rebuilds the index (slow for large datasets)
- 30-second function timeout

✅ **Better for Vercel:**
- Use external vector database (Pinecone, Weaviate)
- Or deploy full stack on VPS/container platform