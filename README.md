# Smart Document Search App

A simple single-file Streamlit app for document search and Q&A using OpenAI and FAISS.

## Features

- **Upload PDFs and text files**
- **Semantic and keyword search**
- **AI-powered Q&A with sources**
- **Simple one-file deployment**

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key
```bash
# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### 3. Run the App
```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

## Usage

1. **Upload Tab**: Upload PDF or text files
2. **Search Tab**: Search through documents (semantic or keyword)
3. **Ask Tab**: Ask questions and get AI answers with sources

## Deploy on Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository and set main file: `app.py`
4. Add `OPENAI_API_KEY` in app settings
5. Deploy!

## Environment Variables

- `OPENAI_API_KEY` (required): Your OpenAI API key
- `EMBEDDING_MODEL` (optional): Default `text-embedding-3-small`
- `CHAT_MODEL` (optional): Default `gpt-4o-mini`
- `CHUNK_SIZE` (optional): Default `500`
- `TOP_K` (optional): Default `5`

That's it! Simple and ready to use.