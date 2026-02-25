# Shetkari Vaani

WhatsApp agricultural voice bot backend. FastAPI API for farmer queries in Marathi or Hindi using LangChain RAG with ChromaDB.

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API keys**  
   Copy `.env.example` to `.env` and add your Groq API key:
   ```bash
   cp .env.example .env
   # Edit .env: GROQ_API_KEY=your-groq-api-key
   ```

3. **Run the server**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## Endpoints

### `POST /query`
Process a transcribed farmer query.

**Request:**
```json
{
  "text": "भात लावण्यासाठी कोणता महिना चांगला?"
}
```

**Response:**
```json
{
  "response": "भात लावण्यासाठी जून महिना योग्य आहे..."
}
```

### `POST /ingest`
Load knowledge base from a folder of `.txt` files.

**Request:**
```json
{
  "folder_path": "knowledge_base"
}
```

Relative paths are resolved from the project root. Use absolute paths for external folders.

**Response:**
```json
{
  "message": "Knowledge base loaded from knowledge_base",
  "files_loaded": 1,
  "chunks_created": 5
}
```

### `GET /health`
Health check.

## Usage

1. Run the server.
2. Call `/ingest` with `{"folder_path": "knowledge_base"}` to load the sample knowledge.
3. Call `/query` with farmer questions in Marathi or Hindi.
4. Responses are in the same language, under 100 words, and farmer-friendly.
