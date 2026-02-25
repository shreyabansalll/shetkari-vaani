"""
Shetkari Vaani - Agricultural Voice Bot Backend
FastAPI backend for WhatsApp agricultural voice bot with LangChain RAG.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from pydantic import BaseModel

load_dotenv()

app = FastAPI(
    title="Shetkari Vaani",
    description="Agricultural voice bot backend for farmer queries in Marathi/Hindi",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CHROMA_PERSIST_DIR = Path("chroma_db")
CHROMA_COLLECTION = "shetkari_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Initialize vector store (lazy init)
vectorstore = None


def get_embeddings():
    """Return local HuggingFace embeddings."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vectorstore():
    """Get or create ChromaDB vector store."""
    global vectorstore
    if vectorstore is None:
        vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_PERSIST_DIR),
        )
    return vectorstore


class QueryRequest(BaseModel):
    text: str


class IngestRequest(BaseModel):
    folder_path: str


class QueryResponse(BaseModel):
    response: str


class IngestResponse(BaseModel):
    message: str
    files_loaded: int
    chunks_created: int


def format_docs(docs):
    """Format retrieved documents for context."""
    return "\n\n".join(doc.page_content for doc in docs)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process farmer query and return agricultural advice.
    Input text can be in Marathi or Hindi.
    Response is in the same language, under 100 words, farmer-friendly.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="text field cannot be empty")

    try:
        vectordb = get_vectorstore()
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are Shetkari Vaani - a friendly agricultural assistant for farmers in Vidarbha, Maharashtra.

CRITICAL RULES:
1. ALWAYS respond in the SAME language as the question.
   - If question is in Marathi, respond in Marathi
   - If question is in Hindi, respond in Hindi
   - If question is in English, respond in English
2. Keep response under 100 words.
3. Use simple language a farmer can understand.
4. Only use information from the provided context.
5. If context does not have the answer say: Kripaya najik KVK ya agriculture office la sampark kara.
6. Always include relevant helpline numbers if available in context.""",
                ),
                ("human", "संदर्भ:\n{context}\n\nप्रश्न: {question}\n\nउत्तर द्या:"),
            ]
        )

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
        )

        chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke(request.text.strip())
        return QueryResponse(response=response.strip())

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    Load knowledge base from a folder of text files.
    Reads all .txt files from the given folder path.
    """
    # Resolve relative paths from project root
    folder_path = Path(request.folder_path)
    if not folder_path.is_absolute():
        project_root = Path(__file__).resolve().parent
        folder = project_root / folder_path
    else:
        folder = folder_path
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder}")
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {folder}")

    txt_files = list(folder.glob("**/*.txt"))
    if not txt_files:
        raise HTTPException(
            status_code=404,
            detail=f"No .txt files found in {folder}",
        )

    documents = []
    for filepath in txt_files:
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            documents.append(
                Document(page_content=content, metadata={"source": str(filepath)})
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error reading {filepath}: {str(e)}",
            )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", "।", "। ", " ", ""],
    )
    splits = text_splitter.split_documents(documents)

    vectordb = get_vectorstore()
    vectordb.add_documents(splits)
    # Chroma auto-persists when persist_directory is set

    return IngestResponse(
        message=f"Knowledge base loaded from {folder}",
        files_loaded=len(txt_files),
        chunks_created=len(splits),
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "Shetkari Vaani"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
