import os
from dotenv import load_dotenv
import traceback
import tempfile

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")


#Step 2: Create a Pinecone index for Vector store

from pinecone import Pinecone, ServerlessSpec
import os

pc = Pinecone(api_key =PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

index_name = "insurepal-index"
if index_name not in [idx.name for idx in pc.list_indexes()]: 
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    pc.create_index(name=index_name, dimension=1536, metric="cosine", spec=spec)

spec = ServerlessSpec(cloud="aws", region="us-east-1")
pinecone_index = pc.Index(index_name, spec=spec)


#Step 3: Initializing a PineconeVectorStore with our pinecone index. This object will serve as the storage and retrieval interface for our document embeddings in Pinecone's vector database
from llama_index.vector_stores.pinecone import PineconeVectorStore

namespace = ''

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)


#Step 4: Basic File Imports
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from llama_index.core import VectorStoreIndex, Settings
from llama_index.readers.file import PDFReader, MboxReader, DocxReader
from io import BytesIO  


#Step 5: Basic CORS handling to avoid any issues with CORS (allowing post requests from all urls change to your frontend later)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Step 6: Handling File Upload call and Indexing the uploaded Document along with the query call
import tempfile
from pathlib import Path

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        ext = file.filename.lower().split(".")[-1]
        content = await file.read()
        documents = None

        if ext == "pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            reader = PDFReader()
            documents = reader.load_data(Path(tmp_path))  # <- positional Path arg
            os.unlink(tmp_path)

        elif ext == "docx":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            reader = DocxReader()
            documents = reader.load_data(Path(tmp_path))
            os.unlink(tmp_path)

        elif ext in ["mbox", "eml"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            reader = MboxReader()
            documents = reader.load_data(Path(tmp_path))
            os.unlink(tmp_path)

        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        VectorStoreIndex.from_documents(documents, vector_store=vector_store)
        return JSONResponse(content={"message": f"File '{file.filename}' uploaded and indexed successfully"})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to process file '{file.filename}': {str(e)}")



@app.post("/query/")
async def query_index(question: str = Form(...)):
    try:
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        query_engine = index.as_query_engine()
        answer = query_engine.query(question)
        return JSONResponse(content={"answer": str(answer)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")