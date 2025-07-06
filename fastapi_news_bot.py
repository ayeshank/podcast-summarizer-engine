from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse, FileResponse
from vector_access import initialize_vector_db, extract_documents, build_ensemble_retriever
from summarizer import summarize, generate_script, synthesize_podcast
import asyncio, os
from shared_config import PODCAST_AUDIO_DIR
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or set to ["https://your-frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryInput(BaseModel):
    query: str

@app.get("/health")
def health_check():
    return {"status": "running"}

@app.post("/query")
async def query_pipeline(input: QueryInput):
    vectordb = initialize_vector_db()
    docs = extract_documents(vectordb)
    retriever = build_ensemble_retriever(docs, vectordb)
    results = retriever.get_relevant_documents(input.query)
    if not results:
        return JSONResponse(status_code=404, content={"message": "No relevant data found."})
    
    merged = "\n".join([doc.page_content for doc in results])
    summary = summarize(merged)
    script = generate_script(summary)
    audio_file = await synthesize_podcast(script)
    
    return {"summary": summary, "script": script, "audio_file": audio_file}

@app.get("/audio/{filename}")
def get_audio(filename: str):
    file_path = os.path.join(PODCAST_AUDIO_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"message": "File not found."})
    return FileResponse(file_path, media_type="audio/mpeg")

@app.get("/gpu")
def check_gpu():
    import torch
    return {"cuda_available": torch.cuda.is_available()}
