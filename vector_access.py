from langchain_chroma import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from shared_config import CHROMA_PERSIST_DIR, EMBED_MODEL_NAME
import boto3
import tempfile
import zipfile
import os
from shared_config import S3_BUCKET, S3_VECTOR_DB

def download_chroma_db_from_s3(temp_dir: str):
    s3 = boto3.client("s3")
    zip_path = os.path.join(temp_dir, "vector_db.zip")

    # Download vector DB zip
    s3.download_file(S3_BUCKET, f"{S3_VECTOR_DB}vector_db.zip", zip_path)

    # Extract
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    return temp_dir

# def initialize_vector_db() -> Chroma:
#     embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
#     return Chroma(
#         persist_directory=CHROMA_PERSIST_DIR,
#         embedding_function=embedding_model,
#         collection_name="news_articles"
#     )

def initialize_vector_db() -> Chroma:
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    temp_dir = tempfile.mkdtemp()
    vector_db_path = download_chroma_db_from_s3(temp_dir)

    return Chroma(
        persist_directory=vector_db_path,
        embedding_function=embedding_model,
        collection_name="news_articles"
    )

def extract_documents(vectordb: Chroma) -> list[Document]:
    raw = vectordb.get()
    return [Document(page_content=doc, metadata=meta) for doc, meta in zip(raw["documents"], raw["metadatas"])]

def build_ensemble_retriever(docs, vectordb, top_k=5) -> EnsembleRetriever:
    bm25 = BM25Retriever.from_documents(docs); bm25.k = top_k
    vector_retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
    return EnsembleRetriever(retrievers=[vector_retriever, bm25], weights=[0.5, 0.5])