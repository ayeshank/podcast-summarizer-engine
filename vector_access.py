from langchain_community.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from shared_config import CHROMA_PERSIST_DIR, EMBED_MODEL_NAME

def initialize_vector_db() -> Chroma:
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
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