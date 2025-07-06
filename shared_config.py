import os
BASE_PATH = "/home/ubuntu/shared_data/"
CHROMA_PERSIST_DIR = f"{BASE_PATH}/vector_db/"
PODCAST_AUDIO_DIR = f"{BASE_PATH}/podcasts/"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLaMA_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")
# S3 Configuration
S3_BUCKET = "llm-instance-bucket"
S3_SCRAPED_ARTICLES = "scraped_articles/"
S3_VECTOR_DB = "vector_db/"
S3_PODCASTS = "podcasts/"