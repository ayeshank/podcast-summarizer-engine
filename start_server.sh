# cd /home/ubuntu/podcast-summarizer-engine
export AWS_ACCESS_KEY_ID=$HF_SECRET_AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$HF_SECRET_AWS_SECRET_ACCESS_KEY
uvicorn fastapi_news_bot:app --host 0.0.0.0 --port 8000 --reload