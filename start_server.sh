# cd /home/ubuntu/podcast-summarizer-engine
# export HF_TOKEN=$HF_TOKEN
# export AWS_ACCESS_KEY_ID=$HF_SECRET_AWS_ACCESS_KEY_ID
# export AWS_SECRET_ACCESS_KEY=$HF_SECRET_AWS_SECRET_ACCESS_KEY
cd /app
uvicorn fastapi_news_bot:app --host 0.0.0.0 --port 7860

# uvicorn fastapi_news_bot:app --host 0.0.0.0 --port 8000 --reload