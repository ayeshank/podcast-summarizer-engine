# Use GPU-compatible base
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Install dependencies
RUN apt-get update && apt-get install -y ffmpeg git && \
    pip install --upgrade pip

WORKDIR /app

# Copy all files
COPY . .

# Install requirements
RUN pip install -r requirements.txt

# Expose port
EXPOSE 7860

# Launch FastAPI (adjust if your entrypoint is different)
CMD ["uvicorn", "fastapi_news_bot:app", "--host", "0.0.0.0", "--port", "7860"]