FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg git curl && rm -rf /var/lib/apt/lists/*

# Set workdir and install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app source
COPY . .

# Expose port (optional for Hugging Face)
EXPOSE 7860

# Entry point script
CMD ["bash", "start_server.sh"]
