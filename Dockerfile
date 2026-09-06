# Use official Python image
FROM python:3.12-slim

# Install system dependencies required for PyTorch and PDFs
RUN apt-get update && apt-get install -y build-essential

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download massive AI models during the build phase!
# This prevents the 60-second Hugging Face timeout from killing your app when a user asks the first question.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; print('Downloading BGE...'); SentenceTransformer('BAAI/bge-large-en-v1.5'); print('Downloading BGE Reranker...'); CrossEncoder('BAAI/bge-reranker-base'); print('Models successfully baked into Docker image!')"

# Copy your application code
COPY . .

# Expose the port Hugging Face expects (7860)
EXPOSE 7860

# Run FastAPI on port 7860
CMD ["uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "7860"]
