# Use official Python image
FROM python:3.12-slim

# Install system dependencies required for PyTorch and PDFs
RUN apt-get update && apt-get install -y build-essential

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose the port Hugging Face expects (7860)
EXPOSE 7860

# Run FastAPI on port 7860
CMD ["uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "7860"]

