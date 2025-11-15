# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port vLLM will run on
EXPOSE 8000

# Start the vLLM server
# NOTE: No commas between arguments, just spaces
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", "--model", "microsoft/Phi-3-mini-4k-instruct", "--host", "0.0.0.0", "--port", "8000"]
