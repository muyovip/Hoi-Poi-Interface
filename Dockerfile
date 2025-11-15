# WHAT: This is like a recipe for building your application environment
# WHY: Cloud Run needs to know exactly how to set up your code to run

# Start with a Python base image (like setting up a kitchen)
FROM python:3.9-slim

# Set working directory (like choosing which counter to work on)
WORKDIR /app

# Copy requirements first (caching trick - like gathering ingredients)
COPY requirements.txt .

# Install dependencies (like setting up your kitchen tools)
RUN pip install --no-cache-dir -r requirements.txt

# Copy your actual application code (like bringing in your recipes)
COPY . .

# Expose the port vLLM will run on (like opening a service window)
EXPOSE 8000

# Command to start the server (like turning on the oven)
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", 
     "--model", "microsoft/Phi-3-mini-4k-instruct", 
     "--host", "0.0.0.0", 
     "--port", "8000"]
