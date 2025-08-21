FROM python:3.13.5

WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements if you have one
COPY requirements.api.txt .

RUN pip install -r requirements.api.txt

# Copy the source code
COPY ../src/ /app/src/

# Start FastAPI app using uvicorn
CMD ["uvicorn", "src.slm_api:app", "--host", "0.0.0.0", "--port", "7777"]