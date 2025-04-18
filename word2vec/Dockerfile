
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files and code
COPY *.pth /app/
COPY app.py /app/

# Create a directory for static files
RUN mkdir -p /app/static

# Copy frontend files
COPY index.html /app/static/

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]