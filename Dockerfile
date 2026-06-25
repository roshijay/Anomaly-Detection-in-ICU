# Start from official Python 3.11 slim image
# 'slim' means minimal Ubuntu - no unnecessary packages, smaller image size
FROM python:3.11-slim

# Set working directory inside the container 
# All subsequent commands run from here 
WORKDIR /app

# Copy requirements first - before copying code
# This is deliberate: Docker caches layers in order.
# If only your code changes (not requirements), Docker reuses
# the cached pip install layer and skips reinstalling everything.
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model
COPY src/ ./src/
COPY models/ ./models/

# Tell Docker this container listens on port 8000
# This is documentation only - doesn't actually open the port
# Port mapping happens at 'docker run' time
EXPOSE 8000

# Command to run when container starts
# Note: we set the working directory to src/ first so uvicorn
# finds api.py correctly, same as when we ran it manually
CMD ["sh", "-c", "cd /app/src && uvicorn api:app --host 0.0.0.0 --port 8000"]