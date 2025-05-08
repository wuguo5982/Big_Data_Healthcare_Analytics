# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy everything into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Expose the app port
EXPOSE 8000

# Run the app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



## bash    \insurance-ci-cd-ml-project>
# docker run -p 8001:8000 insurance-api

## Open brower - localhost connects to your host machine’s exposed port
# http://localhost:8000/predict                     
