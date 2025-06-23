# Use the full Python image for maximum compatibility with TensorFlow
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy all application files
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        fastapi==0.110.0 \
        uvicorn==0.29.0 \
        tensorflow==2.15.0 \
        numpy==1.24.4 \
        pandas==1.5.3 \
        pydantic==1.10.13

# Expose the port used by FastAPI
EXPOSE 8000

# Run the FastAPI application with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
