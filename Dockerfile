# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables to ensure python output is logged
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install uv for fast package resolution
RUN pip install uv

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements first to cache the layer
COPY requirements.txt .

# Install dependencies using uv
RUN uv pip install --system -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
