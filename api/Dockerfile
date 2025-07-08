FROM python:3.10-slim

WORKDIR /app

# Copy API & requirements
COPY . /app
COPY requirements.txt /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501
