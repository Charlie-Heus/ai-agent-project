FROM python:3.11-slim


WORKDIR /app


# Install system dependencies
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*


# Clone the repository
RUN git clone https://github.com/Charlie-Heus/veris-ai-agent.git .


# Install Python requirements
RUN pip install -r requirements.txt


# Create data directory and download dataset
RUN mkdir -p data && python main.py


# Run the CLI by default
CMD ["python", "main.py"]
