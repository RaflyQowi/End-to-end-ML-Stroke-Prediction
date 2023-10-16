# Use a slim Python image as the base
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy only the necessary files to the container
COPY app.py requirements.txt ./
COPY src ./src
COPY artifacts ./artifacts
COPY templates ./templates

# Install packages and clean up
RUN pip install -r requirements.txt && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Run your Flask app with Gunicorn
CMD ["python", "app.py"]