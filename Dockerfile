# Use a  Python image as the base
FROM python:3.11

# Set the working directory to /app
WORKDIR /app

# Copy your application code to the container
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Expose some port
EXPOSE $PORT

# Run your Flask app
CMD gunicorn --bind 0.0.0.0:$PORT app:app
