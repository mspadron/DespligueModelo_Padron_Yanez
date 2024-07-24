FROM python:3.11-slim

# Expose the port your app will run on
EXPOSE 8080

RUN pip install --upgrade pip

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy the rest of the application
COPY . /app
WORKDIR /app

# Run the application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
