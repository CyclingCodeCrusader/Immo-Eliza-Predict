# Starts from the python 3.10 official docker image
FROM python:3.10

# Create a folder "app" at the root of the image
RUN mkdir /app

# Define /app as the working directory
WORKDIR /app

# Copy all the files in the current directory in /app
COPY . /app

# Update pip
RUN pip install --upgrade pip

# Install dependencies from "requirements.txt"
RUN pip install -r requirements.txt

# Run the app
# Set host to 0.0.0.0 to make it run on the container's network
#CMD uvicorn app:app --host 0.0.0.0

# Expose ports for both applications
EXPOSE 10000  # Streamlit default port in Render
EXPOSE 8000   # FastAPI default port

# Use a process manager to run both apps simultaneously
#CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 & streamlit run predictor_streamlit.py --server.port=10000 --server.address=0.0.0.0"]

# Run both FastAPI and Streamlit apps
CMD ["sh", "-c", "uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app.py --server.port=10000 --server.address=0.0.0.0"]