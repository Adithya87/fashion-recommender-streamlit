# ---- Base image ----
FROM python:3.11-slim

# ---- System libs for OpenCV, Mediapipe, and LightGBM ----
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# ---- Workdir ----
WORKDIR /app
COPY . /app

# ---- Python deps ----
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ---- Expose Streamlit port ----
EXPOSE 8501

# ---- Start app ----
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
