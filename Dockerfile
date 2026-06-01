FROM python:3.13-slim

WORKDIR /app

# OpenCV headless runtime libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x start.sh

ENV PORT=8000
EXPOSE 8000

CMD ["/app/start.sh"]
