FROM python:3.10 as base

WORKDIR /app

# Install ffmpeg, libsm6, libxext6
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir .

CMD ["catflow-service-detector"]
