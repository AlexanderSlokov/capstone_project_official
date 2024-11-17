# syntax=docker/dockerfile:1 old version please don't use this anymore.
# Build as `docker build . -t localgpt`, requires BuildKit.
# Run as `docker run -it --mount src="$HOME/.cache",target=/root/.cache,type=bind --gpus=all localgpt`, requires Nvidia container toolkit.

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Cài đặt các công cụ cần thiết
RUN apt-get update && apt-get install -y \
    software-properties-common \
    g++-11 make python3.10 python3.10-dev python3-pip\
    apt-utils

# Cập nhật pip
RUN python3.10 -m pip install --upgrade pip

# Cài đặt PyTorch với phiên bản cụ thể
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Copy và cài đặt requirements
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache pip install --timeout 100 -r requirements.txt --index-url https://pypi.org/simple
RUN pip install "autoawq==0.1.5"
RUN pip install "git+https://github.com/PanQiWei/AutoGPTQ.git@v0.7.1"

# Kiểm tra cài đặt CUDA
RUN python3.10 -c "import torch; print(torch.cuda.is_available())"

# Copy các file liên quan đến ingest.py
COPY ingest.py .
COPY SOURCE_DOCUMENTS ./SOURCE_DOCUMENTS

# Chạy ingest.py
RUN python3.10 ingest.py --device_type cuda

# Copy mã nguồn
COPY . .

# Thiết lập môi trường
ARG device_type=cuda
ENV device_type=${device_type}

# Chạy ứng dụng
CMD python3.10 run_localGPT.py --device_type ${device_type}

