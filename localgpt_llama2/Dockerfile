# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Cài đặt các công cụ cần thiết
RUN apt-get update && apt-get install -y \
    software-properties-common \
    g++-9 gcc-9 make python3.10 python3.10-dev python3-pip git ninja-build \
    libcudnn8 libcudnn8-dev apt-utils build-essential && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 60 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Cập nhật pip
RUN python3.10 -m pip install --upgrade pip

# Thêm biến môi trường cho CUDA và kích hoạt CUDA extensions
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV CUDA_EXT=1

# Cài đặt và xử lý lỗi liên quan đến gói blinker
RUN apt-get remove --purge python3-blinker -y || true
RUN pip install --ignore-installed blinker

# Cài đặt PyTorch với CUDA 11.8
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Biên dịch và cài đặt auto-gptq với cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "git+https://github.com/PanQiWei/AutoGPTQ.git@v0.7.1"

# Copy và cài đặt requirements với cache
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt --timeout 100

# Copy mã nguồn
COPY . .

# Run the app
ENTRYPOINT ["python3.10"]
CMD ["run_localGPT.py", "--device_type",  "cuda"]



