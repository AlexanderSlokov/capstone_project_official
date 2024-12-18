### **Hướng dẫn cập nhật chạy Docker cho localGPT**
Dưới đây là hướng dẫn đầy đủ và chi tiết để chạy Docker phù hợp với cấu trúc file mới của bạn.

---

### **1. Build Docker Image**
Sử dụng Docker BuildKit để build nhanh hơn và tối ưu hóa:

#### **Lệnh build:**
```bash
set DOCKER_BUILDKIT=1
docker build . -t localgpt:latest
```
- **`localgpt:latest`**: Tag image mới build để dễ dàng tham chiếu.

---

### **2. Chạy Container với GPU**
Sử dụng GPU (NVIDIA) để tăng tốc xử lý.

#### **Chạy Streamlit ứng dụng chính (`localGPT_UI.py`):**
```bash
docker run -it --rm \
  --gpus=all \
  --mount src="$HOME/.cache",target=/root/.cache,type=bind \
  -p 8501:8501 \
  localgpt:latest
```

- **Giải thích**:
   - `--gpus=all`: Cho phép container truy cập toàn bộ GPU.
   - `--mount src="$HOME/.cache",target=/root/.cache,type=bind`: Gắn thư mục cache để tăng tốc cài đặt thư viện.
   - `-p 8501:8501`: Map cổng 8501 của container ra máy host để truy cập ứng dụng Streamlit.
   - **Truy cập URL**: `http://localhost:8501` trên trình duyệt để chạy localGPT UI.

---

### **3. Chạy trực tiếp `ingest.py` từ Container**
Nếu bạn muốn gọi file `ingest.py` để nạp dữ liệu vào ChromaDB:

#### **Lệnh chạy ingest.py:**
```bash
docker run -it --rm \
  --gpus=all \
  --mount src="$HOME/.cache",target=/root/.cache,type=bind \
  localgpt:latest python3.10 localGPT_app/ingest.py --device_type cuda
```

- **`localGPT_app/ingest.py`**: Đường dẫn chính xác của file `ingest.py` trong container.
- **`--device_type cuda`**: Sử dụng GPU CUDA. Thay bằng `cpu` nếu không cần GPU.

---

### **4. Chạy Container với CPU (Tùy chọn)**
Nếu bạn không có GPU hoặc muốn kiểm tra ứng dụng với CPU:

#### **Lệnh chạy Streamlit trên CPU:**
```bash
docker run -it --rm \
  -e device_type=cpu \
  --mount src="$HOME/.cache",target=/root/.cache,type=bind \
  -p 8501:8501 \
  localgpt:latest
```
- **`-e device_type=cpu`**: Ghi đè biến môi trường `device_type` trong Dockerfile để chạy trên CPU.

#### **Lệnh chạy ingest.py trên CPU:**
```bash
docker run -it --rm \
  -e device_type=cpu \
  --mount src="$HOME/.cache",target=/root/.cache,type=bind \
  localgpt:latest python3.10 localGPT_app/ingest.py --device_type cpu
```

---

### **5. Kiểm tra CUDA và PyTorch trong Container**
Đảm bảo container đã nhận GPU và PyTorch hoạt động chính xác:

#### **Lệnh kiểm tra CUDA:**
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi
```

#### **Kiểm tra CUDA trong PyTorch:**
```bash
docker run -it --rm localgpt:latest python3.10 -c "import torch; print(torch.cuda.is_available())"
```

---

### **6. Tóm tắt các lệnh Docker**

| **Tác vụ**                        | **Lệnh Docker**                                                                                         |
|----------------------------------|---------------------------------------------------------------------------------------------------------|
| Build Docker Image               | `$env:DOCKER_BUILDKIT=1; docker build . -t localgpt:latest`                                             |
| Chạy Streamlit (GPU)             | `docker run -it --rm --gpus=all -p 8501:8501 -v ${PWD}:/app -w /app localgpt:latest`                                                |
| Chạy Streamlit (CPU)             | `docker run -it --rm -e device_type=cpu -p 8501:8501 localgpt:latest`                                   |
| Chạy ingest.py (GPU)             | `docker run -it --rm --gpus=all localgpt:latest python3.10 localGPT_app/ingest.py --device_type cuda`   |
| Chạy ingest.py (CPU)             | `docker run -it --rm -e device_type=cpu localgpt:latest python3.10 localGPT_app/ingest.py --device_type cpu` |
| Kiểm tra GPU và PyTorch          | `docker run -it --rm localgpt:latest python3.10 -c "import torch; print(torch.cuda.is_available())"`    |

---

### **7. Kết luận**
Hướng dẫn này đã được cập nhật với:
1. **Dockerfile mới** chạy **Streamlit** và `ingest.py`.
2. Các lệnh phù hợp để chạy với **GPU** hoặc **CPU**.
3. Gắn cache thư viện và tối ưu quá trình build/run.

Nếu bạn gặp bất kỳ vấn đề gì trong quá trình chạy Docker, đừng ngần ngại báo mình! 🚀
