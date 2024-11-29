Với Dockerfile mới, bạn cần điều chỉnh một chút trong cách chạy lệnh `docker build` và `docker run` để phù hợp với các thay đổi và đảm bảo hiệu quả. Dưới đây là các điều chỉnh cần thiết:

---

### Lệnh `docker build`
Lệnh này không cần thay đổi nhiều vì cấu hình BuildKit vẫn tương thích với Dockerfile mới. Tuy nhiên, bạn nên sử dụng tên image rõ ràng hơn để dễ quản lý:

```bash
set DOCKER_BUILDKIT=1
docker build . -t localgpt:latest
```

---

### Lệnh `docker run`
Lệnh này cần được cập nhật để phù hợp với các thay đổi trong Dockerfile và đảm bảo sử dụng đúng `device_type` mặc định là `cuda`.

#### Lệnh cập nhật:

```bash
docker run -it --rm \
  --mount src="$HOME/.cache",target=/root/.cache,type=bind \
  --gpus=all \
  localgpt:latest
```

#### Giải thích các thành phần:
1. `--mount src="$HOME/.cache",target=/root/.cache,type=bind`:
   - Gắn thư mục cache từ máy host vào container để giảm thời gian tải thư viện qua pip.

2. `--gpus=all`:
   - Đảm bảo container có quyền truy cập vào GPU (phụ thuộc vào Nvidia Container Toolkit).

3. `--rm`:
   - Tự động xóa container sau khi chạy xong (nếu bạn không cần lưu lại).

4. `localgpt:latest`:
   - Sử dụng image mới được build (`latest` là tag mặc định).

---

### Tùy chọn thêm cho `device_type`
Trong Dockerfile, bạn đã thiết lập `ARG device_type=cuda` và `ENV device_type=${device_type}`. Nếu bạn muốn kiểm tra chế độ CPU, bạn có thể ghi đè giá trị này khi chạy container:

#### Ví dụ với `device_type=cpu`:

```bash
docker run -it --rm \
  --mount src="$HOME/.cache",target=/root/.cache,type=bind \
  -e device_type=cpu \
  localgpt:latest
```

#### Giải thích:
- `-e device_type=cpu`: Ghi đè biến môi trường `device_type` trong Dockerfile, giúp kiểm tra ứng dụng trên CPU thay vì GPU.

---

### Lưu ý quan trọng
1. Nvidia Container Toolkit:
   - Đảm bảo máy chủ đã cài đặt Nvidia Container Toolkit (`nvidia-docker2`) để sử dụng GPU.
   - Kiểm tra bằng lệnh:
     ```bash
     docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi
     ```

2. Chạy kiểm tra cài đặt CUDA trong container:
   - Sau khi chạy container, kiểm tra lại CUDA:
     ```bash
     python3.10 -c "import torch; print(torch.cuda.is_available())"
     ```

3. Gắn thư mục cache:
   - Nếu không muốn gắn cache (`--mount`), bạn có thể bỏ tùy chọn này, nhưng việc cài đặt lại thư viện sẽ tốn thời gian.

---

### Tóm tắt lệnh cập nhật

#### Build image:
```bash
set DOCKER_BUILDKIT=1
docker build . -t localgpt:latest

```
#### Gọi trực tiếp ingest.py từ container
```powershell
docker run -it --rm --gpus=all --mount src="$env:USERPROFILE\.cache",target=/root/.cache,type=bind localgpt:latest python3.10 ingest.py --device_type cuda
```

#### Run container với GPU:
```powershell
docker run -it --rm --gpus=all --mount src="$env:USERPROFILE\.cache",target=/root/.cache,type=bind localgpt:latest
```

#### Run container với CPU:
```powershell
docker run -it --rm -e device_type=cpu --mount src="$env:USERPROFILE\.cache",target=/root/.cache,type=bind localgpt:latest
```

Những điều chỉnh này đảm bảo container của bạn chạy chính xác với cấu hình mới và tối ưu hóa hiệu năng.
