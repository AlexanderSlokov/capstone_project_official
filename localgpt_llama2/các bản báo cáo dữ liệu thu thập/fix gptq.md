Để biên dịch các CUDA extensions nhằm tăng tốc độ suy luận khi cài đặt `auto-gptq`, bạn cần thiết lập biến môi trường `BUILD_CUDA_EXT=1`. Trên Windows, có hai cách để thực hiện điều này:

**1. Thiết lập biến môi trường tạm thời trong Command Prompt:**

Nếu bạn chỉ muốn thiết lập biến môi trường cho phiên làm việc hiện tại, bạn có thể thực hiện như sau:

- Mở Command Prompt.
- Kích hoạt môi trường Conda của bạn bằng lệnh:
  ```bash
  conda activate tên_môi_trường_của_bạn
  ```
- Thiết lập biến môi trường `BUILD_CUDA_EXT`:
  ```bash
  set BUILD_CUDA_EXT=1
  ```
- Tiến hành cài đặt `auto-gptq`:
  ```bash
  pip install "git+https://github.com/PanQiWei/AutoGPTQ.git@v0.7.1"
  ```

Lưu ý rằng biến môi trường này sẽ chỉ tồn tại trong phiên làm việc hiện tại và sẽ bị xóa khi bạn đóng Command Prompt.

**2. Thiết lập biến môi trường vĩnh viễn trong System Variables:**

Nếu bạn muốn biến môi trường `BUILD_CUDA_EXT` luôn được thiết lập cho mọi phiên làm việc, bạn có thể thêm nó vào System Variables:

- Nhấn `Win + R`, nhập `sysdm.cpl` và nhấn Enter để mở cửa sổ System Properties.
- Chọn tab "Advanced" và nhấn vào nút "Environment Variables".
- Trong phần "System variables", nhấn "New..." để tạo biến môi trường mới.
- Trong ô "Variable name", nhập `BUILD_CUDA_EXT`.
- Trong ô "Variable value", nhập `1`.
- Nhấn "OK" để lưu và đóng các cửa sổ.

Sau khi thiết lập, biến môi trường `BUILD_CUDA_EXT` sẽ được áp dụng cho mọi phiên làm việc và bạn có thể tiến hành cài đặt `auto-gptq` như bình thường.

**Lưu ý:** Việc thiết lập biến môi trường vĩnh viễn có thể ảnh hưởng đến các ứng dụng khác. Nếu bạn chỉ cần thiết lập tạm thời cho một phiên làm việc cụ thể, phương pháp đầu tiên sẽ phù hợp hơn.
