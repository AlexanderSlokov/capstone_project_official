import os
import subprocess
import torch
import platform
import psutil

def get_ram_info():
    ram = psutil.virtual_memory()
    total_ram_gb = ram.total / (1024 ** 3)  # Convert to GB
    return total_ram_gb

def check_cpu_features():
    cpu_info = platform.processor()
    is_intel = "Intel" in cpu_info
    return {
        "CPU": cpu_info,
        "Intel Hyper-Threading": is_intel,  # Simple assumption based on CPU type
        "oneAPI Supported": is_intel,  # Assuming oneAPI is supported if CPU is Intel
    }

def get_vram_info():
    """
    Trích xuất thông tin VRAM và điều chỉnh cấu hình.
    """
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_vram = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # VRAM in GB
            return total_vram
        else:
            return None
    except Exception as e:
        print(f"Error checking VRAM: {e}")
        return None


def check_model_requirements(total_vram, total_ram, model_size="7B", precision="GPTQ-4bit"):
    # Định nghĩa yêu cầu phần cứng dựa trên kích thước mô hình và độ chính xác
    model_requirements = {
        "3B": {
            "float32": {"VRAM": 14, "RAM": 8},
            "float16": {"VRAM": 7, "RAM": 8},
            "GPTQ-4bit": {"VRAM": 3.5, "RAM": 8},
            "GPTQ-8bit": {"VRAM": 7, "RAM": 8},
        },
        "7B": {
            "float32": {"VRAM": 28, "RAM": 12},
            "float16": {"VRAM": 14, "RAM": 12},
            "GPTQ-4bit": {"VRAM": 4, "RAM": 12},
            "GPTQ-8bit": {"VRAM": 7, "RAM": 12},
        },
        "13B": {
            "float32": {"VRAM": 52, "RAM": 16},
            "float16": {"VRAM": 26, "RAM": 16},
            "GPTQ-4bit": {"VRAM": 6.5, "RAM": 16},
            "GPTQ-8bit": {"VRAM": 13, "RAM": 16},
        },
        "30B": {
            "float32": {"VRAM": 130, "RAM": 24},
            "float16": {"VRAM": 65, "RAM": 24},
            "GPTQ-4bit": {"VRAM": 16.25, "RAM": 24},
            "GPTQ-8bit": {"VRAM": 32.5, "RAM": 24},
        },
    }

    # Lấy yêu cầu cho mô hình cụ thể
    requirements = model_requirements.get(model_size, {}).get(precision)
    if not requirements:
        return {
            "error": "Model size or precision not recognized.",
            "VRAM Required": None,
            "RAM Required": None,
            "Sufficient VRAM": False,
            "Sufficient RAM": False,
        }

    # Kiểm tra điều kiện đủ VRAM và RAM
    sufficient_vram = total_vram >= requirements["VRAM"]
    sufficient_ram = total_ram >= requirements["RAM"]

    # Đề xuất cấu hình N_GPU_LAYERS và N_BATCH dựa trên VRAM
    suggested_gpu_layers = max(1, int(total_vram // 0.5))  # Mỗi GPU layer cần khoảng 0.5GB
    suggested_batch_size = max(1, int(total_vram // 0.1))  # Batch size tối ưu theo VRAM (0.1GB mỗi batch)

    return {
        "Sufficient VRAM": sufficient_vram,
        "Sufficient RAM": sufficient_ram,
        "VRAM Required": requirements["VRAM"],
        "RAM Required": requirements["RAM"],
        "Suggested GPU Layers": suggested_gpu_layers,
        "Suggested Batch Size": suggested_batch_size,
    }



def check_cuda_availability():
    """
    Kiểm tra CUDA và phiên bản CUDA.
    """
    try:
        is_cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if is_cuda_available else "Not Available"
        return is_cuda_available, cuda_version
    except Exception as e:
        print(f"Error checking CUDA: {e}")
        return False, "Unknown"

def check_cuda_compute_capability():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_properties(device).major
        return capability
    return None

def warn_insufficient_vram(total_vram, required_vram):
    if total_vram < required_vram:
        print("Warning: Insufficient VRAM. Tasks may shift to CPU or RAM, reducing performance.")


def check_conda_cuda():
    """
    Kiểm tra CUDA toolkit trong môi trường Conda và định dạng kết quả.
    """
    try:
        conda_cuda_path = os.environ.get("CONDA_PREFIX")
        conda_cuda_output = subprocess.check_output("conda list cudatoolkit", shell=True).decode()

        # Xử lý dữ liệu để chỉ lấy dòng chứa tên và phiên bản của cudatoolkit
        filtered_output = [
            line for line in conda_cuda_output.splitlines() if "cudatoolkit" in line
        ]

        conda_version_info = "\n".join(filtered_output) if filtered_output else "Không tìm thấy CUDA toolkit"
        return conda_cuda_path, conda_version_info
    except Exception as e:
        print(f"Error checking Conda CUDA: {e}")
        return None, "Không xác định"


def check_host_nvcc():
    """
    Kiểm tra CUDA toolkit trên hệ điều hành host.
    """
    try:
        nvcc_version = subprocess.check_output("nvcc --version", shell=True).decode()
        return nvcc_version
    except Exception as e:
        print(f"Error checking NVCC version: {e}")
        return None


def export_dependencies():
    """
    Xuất danh sách các dependencies từ pip và conda.
    """
    try:
        pip_requirements = subprocess.check_output("pip freeze", shell=True).decode()
        with open("pip_requirements.txt", "w") as f:
            f.write(pip_requirements)

        conda_requirements = subprocess.check_output("conda list", shell=True).decode()
        with open("conda_requirements.txt", "w") as f:
            f.write(conda_requirements)
        return True
    except Exception as e:
        print(f"Error exporting dependencies: {e}")
        return False


def system_check():
    print("Performing system check...")

    # Kết quả cuối cùng lưu trữ dưới dạng dictionary
    results = {}

    # Kiểm tra CUDA
    cuda_available, cuda_version = check_cuda_availability()
    results["CUDA Available"] = cuda_available
    results["CUDA Version"] = cuda_version
    print(f"CUDA Available: {cuda_available}, CUDA Version: {cuda_version}")

    # Kiểm tra VRAM
    total_vram = get_vram_info()
    results["VRAM"] = total_vram
    print(f"Total VRAM: {total_vram:.2f} GB" if total_vram else "VRAM Info Unavailable")

    # Kiểm tra RAM
    total_ram = get_ram_info()
    results["RAM"] = total_ram
    print(f"Total RAM: {total_ram:.2f} GB")

    # Kiểm tra CPU
    cpu_features = check_cpu_features()
    results["CPU"] = cpu_features
    print(f"CPU Info: {cpu_features}")

    # Kiểm tra yêu cầu mô hình (nếu có VRAM và RAM)
    if total_vram and total_ram:
        requirements = check_model_requirements(
            total_vram=total_vram,
            total_ram=total_ram,
            model_size="7B",  # Có thể tùy chỉnh theo mô hình
            precision="GPTQ-4bit"  # Có thể tùy chỉnh theo độ chính xác
        )
        results["Model Requirements"] = requirements
        print("Model Requirements Check:", requirements)
        warn_insufficient_vram(total_vram, requirements["VRAM Required"])
    else:
        results["Model Requirements"] = {
            "error": "Không đủ thông tin để kiểm tra yêu cầu mô hình",
            "VRAM Required": None,
            "RAM Required": None,
            "Sufficient VRAM": False,
            "Sufficient RAM": False,
        }

    # Kiểm tra kiến trúc CUDA Compute Capability
    cuda_capability = check_cuda_compute_capability()
    results["CUDA Capability"] = cuda_capability
    print(f"CUDA Compute Capability: {cuda_capability}" if cuda_capability else "CUDA Compute Capability Unavailable")

    # Kiểm tra Conda CUDA Toolkit
    conda_path, conda_version = check_conda_cuda()
    results["Conda Path"] = conda_path
    results["Conda Version"] = conda_version
    print(f"Conda CUDA Path: {conda_path}, Conda CUDA Version: {conda_version}")

    # Kiểm tra NVCC trên host
    host_nvcc_version = check_host_nvcc()
    results["NVCC Version"] = host_nvcc_version
    print(f"Host NVCC Version: {host_nvcc_version}" if host_nvcc_version else "NVCC not found on host.")

    print("System check complete.")
    return results

