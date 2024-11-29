import os
import subprocess
import torch


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


def check_conda_cuda():
    """
    Kiểm tra CUDA toolkit trong môi trường conda.
    """
    try:
        conda_cuda_path = os.environ.get("CONDA_PREFIX")
        conda_cuda_version = subprocess.check_output("conda list cudatoolkit", shell=True).decode()
        return conda_cuda_path, conda_cuda_version
    except Exception as e:
        print(f"Error checking Conda CUDA: {e}")
        return None, None


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
    """
    Tổng hợp các bước kiểm tra môi trường.
    """
    print("Performing system check...")
    cuda_available, cuda_version = check_cuda_availability()
    print(f"CUDA Available: {cuda_available}, CUDA Version: {cuda_version}")

    total_vram = get_vram_info()
    print(f"Total VRAM: {total_vram} GB" if total_vram else "VRAM Info Unavailable")

    conda_path, conda_version = check_conda_cuda()
    if conda_path:
        print(f"Conda CUDA Path: {conda_path}")
        print(f"Conda CUDA Version:\n{conda_version}")
    else:
        print("Conda CUDA not found.")

    host_nvcc_version = check_host_nvcc()
    if host_nvcc_version:
        print(f"Host NVCC Version:\n{host_nvcc_version}")
    else:
        print("NVCC not found on host.")

    dependencies_exported = export_dependencies()
    print("Dependencies exported successfully." if dependencies_exported else "Failed to export dependencies.")

    print("System check complete.")
    return cuda_available, total_vram, cuda_version
