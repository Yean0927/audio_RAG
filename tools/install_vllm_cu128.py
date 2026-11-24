"""Install the Linux x86_64 vLLM environment with resumable wheel downloads."""

import fcntl
import hashlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit


from voice_service import ROOT
INSTALL = (
    "uv", "--no-config", "pip", "install",
    "--only-binary", ":all:", "--index", "https://mirrors.nju.edu.cn/pytorch/whl/cu128/",
    "--default-index", "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
    "--index-strategy", "unsafe-best-match",
)
REQUIREMENTS = (
    "vllm==0.11.2", "torch==2.9.0+cu128", "torchaudio==2.9.0+cu128",
    "torchvision==0.24.0+cu128", "bitsandbytes==0.48.2",
)

WHEELS = """
https://mirrors.nju.edu.cn/pytorch/whl/cu128/torch-2.9.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl 87c62d3b95f1a2270bd116dbd47dc515c0b2035076fbb4a03b4365ea289e89c4
https://mirrors.nju.edu.cn/pytorch/whl/triton-3.5.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl 76f8651a5e38c2a7da6fa2b2e41cbc00a5a32cb52bf3f520113fe90b723a310d
https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/75/5d/d6af7818e41957a5d35f1b0ecd0186ac80e322f228dc390dcbc4aafce58d/vllm-0.11.2-cp38-abi3-manylinux1_x86_64.whl ea473bd4fde06940fe3f681a00476060652f62b3279ef11aaffac5768856cfe8
https://mirrors.nju.edu.cn/pytorch/nvidia-cudnn-cu12/nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl 949452be657fa16687d0930933f032835951ef0892b37d2d53824d1a84dc97a8
https://mirrors.nju.edu.cn/pytorch/nvidia-cublas-cu12/nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl 8ac4e771d5a348c551b2a426eda6193c19aa630236b418086020df5ba9667142
https://mirrors.nju.edu.cn/pytorch/nvidia-nccl-cu12/nvidia_nccl_cu12-2.27.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl ad730cf15cb5d25fe849c6e6ca9eb5b76db16a80f13f425ac68d8e2e55624457
https://mirrors.nju.edu.cn/pytorch/nvidia-cusparse-cu12/nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 1ec05d76bbbd8b61b06a80e1eaf8cf4959c3d4ce8e711b65ebd0443bb0ebb13b
https://mirrors.nju.edu.cn/pytorch/nvidia-cusolver-cu12/nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl 4376c11ad263152bd50ea295c05370360776f8c3427b30991df774f9fb26c450
https://mirrors.nju.edu.cn/pytorch/nvidia-cufft-cu12/nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 4d2dd21ec0b88cf61b62e6b43564355e5222e4a3fb394cac0db101f2dd0d4f74
https://mirrors.nju.edu.cn/pytorch/nvidia-cusparselt-cu12/nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl f1bb701d6b930d5a7cea44c19ceb973311500847f81b634d802b7b539dc55623
https://mirrors.nju.edu.cn/pytorch/nvidia-cuda-nvrtc-cu12/nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl a7756528852ef889772a84c6cd89d41dfa74667e24cca16bb31f8f061e3e9994
https://mirrors.nju.edu.cn/pytorch/nvidia-nvshmem-cu12/nvidia_nvshmem_cu12-3.3.20-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl d00f26d3f9b2e3c3065be895e3059d6479ea5c638a3f38c9fec49b1b9dd7c1e5
https://mirrors.nju.edu.cn/pytorch/nvidia-curand-cu12/nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl b32331d4f4df5d6eefa0554c565b626c7216f87a06a4f56fab27c3b68a830ec9
https://mirrors.nju.edu.cn/pytorch/nvidia-nvjitlink-cu12/nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl 81ff63371a7ebd6e6451970684f916be2eab07321b73c9d244dc2b4da7f73b88
https://mirrors.nju.edu.cn/pytorch/nvidia-cufile-cu12/nvidia_cufile_cu12-1.13.1.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 1d069003be650e131b21c932ec3d8969c1715379251f8d23a1860554b1cb24fc
https://mirrors.nju.edu.cn/pytorch/nvidia-cuda-cupti-cu12/nvidia_cuda_cupti_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl ea0cb07ebda26bb9b29ba82cda34849e73c166c18162d3913575b0c9db9a6182
"""


def matches_digest(path, expected):
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == expected


def download_wheel(cache, url, expected):
    filename = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
    wheel = cache / filename
    if matches_digest(wheel, expected):
        return wheel

    partial = cache / (filename + ".part")
    if matches_digest(partial, expected):
        partial.replace(wheel)
        return wheel

    print(f"下载 {filename}", flush=True)
    for attempt in range(4):
        result = subprocess.run((
            "curl", "--fail", "--location", "--continue-at", "-",
            "--connect-timeout", "15", "--speed-time", "60", "--speed-limit", "1024",
            "--output", str(partial), url,
        ))
        if result.returncode == 0:
            if matches_digest(partial, expected):
                partial.replace(wheel)
                return wheel
            partial.unlink(missing_ok=True)
            raise SystemExit(f"SHA256 校验失败，重新运行以下载 {filename}。")
        if attempt < 3:
            time.sleep(3)
    raise SystemExit(f"下载未完成，重新运行可从 {partial} 续传。")


def main():
    if sys.platform != "linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 12):
        raise SystemExit("请使用 Linux x86_64 服务器上的 Python 3.12。")

    data_dir = os.environ.get("VLLM_DATA_DIR")
    install_root = Path(data_dir).expanduser().resolve() if data_dir else ROOT
    venv = install_root / ".venv-vllm-cu128"
    python = venv / "bin/python"
    env = {
        **os.environ, "UV_HTTP_TIMEOUT": "300", "UV_HTTP_RETRIES": "5", "UV_CONCURRENT_DOWNLOADS": "2",
    }
    if data_dir:
        env["UV_CACHE_DIR"] = str(install_root / "uv-cache")
    install = (*INSTALL, "--python", str(python))

    cache_root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    cache = cache_root / "glm-voice-rag" / "vllm-cu128-wheels"
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not python.is_file():
            venv.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(("uv", "venv", "--python", sys.executable, str(venv)), check=True, env=env)
        print(f"安装环境 {venv}", flush=True)
        if data_dir:
            print(f"uv 解压缓存 {env['UV_CACHE_DIR']}", flush=True)
        cached = subprocess.run((*install, "--offline", *REQUIREMENTS), capture_output=True, env=env)
        if cached.returncode:
            print(f"断点续传目录 {cache}", flush=True)
            wheels = [download_wheel(cache, *line.split()) for line in WHEELS.strip().splitlines()]
            subprocess.run((*install, *REQUIREMENTS, *map(str, wheels)), check=True, env=env)
    print(f"vLLM 安装完成，入口为 {venv / 'bin/vllm'}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from None
