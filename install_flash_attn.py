# -*- coding: utf-8 -*-
import subprocess
import sys
import os

def install_flash_attn():
    print("🚀 [System] Starting Flash Attention Installation...")
    print("   Required for maximizing performance on RTX 5090.")

    try:
        # 1. Check if installed
        import flash_attn
        print(f"✅ [Check] Flash Attention is already installed. (Version: {flash_attn.__version__})")
        return
    except ImportError:
        print("ℹ️ [Check] Flash Attention not found. Installing...")

    # 2. Detect OS and Try Wheel Installation
    import platform
    import torch
    
    system = platform.system()
    torch_ver_full = torch.__version__
    torch_ver = torch_ver_full.split('+')[0] # e.g. 2.5.1
    cuda_ver_tag = "cu12" # Assume cu12 for simplification as user environment implies it
    
    # Python version string for wheel, e.g., cp312
    py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
    
    wheel_url = None
    
    if system == "Linux":
        print("🐧 [System] Detected Linux Environment.")
        print("   Attempting to install official pre-built wheel from Dao-AILab...")
        
        # Construct official URL pattern: 
        # https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
        # Note: We need to be careful with exact version matching.
        # Let's try to map common recent versions.
        
        # Valid releases check map (simplified)
        # v2.7.3 supports torch 2.5.1
        # v2.6.3 supports torch 2.4.0
        
        linux_releases = {
            "2.5.1": "v2.7.3",
            "2.4.0": "v2.6.3",
        }
        
        release_tag = linux_releases.get(torch_ver)
        if release_tag:
            # Construct URL
            # Note: cxx11abiFALSE is common for PyTorch defaults
            filename = f"flash_attn-{release_tag.replace('v','')}+{cuda_ver_tag}torch{torch_ver}cxx11abiFALSE-{py_ver}-{py_ver}-linux_x86_64.whl"
            wheel_url = f"https://github.com/Dao-AILab/flash-attention/releases/download/{release_tag}/{filename}"
        else:
             print(f"⚠️ [Warning] No explicit mapping for Torch {torch_ver} on Linux. Trying generic fallback or source build.")

    elif system == "Windows":
        print("🪟 [System] Detected Windows Environment.")
        print("   Skipping source compilation to avoid hangs. Attempting to install pre-built wheel...")
        
        # Mapping based on bdashore3 releases
        win_wheel_urls = {
            "2.5.1": f"https://github.com/bdashore3/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu124torch2.5.1cxx11abiFALSE-{py_ver}-{py_ver}-win_amd64.whl",
            "2.4.0": f"https://github.com/bdashore3/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu124torch2.4.0cxx11abiFALSE-{py_ver}-{py_ver}-win_amd64.whl",
        }
        wheel_url = win_wheel_urls.get(torch_ver)

    
    # Try installing the wheel if found
    if wheel_url:
        print(f"🔗 [Wheel] Found compatible wheel: {wheel_url}")
        cmd = [sys.executable, "-m", "pip", "install", wheel_url, "--no-build-isolation"]
        print(f"💻 [Exec] Running: {' '.join(cmd)}")
        try:
            subprocess.check_call(cmd)
            print("✅ [Success] Flash Attention Installed via Wheel!")
            return
        except subprocess.CalledProcessError:
            print("❌ [Error] Wheel installation failed. Falling back to source build...")
            print("   (This might take 10-20 minutes on Linux, or hang on Windows)")
    else:
        print(f"⚠️ [Warning] No pre-built wheel found for {system} / Torch {torch_ver}. Falling back to source build.")

    # 3. Fallback / Standard Install (Source Build)
    # Using --no-build-isolation to use current torch
    print("\n⚠️ [Fallback] Wheel installation not possible. Starting source build...")
    print("   This is a CPU/RAM intensive process.")
    
    # [Backend Developer Fix] OOM Prevention
    # Enforce MAX_JOBS=1 to limit Ninja build to single thread.
    # This prevents the build system from spawning 32+ processes on high-core CPUs,
    # which causes instant OOM kill.
    os.environ["MAX_JOBS"] = "1"
    print("🔒 [Safety] Enforcing MAX_JOBS=1 to prevent OOM (Out of Memory) Kills.")
    print("   Note: This will make compilation SLOWER (10-20 min) but STABLE.")
    
    if system == "Linux":
        print(f"ℹ️ [Debug] System Details for troubleshooting:")
        print(f"   - Torch: {torch_ver}")
        print(f"   - CUDA(Torch): {torch.version.cuda}")
        print(f"   - Python: {py_ver}")
        print("   If you want to use a wheel, check if your version matches recent releases.")

    cmd = [
        sys.executable, "-m", "pip", "install", 
        "flash-attn", 
        "--no-build-isolation", 
        "--upgrade"
    ]
    
    print(f"💻 [Exec] Running: {' '.join(cmd)}")

    try:
        subprocess.check_call(cmd)
        print("✅ [Success] Flash Attention Installed!")
        print("👉 Please set the environment variable before running api_server.py:")
        print("   export USE_FLASH_ATTN=1")
    except subprocess.CalledProcessError as e:
        print(f"❌ [Error] Installation failed: {e}")
        print("   Please check if build tools (ninja, packaging) are installed.")
        print("   Try: pip install ninja packaging")

if __name__ == "__main__":
    install_flash_attn()
