import torch
import sys
import platform

def check_gpu():
    print("🖥️ System Info:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   OS: {platform.system()} {platform.release()}")
    print("-" * 30)
    
    print("🔥 PyTorch Info:")
    print(f"   Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA Version (Compile): {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"   GPU Count: {device_count}")
        
        for i in range(device_count):
            print(f"\n   [GPU {i}]")
            print(f"   Name: {torch.cuda.get_device_name(i)}")
            
            try:
                capability = torch.cuda.get_device_capability(i)
                print(f"   Compute Capability: {capability[0]}.{capability[1]}")
                
                # Simple tensor check
                print("   ⚡ Testing Tensor Cores...", end=" ")
                x = torch.tensor([1.0, 2.0], device=f"cuda:{i}")
                y = x * 2
                print(f"OK! Result: {y.tolist()}")
                
                # Check bfloat16 support (Qwen3 uses it)
                print("   ⚡ Testing bfloat16...", end=" ")
                if torch.cuda.is_bf16_supported():
                    z = torch.tensor([1.0], device=f"cuda:{i}", dtype=torch.bfloat16)
                    print("Supported ✅")
                else:
                    print("Not Supported ❌ (This might be the issue if model uses bf16)")
                    
            except Exception as e:
                print(f"\n   ❌ Error accessing GPU {i}: {e}")
    else:
        print("   ❌ CUDA is NOT available. Review your installation.")

if __name__ == "__main__":
    check_gpu()
