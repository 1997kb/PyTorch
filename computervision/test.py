import torch

# Check if PyTorch is installed
print(f"PyTorch version: {torch.__version__}")

# Check for Apple Silicon GPU (MPS) support
if torch.backends.mps.is_available():
    print("Apple Silicon GPU (MPS) is available!")
    device = torch.device("mps")
elif torch.cuda.is_available():
    print("CUDA (NVIDIA GPU) is available!")
    device = torch.device("cuda")
else:
    print("Running on CPU.")
    device = torch.device("cpu")

# Create a random tensor and move it to the device
x = torch.rand(5, 3).to(device)
print(x)