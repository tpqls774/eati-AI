import torch
import torch.version

# print("PyTorch version:", torch.__version__)
# print("Is CUDA available:", torch.cuda.is_available())
# print("CUDA version in PyTorch:", torch.version.cuda)

print("CUDA:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("PyTorch version:", torch.__version__)
print("Available GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current GPU:", torch.cuda.get_device_name(0))

# state_dict = torch.load("final_model", map_location="cpu")
# print(state_dict.keys())