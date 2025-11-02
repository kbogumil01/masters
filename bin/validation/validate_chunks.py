import torch

data = torch.load("/mnt/d/data_mgr/orig_chunks_pt/FourPeople_1280x720_60/chunks_poc002.pt")

print("seq_meta:", data["seq_meta"])
print("chunks shape:", data["chunks"].shape)  # (N, 3, H, W)
print("coords shape:", data["coords"].shape)
print("corner_flags:", data["corner_flags"][:10])