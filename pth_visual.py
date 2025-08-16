import torch

checkpoint = torch.load("/remote-home/ums_wangdantong/checkpoints/latest-shard_pano_mse_amber-lake-20.pth")
print("Keys in checkpoint:", checkpoint.keys())
print("Epoch:", checkpoint['epoch'])
print("Train Loss:", checkpoint['train_loss'])