import torch
import time
from monai.networks.nets import UNet
from config import DATA_DIR, MODELS_DIR, get_device


device = get_device()

model = UNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

model.load_state_dict(torch.load('unet_model_cvc.pth', map_location=device))
model.eval()

dummy = torch.randn(1, 3, 256, 256).to(device)


with torch.no_grad():
    for _ in range(10):
        _ = model(dummy)


durations = []
with torch.no_grad():
    for _ in range(100):
        torch.mps.synchronize()
        start = time.perf_counter()
        _ = model(dummy)
        torch.mps.synchronize()
        end = time.perf_counter()
        durations.append(end - start)

mean_ms = (sum(durations) / len(durations)) * 1000
print(f"Mean: {mean_ms:.2f} ms")
print(f"Min:  {min(durations)*1000:.2f} ms")
print(f"Max:  {max(durations)*1000:.2f} ms")