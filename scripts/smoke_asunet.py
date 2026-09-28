import os, sys, cv2, numpy as np, torch
from pathlib import Path

REPO = Path(r"D:\landslide_unet_project\repos\AS-UNet")
BASE = REPO / "inputs" / "data_sum_moxizhen+bijie" / "train"
sys.path.insert(0, str(REPO))
import archs
import losses

def main():
    images = sorted((BASE / "images").glob("*.png"))[:8]
    xs, ys = [], []
    for image_path in images:
        mask_path = BASE / "masks" / image_path.name
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Failed to read {image_path} or {mask_path}")
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR).astype("float32") / 255.0
        mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST).astype("float32") / 255.0
        xs.append(image.transpose(2, 0, 1))
        ys.append(mask[None])
    x = torch.from_numpy(np.stack(xs)).cuda()
    y = torch.from_numpy(np.stack(ys)).cuda()
    model = archs.AS_UNet(num_classes=1, input_channels=3).cuda().train()
    criterion = losses.PolyGHMDiceLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    output = model(x)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"input={tuple(x.shape)} target={tuple(y.shape)} output={tuple(output.shape)}")
    print(f"loss={loss.item():.6f} gpu={torch.cuda.get_device_name(0)}")
    print(f"allocated_MB={torch.cuda.memory_allocated() / 1024**2:.1f}")
    print("SMOKE_OK")

if __name__ == "__main__":
    main()
