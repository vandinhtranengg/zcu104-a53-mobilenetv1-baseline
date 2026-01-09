
# synth_digits_train.py
import os, math, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------- Quantization params (match your firmware) ----------------
IN_SCALE = 0.02
IN_ZP    = 128

# ---------------- Model (DW3x3 + ReLU6 -> PW1x1 -> GAP -> logits[10]) ------
class TinyDigitDSCNN(nn.Module):
    def __init__(self, cout=10):
        super().__init__()
        # Depthwise 3x3 (groups=3), Cin=3 -> Cout_dw=3
        self.dw = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, bias=False)
        # Pointwise 1x1: 3 -> 10 classes
        self.pw = nn.Conv2d(3, cout, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.dw(x)                 # DW
        x = torch.clamp(x, 0.0, 6.0)   # ReLU6 clamp (matches your DW kernel)
        x = self.pw(x)                 # PW
        x = x.mean(dim=(2, 3))         # Global Average Pool (H,W)
        return x                        # logits [N,10]

# ---------------- Synthetic digits dataset (32x32 RGB) ---------------------
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
def load_font(size):
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    try:
        return ImageFont.load_default()
    except Exception:
        return None

def render_digit_32x32(d: int) -> np.ndarray:
    W, H = 32, 32
    size = random.randint(22, 30)
    font = load_font(size)
    img = Image.new('L', (W, H), color=0)
    draw = ImageDraw.Draw(img)
    text = str(d)
    stroke = random.randint(0, 2)
    try:
        bbox = draw.textbbox((0,0), text, font=font, stroke_width=stroke)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    except Exception:
        tw, th = 16, 16
    x = (W - tw)//2 + random.randint(-4, 4)
    y = (H - th)//2 + random.randint(-4, 4)
    draw.text((x, y), text, fill=255, font=font, stroke_width=stroke, stroke_fill=255)
    img = img.rotate(random.uniform(-15, 15), resample=Image.BILINEAR, expand=False, fillcolor=0)

    arr = np.array(img, dtype=np.uint8)
    noise = np.random.normal(0, 8.0, size=arr.shape).astype(np.int32)
    arr = np.clip(arr.astype(np.int32) + noise, 0, 255).astype(np.uint8)

    rgb = np.stack([arr, arr, arr], axis=0)  # [3, H, W]
    return rgb

class SyntheticDigits(Dataset):
    def __init__(self, n_per_class: int, seed: int = 0):
        self.W, self.H = 32, 32
        self.npc = n_per_class
        self.N = n_per_class * 10
        self.rng = random.Random(seed)

    def __len__(self): return self.N

    def __getitem__(self, idx):
        # class label evenly distributed
        d = idx % 10
        # render one RGB digit image
        rgb = render_digit_32x32(d)            # [3,32,32], uint8
        rgb_u8 = torch.from_numpy(rgb.copy())  # torch.uint8
        rgb_f  = rgb_u8.float()                # [0..255]
        # Simulate board's dequant: real = 0.02 * (uint8 - 128)
        x_real = IN_SCALE * (rgb_f - IN_ZP)    # float32
        y = torch.tensor(d, dtype=torch.long)
        return x_real, y

# ---------------- Training routine ----------------------------------------
def train(epochs=5, batch=128, n_per_class_train=800, n_per_class_val=200, lr=1e-3, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    train_set = SyntheticDigits(n_per_class=n_per_class_train, seed=123)
    val_set   = SyntheticDigits(n_per_class=n_per_class_val,   seed=456)

    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)

    model = TinyDigitDSCNN(cout=10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        tr_loss = running / len(train_loader.dataset)

        # quick validation accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += x.size(0)
        val_acc = 100.0 * correct / total
        print(f"Epoch {ep}: train loss={tr_loss:.4f}  val acc={val_acc:.2f}%")

    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/digits_tiny_dsconv.pt")
    print("Saved: artifacts/digits_tiny_dsconv.pt")
    return model

if __name__ == "__main__":
    # Defaults are good; tune epochs/batch if you want more accuracy or speed.
    train(epochs=5, batch=128, n_per_class_train=800, n_per_class_val=200, lr=1e-3)

