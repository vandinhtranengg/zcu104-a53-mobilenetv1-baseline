
# export_bins.py
import os
import numpy as np
import torch
from synth_digits_train import TinyDigitDSCNN

W_ZP = 128  # weight zero-point (single for both layers)

def compute_single_w_scale(dw_w, pw_w):
    # Single scale across both layers to match firmware API
    max_abs = max(dw_w.abs().max().item(), pw_w.abs().max().item())
    return max_abs / 127.0 if max_abs > 1e-12 else 0.01

def quant_u8(w, w_scale, w_zp=W_ZP):
    q = np.round(w / w_scale).astype(np.int32) + w_zp
    q = np.clip(q, 0, 255).astype(np.uint8)
    return q

def main():
    state = torch.load("artifacts/digits_tiny_dsconv.pt", map_location="cpu")
    model = TinyDigitDSCNN(cout=10)
    model.load_state_dict(state)
    model.eval()

    # Extract float weights
    dw = model.dw.weight.detach().cpu()          # [3,1,3,3]
    pw = model.pw.weight.detach().cpu()          # [10,3,1,1]
    dw_w = dw.squeeze(1)                         # [3,3,3]
    pw_w = pw.squeeze(2).squeeze(2)              # [10,3]

    # Single scale
    w_scale = compute_single_w_scale(dw_w, pw_w)
    print(f"Chosen w_scale = {w_scale:.9f} (w_zp=128)")

    # ---- Quantize ----
    dw_q = quant_u8(dw_w.numpy(), w_scale, W_ZP)   # uint8 [3,3,3]
    pw_q = quant_u8(pw_w.numpy(), w_scale, W_ZP)   # uint8 [10,3]

    # ---- Flatten to firmware layouts ----
    # DW: c*9 + (ky+1)*3 + (kx+1)
    dw_flat = []
    for c in range(3):
        for ky in (-1, 0, 1):
            for kx in (-1, 0, 1):
                dw_flat.append(int(dw_q[c, ky+1, kx+1]))

    # PW: co*Cin + ci
    pw_flat = []
    for co in range(10):
        for ci in range(3):
            pw_flat.append(int(pw_q[co, ci]))

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/dw3x3_c3.bin", "wb") as f: f.write(bytearray(dw_flat))
    with open("artifacts/pw1x1_c10x3.bin", "wb") as f: f.write(bytearray(pw_flat))
    with open("artifacts/labels.txt", "w", encoding="utf-8") as f:
        for d in range(10): f.write(f"{d}\n")

    # Optional: one sample BMP to test on-board
    from PIL import Image
    sample = (np.random.randint(0, 2, size=(32,32))*255).astype(np.uint8)
    sample_rgb = np.stack([sample,sample,sample], axis=-1)
    Image.fromarray(sample_rgb, mode="RGB").save("artifacts/sample_32x32.bmp", format="BMP")

    # Report sizes & SHA256 for sanity
    import hashlib
    for fn in ["dw3x3_c3.bin", "pw1x1_c10x3.bin", "labels.txt", "sample_32x32.bmp"]:
        p = os.path.join("artifacts", fn)
        with open(p, "rb") as f: data = f.read()
        sha = hashlib.sha256(data).hexdigest()
        print(f"{fn:>20}  size={len(data):4d}  sha256={sha}")
    print("Done.")

if __name__ == "__main__":
    main()

