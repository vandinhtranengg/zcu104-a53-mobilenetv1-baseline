# zcu104-a53-mobilenetv1-baseline
A working bare‑metal classifier(minimal MobileNetV1‑style pipeline) for A53 on ZCU104 (Vivado/Vitis 2022.1)


# A53 Bare‑Metal Quantized Digit Recognizer (DW3×3 → ReLU6 → PW1×1 → GAP → Softmax)

A minimal bare‑metal Cortex‑A53 demo that loads **32×32 RGB BMP** from SD, runs a quantized **MobileNet‑style** path (Depthwise 3×3 + ReLU6 → Pointwise 1×1 → Global Average Pool → Softmax), and prints Top‑K predictions for digits **0–9**.

## Features
- Reference NHWC uint8 kernels: `dwconv3x3_nhwc_u8`, `pwconv1x1_nhwc_u8`, `avgpool_global_nhwc_u8`, `softmax_u8`
- Single weight scale `w_scale` with `w_zp=128`
- SD assets: weights (`dw3x3_c3.bin`, `pw1x1_c10x3.bin`), `labels.txt`, BMP samples (`digit_0.bmp`..`digit_9.bmp`)

## Directory
- `firmware/` — Bare‑metal sources (`mobilenet_bm.cpp`, `ref_kernels.*`)
- `tools/` — Training & export scripts (synthetic digits)
- `assets/` — Quantized weights, labels, and sample images
- `docs/` — Deep-dive and quickstart

## Build (firmware)
> Requires Xilinx/AMD Vitis or SDK with Standalone A53 BSP.  
> Create a new application project, add `firmware/` sources, and link FatFs (`xilffs`) & timer (`xtime_l`).

Key constants (in `mobilenet_bm.cpp`):
```c
const int Cin = 3;
const int Cout = 10;

const float in_scale = 0.02f;  const int in_zp = 128;
const float w_scale  = /* your exported value */;  const int w_zp = 128;
const float out_scale = 0.02f; const int out_zp = 128;
const float sm_scale  = 1.0f/255.0f; const int sm_zp = 0;
