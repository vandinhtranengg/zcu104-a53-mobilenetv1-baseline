# A53 Bare‑Metal Quantized Digit Recognizer 

A minimal bare‑metal Cortex‑A53 demo that loads **32×32 RGB BMP** from SD, runs a quantized **MobileNet‑style** path 
(Depthwise 3×3 + ReLU6 → Pointwise 1×1 → Global Average Pool → Softmax), and prints Top‑K predictions for digits **0–9**.


## Project Lineage

This project is a continuation of, and builds upon, the SD-card image I/O and bare‑metal scaffolding from:
- **zcu104-baremetal-imgio** — https://github.com/vandinhtranengg/zcu104-baremetal-imgio

The prior repository provided SD file handling and BMP loading patterns that this demo reuses and extends with a quantized
MobileNet‑style pipeline (DW3×3 → ReLU6 → PW1×1 → GAP → Softmax) and digit classification (0–9).


## Features
- Reference NHWC uint8 kernels: `dwconv3x3_nhwc_u8`, `pwconv1x1_nhwc_u8`, `avgpool_global_nhwc_u8`, `softmax_u8`
- Single weight scale `w_scale` with `w_zp=128`
- SD assets: weights (`dw3x3_c3.bin`, `pw1x1_c10x3.bin`), `labels.txt`, BMP samples (`digit_0.bmp`..`digit_9.bmp`)

## Directory
- `vitis_src/` — Bare‑metal sources (`mobilenet_bm.cpp`, `ref_kernels.*`)
- `tools/` — Training & export scripts (synthetic digits)
- `assets/` — Quantized weights, labels, and sample images
- `docs/` — Deep-dive and quickstart

## Build (firmware)
> Requires Xilinx/AMD Vitis IDE with Standalone A53 BSP.  
> Create a new application project, add `vitis_src/` sources, and link FatFs (`xilffs`) & timer (`xtime_l`).

## SD Card Layout
```c
0:/assets/dw3x3_c3.bin        (27 bytes)  
0:/assets/pw1x1_c10x3.bin     (30 bytes)
0:/assets/labels.txt          (10 lines: 0..9)
0:/assets/samples             (sample image folder)
```

## Pre-built Hardware Platform (XSA)

This repository includes a pre-built hardware platform:
- **`standalone_zynq_core.xsa`** — Vivado-exported hardware (Zynq UltraScale+ MPSoC), suitable for creating a Standalone A53 domain in Vitis.

You can use this XSA directly to create the platform and domain (Standalone on A53), then import the firmware sources and build the bare‑metal app without opening Vivado.


## Getting Started

1. Build the bare‑metal app in Vitis/SDK (A53 Standalone).
2. Copy assets to `0:/assets/` on SD:
   - `dw3x3_c3.bin` (27 B), `pw1x1_c10x3.bin` (30 B), `labels.txt` (10 lines).
   - BMP test images: `digit_0.bmp`..`digit_9.bmp`.
3. Run on board; UART prints timings and Top‑K.

Troubleshooting:
- If timings show `DWConv: ms` only, replace float prints with integer **ms/µs**.
- If size mismatches occur, verify filenames and sizes:
  - DW = `Cin*3*3` = 27 bytes; PW = `Cout*Cin` = 30 bytes; labels = 10 lines.

---

## 🚀 What to do next

### 1) Swap in real MobileNetV1 weights (quantized INT8)

### 2) Introduce more blocks (DW + PW) to resemble MobileNetV1

---






