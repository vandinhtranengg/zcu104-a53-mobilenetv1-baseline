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
### 3) Buil the accelerator HLS IPs for:  
- DepthwiseConv3×3 Operator
```c
void dwconv3x3_nhwc_u8(const tensor_u8_nhwc_t *in,const uint8_t *k3x3,const int32_t *bias,
                       float w_scale,int w_zp,tensor_u8_nhwc_t *out,int apply_relu6){
  int H=in->H,W=in->W,C=in->C;
  for(int y=0;y<H;++y)for(int x=0;x<W;++x)for(int c=0;c<C;++c){
    float acc=(bias? (float)bias[c]*in->scale*w_scale : 0.0f);
    for(int ky=-1;ky<=1;++ky){int iy=y+ky; if(iy<0||iy>=H) continue;
      for(int kx=-1;kx<=1;++kx){int ix=x+kx; if(ix<0||ix>=W) continue;
        uint8_t q_in=in->data[(iy*W+ix)*C+c];
        uint8_t q_k=k3x3[c*9+(ky+1)*3+(kx+1)];
        acc+=deq(q_in,in->scale,in->zp)*deq(q_k,w_scale,w_zp);
      }}
    uint8_t q=req(acc,out->scale,out->zp);
    if(apply_relu6){
      int q6=out->zp+(int)lrintf(6.0f/out->scale);
      if(q>q6) q=(q6>255?255:(q6<0?0:(uint8_t)q6));
      if(q<out->zp) q=(uint8_t)out->zp;
    }
    out->data[(y*W+x)*C+c]=q;
  }
}
```
- PWConv 1×1 Operator
```c
void pwconv1x1_nhwc_u8(const tensor_u8_nhwc_t *in,const uint8_t *k1x1,const int32_t *bias,
                       float w_scale,int w_zp,tensor_u8_nhwc_t *out){
  int H=in->H,W=in->W,Cin=in->C,Cout=out->C;
  for(int y=0;y<H;++y)for(int x=0;x<W;++x){
    const uint8_t* vin=&in->data[(y*W+x)*Cin];
    for(int co=0;co<Cout;++co){
      float acc=(bias? (float)bias[co]*in->scale*w_scale:0.0f);
      const uint8_t* wrow=&k1x1[co*Cin];
      for(int ci=0;ci<Cin;++ci){ acc+=deq(vin[ci],in->scale,in->zp)*deq(wrow[ci],w_scale,w_zp); }
      out->data[(y*W+x)*Cout+co]=req(acc,out->scale,out->zp);
    }
  }
}
```

### System architecture Suggestion

Two standalone HLS IPs with **AXI4-Stream** data and **AXI4-Lite** control:
MM2S (AXI DMA)   →   DW3x3 IP   →   PW1x1 IP   →   S2MM (AXI DMA)
                    (AXI-Lite)    (AXIS-Lite)
            

- **MM2S — Memory-Mapped to Stream**
  - **Direction:** DDR → AXI4-Stream  
  - **Purpose:** Reads data from memory and sends it as a streaming interface (feeding input data to an accelerator).  
  - **Data path:**  
    `DDR (AXI4-MM) → AXI DMA (MM2S) → AXI4-Stream → Accelerator`

- **S2MM — Stream to Memory-Mapped**
  - **Direction:** AXI4-Stream → DDR  
  - **Purpose:** Receives streaming data and writes it back to memory (collecting results from an accelerator).  
  - **Data path:**  
    `Accelerator → AXI4-Stream → AXI DMA (S2MM) → DDR (AXI4-MM)`

- **AXI4-Lite registers per IP** for control/status.

- **Standalone mode:** you can run each IP with MM2S/S2MM separately  
  (e.g., `DW → DDR`, then `DDR → PW`) while you debug.

- **Chained mode:** once stable, connect `DW`’s `M_AXIS` directly to `PW`’s `S_AXIS` to avoid intermediate DDR traffic.


---






