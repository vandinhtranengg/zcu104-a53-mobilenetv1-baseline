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

### System Architecture Suggestion

Two standalone HLS IPs with **AXI4-Stream** data and **AXI4-Lite** control:
```c
MM2S (AXI DMA)   →   DW3x3 IP   →   PW1x1 IP   →   S2MM (AXI DMA)
                   (AXI-Lite)    (AXIS-Lite)
```
            

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
### Notice: Depthwise (DW) and Pointwise (PW) compute characteristics

- **DW (Depthwise Convolution)**  
  - **Low compute per byte** → bandwidth‑bound rather than compute‑bound.  
  - Best to **maximize channel parallelism**, **use line buffers**, and apply **DW→PW fusion** to reduce unnecessary DRAM round‑trips.

- **PW (1×1 Pointwise Convolution)**  
  - Behaves like **GEMM** and typically **dominates both runtime and parameter size**.  
  - Optimize using **aggressive tiling** and **on‑chip data reuse**, keeping both weights and activation tiles in **BRAM** whenever possible.

---

## Suggested HLS Skeleton for Implementation
- **Put the systolic array on PW (1×1)** and treat it as **GEMM** with tiling: M = H·W, K = Cin, N = Cout.
- **Handle DW (3×3) as a streaming stencil** using line buffers and channel parallelism.
- All arithmetic is **INT8 × INT8 → INT32 accumulate**, then **requantize to u8**.
- Provide a **DW→PW fused top** that streams DW output directly into the PW systolic array to **avoid DRAM traffic**.

### 0) Common types & quant helpers — accel_common.hpp
```c

// Common types (used by all blocks)
using axis_cvec_t = Axis<DW_BITS>;   // DW_BITS = CHAN_ALIGN * 8
using q24_t      = ap_int<32>;       // Q24 requant multiplier

```

### 1) DW 3×3 streaming stencil — dw3x3_stream.hpp
- **AXIS in (u8 NHWC vectors)** → **line buffers** + **3×3 window** per channel → **INT32 MAC** → **Q24 requant** → **ReLU6 clamp** → **AXIS out (u8 NHWC vectors)**.
- **Channel parallelism**: process P_C channels in parallel each cycle.
- Designed to feed PW directly (DW→PW fusion).
```c
void dw3x3_stream(
  hls::stream<axis_cvec_t> &s_in,     // AXIS(u8, NHWC, CHAN_ALIGN per beat)
  hls::stream<axis_cvec_t> &s_out,    // AXIS(u8, same width)
  const ap_uint<8>* w_dw,             // m_axi: [C * 9], layout c*9 + ky*3 + kx
  int H, int W, int C, int stride,    // stride = 1 (skeleton)
  int in_zp, int w_zp, int out_zp,    // quant zero-points
  q24_t alpha_q,                      // Q24: (in_s * w_s / out_s)
  ap_uint<8> relu6_qmax               // out_zp + round(6/out_s)
);
//-------------Pseudocode---------------------
for y in [0..H-1]:
  for x in [0..W-1]:
    for cb in [0..(C/CHAN_ALIGN)-1]:      // channel-block index
      (u8[CHAN_ALIGN]) pix_pack = AXIS_read(s_in)          // one beat
      // 1) Line buffers: write current pack into row ring (y % 3) at column x
      LB[(y % 3)][x][:] = pix_pack[:]

      // 2) 3×3 window per lane (zero-pad at borders)
      for pc in 0..CHAN_ALIGN-1:
        WIN[pc] = gather_3x3_from_LB(LB, y, x, pc)

      // 3) Load DW weights for current channel block (9 taps per lane)
      WDWC[CHAN_ALIGN][3][3] = load_from(w_dw, c_base=cb*CHAN_ALIGN)

      // 4) Per-lane compute: INT8xINT8→INT32, Q24 requant, ReLU6 clamp
      for pc in 0..CHAN_ALIGN-1:
        acc = sum_{ky,kx}( (WIN[pc][ky][kx] - in_zp) * (WDWC[pc][ky][kx] - w_zp) )
        r   = (acc * alpha_q) >> 24
        q   = out_zp + r
        q   = clamp(q, lo=out_zp, hi=relu6_qmax)
        out_pack[pc] = saturate_u8(q)

      // 5) Pack & write one AXIS beat; TLAST asserted only on final (y,x,cb)
      AXIS_write(s_out, out_pack, last = (y==H-1 && x==W-1 && cb==C/CHAN_ALIGN-1))
```

### 2) PW 1×1 systolic (GEMM) — pw1x1_systolic.hpp
- Consumes DW output stream (NHWC u8) or an external stream and treats PW as GEMM:
- A (M×K) × B (K×N) → C (M×N), where M = H·W, K = Cin, N = Cout.
- Tiling with on‑chip A_tile (Tk) and B_tile (Tk×Tn); PE mesh computes Tn outputs per activation vector.
- Weight‑stationary flavor shown (weights kept in local tile while streaming activations).
```c
void pw1x1_systolic(
  hls::stream<axis_cvec_t> &s_in,     // AXIS(u8, NHWC)
  hls::stream<axis_cvec_t> &s_out,    // AXIS(u8, NHWC)
  const ap_uint<8>* w_pw,             // m_axi: [Cout * Cin], row-major co*Cin + ci
  int H, int W, int Cin, int Cout,
  int in_zp, int w_zp, int out_zp,    // quant zero-points
  q24_t alpha_q                       // Q24: (in_s * w_s / out_s)
)
//-------------Pseudocode---------------------



```
 
### 3) DW→PW fused top (streaming) — dw_pw_fused_top.cpp
- Builds dataflow pipeline: DW stream → PW systolic with no intermediate DRAM.
- AXI‑Lite arguments set all quant parameters consistently.
```c
void dw_pw_fused_top(
  hls::stream<axis_cvec_t> &s_in,
  hls::stream<axis_cvec_t> &s_out,
  const ap_uint<8>* w_dw,             // DW weights
  const ap_uint<8>* w_pw,             // PW weights
  int H, int W, int Cin, int Cout, int stride,
  // DW quant
  int dw_in_zp, int dw_w_zp, int dw_out_zp, q24_t dw_alpha_q, ap_uint<8> dw_relu6_qmax,
  // PW quant (note: pw_in_zp == dw_out_zp)
  int pw_in_zp, int pw_w_zp, int pw_out_zp, q24_t pw_alpha_q
);
//-------------Pseudocode---------------------

create FIFO s_dw2pw (depth ~ 32–128)

DATAFLOW {
  // Stage 1: DW 3×3 (streams NHWC u8 vectors out)
  dw3x3_stream(s_in, s_dw2pw, w_dw, H, W, Cin, stride,
               dw_in_zp, dw_w_zp, dw_out_zp, dw_alpha_q, dw_relu6_qmax)

  // Stage 2: PW 1×1 GEMM (consumes DW stream directly)
  pw1x1_systolic(s_dw2pw, s_out, w_pw, H, W, Cin, Cout,
                 pw_in_zp /*= dw_out_zp*/, pw_w_zp, pw_out_zp, pw_alpha_q)
}

```



