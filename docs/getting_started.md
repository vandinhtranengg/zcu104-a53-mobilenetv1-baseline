
# Getting Started

1. Build the bare‑metal app in Vitis/SDK (A53 Standalone).
2. Copy assets to `0:/assets/` on SD:
   - `dw3x3_c3.bin` (27 B), `pw1x1_c10x3.bin` (30 B), `labels.txt` (10 lines).
   - BMP test images: `digit_0.bmp`..`digit_9.bmp`.
3. Run on board; UART prints timings and Top‑K.

Troubleshooting:
- If timings show `DWConv: ms` only, replace float prints with integer **ms/µs**.
- If size mismatches occur, verify filenames and sizes:
  - DW = `Cin*3*3` = 27 bytes; PW = `Cout*Cin` = 30 bytes; labels = 20 bytes(10 lines).
