
// Bare-metal A53 demo: SD BMP -> Preproc -> DWConv3x3+ReLU6 -> PWConv1x1 -> AvgPool -> Softmax -> Top-5
#include "xil_printf.h"
#include "xil_cache.h"
#include "ff.h"            // FatFs (xilffs)
#include "xtime_l.h"       // COUNTS_PER_SECOND, XTime_GetTime
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include "ref_kernels.h"


using namespace std;

//Quick diagnostic: list the SD directory
//static void list_dir(const char* path) {
//    DIR dir; FILINFO fno;
//    FRESULT rc = f_opendir(&dir, path);
//    xil_printf("opendir(%s) rc=%d\r\n", path, rc);
//    if (rc != FR_OK) return;
//    for (;;) {
//        rc = f_readdir(&dir, &fno);
//        if (rc != FR_OK || fno.fname[0] == 0) break; // end
//        // When LFN is OFF, fno.fname shows the 8.3 alias (e.g., TEST_3~1.BMP).
//        xil_printf("  %s\r\n", fno.fname);
//    }
//    f_closedir(&dir);
//}
// call it:
//list_dir("0:/");
//list_dir("0:/ASSETS");  // or "0:/assets" if you used lower-case


// Helper: print integer milliseconds with xil_printf
static inline void print_ms(XTime t0, XTime t1) {
    u64 diff = t1 - t0;
    // integer ms (avoid float completely)
    u32 ms = (u32)((diff * 1000U) / COUNTS_PER_SECOND);
    xil_printf("%u ms\r\n", ms);
}


static inline u32 ms_from_counts(XTime t0, XTime t1) {
    u64 diff = t1 - t0;
    return (u32)((diff * 1000U) / COUNTS_PER_SECOND);
}
static inline u32 us_from_counts(XTime t0, XTime t1) {
    u64 diff = t1 - t0;
    return (u32)((diff * 1000000U) / COUNTS_PER_SECOND);
}
static inline void print_ms_us(XTime t0, XTime t1) {
    xil_printf("%u ms (%u us)\r\n", ms_from_counts(t0, t1), us_from_counts(t0, t1));
}


// ---------- SD helpers ----------
static FATFS fatfs;
static FIL   file;

static bool sd_mount() {
    FRESULT rc = f_mount(&fatfs, "0:/", 1);
    if(rc != FR_OK){ xil_printf("f_mount failed: %d\r\n", rc); return false; }
    xil_printf("SD mounted: 0:/\r\n");
    return true;
}
static bool sd_read_all(const char* path, vector<uint8_t>& out){
    FRESULT rc = f_open(&file, path, FA_READ);
    if(rc != FR_OK){ xil_printf("f_open %s failed: %d\r\n", path, rc); return false; }
    UINT fsize = f_size(&file);
    xil_printf("File %s size: %u bytes\r\n", path, fsize);
    out.resize(fsize);
    UINT br=0; rc = f_read(&file, out.data(), fsize, &br); f_close(&file);
    if(rc!=FR_OK || br!=fsize){ xil_printf("f_read failed rc=%d br=%u size=%u\r\n", rc, br, fsize); return false; }
    return true;
}

// ---------- Minimal BMP loader (24-bit BI_RGB) ----------
#pragma pack(push,1)
struct BMPFileHeader{ uint16_t bfType; uint32_t bfSize; uint16_t r1; uint16_t r2; uint32_t bfOffBits; };
struct BMPInfoHeader{ uint32_t biSize; int32_t biWidth; int32_t biHeight; uint16_t biPlanes; uint16_t biBitCount; uint32_t biCompression; uint32_t biSizeImage; int32_t biXPelsPerMeter; int32_t biYPelsPerMeter; uint32_t biClrUsed; uint32_t biClrImportant; };
#pragma pack(pop)

static bool load_bmp_24_stream(const char* path, int& W, int& H, vector<uint8_t>& rgb){
    FRESULT rc = f_open(&file, path, FA_READ); if(rc!=FR_OK){ xil_printf("open %s failed\r\n", path); return false; }
    BMPFileHeader fh; BMPInfoHeader ih; UINT br=0;
    rc = f_read(&file, &fh, sizeof(fh), &br); if(rc!=FR_OK||br!=sizeof(fh)){ xil_printf("read FH failed\r\n"); f_close(&file); return false; }
    rc = f_read(&file, &ih, sizeof(ih), &br); if(rc!=FR_OK||br!=sizeof(ih)){ xil_printf("read IH failed\r\n"); f_close(&file); return false; }
    if(fh.bfType!=0x4D42 || ih.biSize!=40 || ih.biPlanes!=1 || ih.biBitCount!=24 || ih.biCompression!=0){ xil_printf("unsupported BMP\r\n"); f_close(&file); return false; }
    W = ih.biWidth; H = (ih.biHeight>0? ih.biHeight : -ih.biHeight); bool bottom_up = (ih.biHeight>0);
    uint32_t row_stride = ((W*3 + 3) & ~3);
    rc = f_lseek(&file, fh.bfOffBits); if(rc!=FR_OK){ xil_printf("seek pixels failed\r\n"); f_close(&file); return false; }
    rgb.resize(W*H*3);
    vector<uint8_t> row(row_stride);
    for(int y=0;y<H;++y){
        UINT rb=0; rc=f_read(&file,row.data(),row_stride,&rb); if(rc!=FR_OK||rb!=row_stride){ xil_printf("row read failed\r\n"); f_close(&file); return false; }
        int dst_y = bottom_up? (H-1-y) : y;
        uint8_t* dst = &rgb[(dst_y*W)*3];
        for(int x=0;x<W;++x){ // BGR->RGB
            dst[x*3+0]=row[x*3+2]; dst[x*3+1]=row[x*3+1]; dst[x*3+2]=row[x*3+0];
        }
    }
    f_close(&file);
    return true;
}

// ---------- Labels ----------
static vector<string> load_labels(const char* path){
    vector<string> labels; vector<uint8_t> buf;
    if(!sd_read_all(path, buf)) return labels;
    string s((char*)buf.data(), (char*)buf.data()+buf.size());
    size_t pos=0;
    while(pos<s.size()){
        size_t nl=s.find('\n',pos); if(nl==string::npos) nl=s.size();
        string line=s.substr(pos,nl-pos); if(!line.empty() && line.back()=='\r') line.pop_back();
        if(!line.empty()) labels.push_back(line);
        pos=(nl==s.size()? nl : nl+1);
    }
    return labels;
}

// ---------- Top-5 ----------
static void print_top5(const std::vector<float>& probs, const std::vector<std::string>& labels) {
    struct Score { int idx; float val; };
    std::vector<Score> s; s.reserve(probs.size());
    for (size_t i = 0; i < probs.size(); ++i) s.push_back({(int)i, probs[i]});
    std::sort(s.begin(), s.end(), [](const Score& a, const Score& b){ return a.val > b.val; });

    int k = std::min(5, (int)s.size());
    xil_printf("Top-%d:\r\n", k);
    for (int i = 0; i < k; ++i) {
        const char* name = (s[i].idx < (int)labels.size() ? labels[s[i].idx].c_str() : "(no-label)");
        u32 score255 = (u32)lrintf(s[i].val * 255.0f);
        u32 pct      = (u32)lrintf(s[i].val * 100.0f);
        xil_printf(" %d) %s : %u/255 (%u%%)\r\n", i+1, name, score255, pct);
    }
}


// ---------- Main ----------
int main(){

    xil_printf("mobilenet_bm: bare-metal demo\r\n");
    if(!sd_mount()) return -1;

    // SD assets required:
    //   0:/assets/test_32x32.bmp        (24-bit BMP, 32x32 RGB)
    //   0:/assets/labels.txt            (8 lines, one per class)
    //   0:/assets/dw3x3_c3.bin          (DW weights, Cin=3, 3x3 per channel)
    //   0:/assets/pw1x1_c8x3.bin        (PW weights, Cout=8, Cin=3)

    int W=0,H=0; vector<uint8_t> rgb;
    if(!load_bmp_24_stream("0:/assets/samples/digit_1.bmp", W, H, rgb)){ xil_printf("BMP load failed\r\n"); return -1; }
    xil_printf("Image 2 loaded: %dx%d\r\n", W, H);
    if(W!=32 || H!=32){ xil_printf("For this demo, please use 32x32 BMP.\r\n"); return -1; }

    // Quant params - scale
    const float in_scale = 0.02f;  const int in_zp = 128;
    const float w_scale  = 0.0019579321f;  const int w_zp = 128;
    const float out_scale = 0.02f; const int out_zp = 128;
    const float sm_scale = 1.0f/255.0f; const int sm_zp = 0;
//    const float in_scale=0.02f; const int in_zp=128;
//    const float w_scale=0.02f;  const int w_zp=128;
//    const float out_scale=0.02f; const int out_zp=128;
//    const float sm_scale=1.0f/255.0f; const int sm_zp=0;

    // NHWC buffers - class count
    const int Cin=3; const int Cout=10;
    vector<uint8_t> in_u8(W*H*Cin), dw_out(W*H*Cin), pw_out(W*H*Cout), sm_u8(Cout);

    // Preproc: RGB -> NHWC uint8 (3 channels)
    for(int y=0;y<H;++y){
        for(int x=0;x<W;++x){
            uint8_t r=rgb[(y*W+x)*3+0], g=rgb[(y*W+x)*3+1], b=rgb[(y*W+x)*3+2];
            in_u8[(y*W+x)*Cin+0]=r; in_u8[(y*W+x)*Cin+1]=g; in_u8[(y*W+x)*Cin+2]=b;
        }
    }

    // Load weights
    vector<uint8_t> dw_w, pw_w;
    if(!sd_read_all("0:/assets/dw3x3_c3.bin", dw_w)) return -1;
    if((int)dw_w.size()!=Cin*3*3){ xil_printf("DW weights size mismatch\r\n"); return -1; }
    if(!sd_read_all("0:/assets/pw1x1_c10x3.bin", pw_w)) return -1;
    if((int)pw_w.size()!=Cout*Cin){ xil_printf("PW weights size mismatch\r\n"); return -1; }

    // Labels
    vector<string> labels = load_labels("0:/assets/labels.txt");
    if(labels.size()!= (size_t)Cout){ xil_printf("labels count != Cout (%d)\r\n", Cout); }

    // size checks
    // DW: Cin*3*3 == 27, PW: Cout*Cin == 30, labels: 10 lines (20 bytes)


    // Tensors
    tensor_u8_nhwc_t tin={H,W,Cin,in_u8.data(), in_scale, in_zp};
    tensor_u8_nhwc_t tdw={H,W,Cin,dw_out.data(), out_scale, out_zp};
    tensor_u8_nhwc_t tpw={H,W,Cout,pw_out.data(), out_scale, out_zp};
    tensor_u8_nhwc_t tsm_in={1,1,Cout,sm_u8.data(), sm_scale, sm_zp};
    tensor_u8_nhwc_t tsm_out={1,1,Cout,sm_u8.data(), sm_scale, sm_zp};


    // --- (Optional but recommended) use reference avgpool ---
    tensor_u8_nhwc_t tavg_out = {1,1,Cout, sm_u8.data(), sm_scale, sm_zp};
    avgpool_global_nhwc_u8(&tpw, &tavg_out);


    // Runtimes
    XTime t0,t1;

    // Depthwise
    XTime_GetTime(&t0);
    dwconv3x3_nhwc_u8(&tin, dw_w.data(), NULL, w_scale, w_zp, &tdw, 1);
    XTime_GetTime(&t1);
    xil_printf("DWConv: "); print_ms_us(t0, t1);

    // Pointwise
    XTime_GetTime(&t0);
    pwconv1x1_nhwc_u8(&tdw, pw_w.data(), NULL, w_scale, w_zp, &tpw);
    XTime_GetTime(&t1);
    xil_printf("PWConv: "); print_ms_us(t0, t1);

    // Global average pool -> softmax input (sm_u8)
    XTime_GetTime(&t0);
    for(int c=0;c<Cout;++c){
        float sum=0.0f;
        for(int y=0;y<H;++y) for(int x=0;x<W;++x){
            uint8_t q=pw_out[(y*W+x)*Cout+c];
            sum += (out_scale*((int)q - out_zp));
        }
        float mean=sum/(float)(H*W);
        sm_u8[c] = (uint8_t)((int)lrintf(mean/sm_scale));
    }
    XTime_GetTime(&t1);
    xil_printf("AvgPool: "); print_ms_us(t0, t1);

    // Softmax
    XTime_GetTime(&t0);
    softmax_u8(&tsm_in, &tsm_out);
    XTime_GetTime(&t1);
    xil_printf("Softmax: "); print_ms_us(t0, t1);

    // To float and Top-5
    vector<float> probs(Cout);
    for(int c=0;c<Cout;++c) probs[c] = sm_scale * ((int)sm_u8[c] - sm_zp);
    print_top5(probs, labels);

    // RAM footprint (allocated buffers)
    xil_printf("RAM bytes: in=%lu dw_out=%lu pw_out=%lu sm=%lu\r\n",
        (unsigned long)in_u8.size(), (unsigned long)dw_out.size(),
        (unsigned long)pw_out.size(), (unsigned long)sm_u8.size());

    xil_printf("Done.\r\n");


    while(1){}
    return 0;
}

