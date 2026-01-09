#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct { int H; int W; int C; uint8_t *data; float scale; int zp; } tensor_u8_nhwc_t;

void dwconv3x3_nhwc_u8(const tensor_u8_nhwc_t *in, const uint8_t *k3x3, const int32_t *bias,
                       float w_scale, int w_zp, tensor_u8_nhwc_t *out, int apply_relu6);
void pwconv1x1_nhwc_u8(const tensor_u8_nhwc_t *in, const uint8_t *k1x1, const int32_t *bias,
                       float w_scale, int w_zp, tensor_u8_nhwc_t *out);
void avgpool_global_nhwc_u8(const tensor_u8_nhwc_t *in, tensor_u8_nhwc_t *out);
void softmax_u8(const tensor_u8_nhwc_t *in, tensor_u8_nhwc_t *out);

#ifdef __cplusplus
}
#endif
