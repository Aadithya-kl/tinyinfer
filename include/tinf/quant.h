// tinyinfer - weight quantization (mirrors tools/convert_hf.py::quantize)
//
//   q8: w ~= q * s,        q in [-127, 127] (int8),                         s = amax/127
//   q4: w ~= (n - 8) * s,  n in [1, 15], two per byte (low nibble = even),  s = amax/7
//   s is stored as fp16 per group of `group` consecutive weights in a row.
#pragma once

#include <cstdint>
#include <vector>

#include "tinf/model_file.h"

namespace tinf {

struct QuantizedMatrix {
    std::vector<uint8_t> payload;  // f16 bits / int8 / packed uint4
    std::vector<uint16_t> scales;  // fp16 bits (q8/q4)
    QMatrix view(int rows, int cols, Quant q, int group) const;
};

QuantizedMatrix quantize(const float* w, int rows, int cols, Quant q, int group);
// Dequantize one element (slow; for tests).
float dequant_at(const QMatrix& m, int r, int c);

}  // namespace tinf
