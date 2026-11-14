#include "tinf/quant.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "tinf/half.h"

namespace tinf {

QMatrix QuantizedMatrix::view(int rows, int cols, Quant q, int group) const {
    QMatrix m;
    m.w = payload.data();
    m.scales = scales.empty() ? nullptr : scales.data();
    m.rows = rows;
    m.cols = cols;
    m.quant = q;
    m.group = group;
    return m;
}

QuantizedMatrix quantize(const float* w, int rows, int cols, Quant q, int group) {
    QuantizedMatrix out;
    size_t n = (size_t)rows * cols;
    if (q == Quant::F16) {
        out.payload.resize(2 * n);
        uint16_t* h = reinterpret_cast<uint16_t*>(out.payload.data());
        for (size_t i = 0; i < n; ++i) h[i] = float_to_half(w[i]);
        return out;
    }
    if (cols % group) throw std::runtime_error("quantize: cols not a multiple of group");
    const float qmax = q == Quant::Q8 ? 127.f : 7.f;
    const int ng = cols / group;
    out.scales.resize((size_t)rows * ng);
    out.payload.resize(q == Quant::Q8 ? n : n / 2);
    for (int r = 0; r < rows; ++r) {
        for (int g = 0; g < ng; ++g) {
            const float* src = w + (size_t)r * cols + (size_t)g * group;
            float amax = 0.f;
            for (int i = 0; i < group; ++i) amax = std::max(amax, std::fabs(src[i]));
            uint16_t sh = float_to_half(amax / qmax);
            float s = half_to_float(sh);
            float inv = s > 0.f ? 1.f / s : 0.f;
            out.scales[(size_t)r * ng + g] = sh;
            for (int i = 0; i < group; ++i) {
                float v = std::nearbyint(src[i] * inv);  // round-half-even, like np.rint
                int qi = (int)std::min(qmax, std::max(-qmax, v));
                size_t idx = (size_t)r * cols + (size_t)g * group + i;
                if (q == Quant::Q8) {
                    out.payload[idx] = (uint8_t)(int8_t)qi;
                } else {
                    uint8_t nib = (uint8_t)(qi + 8);
                    uint8_t& b = out.payload[idx / 2];
                    if (idx % 2 == 0)
                        b = (uint8_t)((b & 0xF0) | nib);
                    else
                        b = (uint8_t)((b & 0x0F) | (nib << 4));
                }
            }
        }
    }
    return out;
}

float dequant_at(const QMatrix& m, int r, int c) {
    size_t idx = (size_t)r * m.cols + c;
    switch (m.quant) {
        case Quant::F16: return half_to_float(static_cast<const uint16_t*>(m.w)[idx]);
        case Quant::Q8: {
            float s = half_to_float(m.scales[(size_t)r * (m.cols / m.group) + c / m.group]);
            return (float)static_cast<const int8_t*>(m.w)[idx] * s;
        }
        case Quant::Q4: {
            float s = half_to_float(m.scales[(size_t)r * (m.cols / m.group) + c / m.group]);
            uint8_t b = static_cast<const uint8_t*>(m.w)[idx / 2];
            int nib = (idx % 2 == 0) ? (b & 15) : (b >> 4);
            return (float)(nib - 8) * s;
        }
    }
    return 0.f;
}

}  // namespace tinf
