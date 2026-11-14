// tinyinfer - portable IEEE fp16 <-> fp32 conversion on the host (round-to-nearest-even)
#pragma once

#include <cstdint>
#include <cstring>

namespace tinf {

inline float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {  // subnormal: normalize
            int e = -1;
            do {
                e++;
                mant <<= 1;
            } while ((mant & 0x400u) == 0);
            mant &= 0x3FFu;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// Fabian Giesen's float_to_half_fast3_rtne (public domain); exact RNE incl. subnormals.
inline uint16_t float_to_half(float ff) {
    uint32_t f;
    std::memcpy(&f, &ff, 4);
    const uint32_t f32infty = 255u << 23;
    const uint32_t f16max = (127u + 16u) << 23;
    const uint32_t denorm_magic_u = ((127u - 15u) + (23u - 10u) + 1u) << 23;
    float denorm_magic;
    std::memcpy(&denorm_magic, &denorm_magic_u, 4);
    uint32_t sign = f & 0x80000000u;
    f ^= sign;
    uint16_t o;
    if (f >= f16max) {
        o = (f > f32infty) ? 0x7E00 : 0x7C00;  // NaN : Inf
    } else if (f < (113u << 23)) {             // result is subnormal or zero
        float tmp;
        std::memcpy(&tmp, &f, 4);
        tmp += denorm_magic;
        uint32_t t;
        std::memcpy(&t, &tmp, 4);
        o = (uint16_t)(t - denorm_magic_u);
    } else {
        uint32_t mant_odd = (f >> 13) & 1u;
        f += ((uint32_t)(15 - 127) << 23) + 0xFFFu;
        f += mant_odd;
        o = (uint16_t)(f >> 13);
    }
    return (uint16_t)(o | (sign >> 16));
}

// 256 KiB lookup table for fast fp16 -> fp32 in the CPU reference engine.
const float* half_table();

}  // namespace tinf
