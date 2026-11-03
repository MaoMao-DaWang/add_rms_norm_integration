#pragma once
#include <cstdint>
extern "C" void arm_launch(void* x1, void* x2, void* gamma,
    void* y, void* rstd, void* residual, uint32_t rows, uint32_t cols,
    int32_t dtype, uint32_t cores, float epsilon, int32_t modelCompatible,
    void* stream);
