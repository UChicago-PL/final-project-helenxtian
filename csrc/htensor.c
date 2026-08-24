#include <stddef.h>

void htensor_matmul_f32(
    const float *left,
    const float *right,
    float *result,
    size_t rows,
    size_t inner,
    size_t cols)
{
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            float sum = 0.0f;
            for (size_t index = 0; index < inner; ++index) {
                sum += left[row * inner + index] * right[index * cols + col];
            }
            result[row * cols + col] = sum;
        }
    }
}