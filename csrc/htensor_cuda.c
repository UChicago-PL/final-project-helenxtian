#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

static char htensor_cuda_error[256] = "no CUDA error";

static int set_cuda_error(const char *operation, cudaError_t status)
{
    snprintf(
        htensor_cuda_error,
        sizeof(htensor_cuda_error),
        "%s: %s",
        operation,
        cudaGetErrorString(status));
    return (int)status;
}

static int set_cublas_error(const char *operation, cublasStatus_t status)
{
    snprintf(
        htensor_cuda_error,
        sizeof(htensor_cuda_error),
        "%s: cuBLAS status %d",
        operation,
        (int)status);
    return 1000 + (int)status;
}

const char *htensor_cuda_last_error(void)
{
    return htensor_cuda_error;
}

int htensor_cuda_device_name(char *buffer, size_t buffer_size)
{
    int device = 0;
    cudaDeviceProp properties;
    cudaError_t status;

    if (buffer_size == 0) {
        return -1;
    }
    status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return set_cuda_error("cudaGetDevice", status);
    }
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        return set_cuda_error("cudaGetDeviceProperties", status);
    }
    snprintf(buffer, buffer_size, "%s", properties.name);
    return 0;
}

int htensor_cuda_matmul_f32(
    const float *left,
    const float *right,
    float *result,
    size_t rows,
    size_t inner,
    size_t cols,
    float *h2d_ms,
    float *kernel_ms,
    float *d2h_ms,
    float *device_total_ms)
{
    float *device_left = NULL;
    float *device_right = NULL;
    float *device_result = NULL;
    cublasHandle_t handle = NULL;
    cudaEvent_t start = NULL;
    cudaEvent_t after_h2d = NULL;
    cudaEvent_t after_kernel = NULL;
    cudaEvent_t after_d2h = NULL;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const size_t left_bytes = rows * inner * sizeof(float);
    const size_t right_bytes = inner * cols * sizeof(float);
    const size_t result_bytes = rows * cols * sizeof(float);
    int code = 0;
    cudaError_t cuda_status;
    cublasStatus_t cublas_status;

    if (rows == 0 || inner == 0 || cols == 0 ||
        rows > INT_MAX || inner > INT_MAX || cols > INT_MAX ||
        rows > SIZE_MAX / inner || inner > SIZE_MAX / cols ||
        rows > SIZE_MAX / cols) {
        snprintf(htensor_cuda_error, sizeof(htensor_cuda_error), "invalid or overflowing matrix dimensions");
        return -1;
    }

#define CHECK_CUDA(operation) \
    do { \
        cuda_status = (operation); \
        if (cuda_status != cudaSuccess) { \
            code = set_cuda_error(#operation, cuda_status); \
            goto cleanup; \
        } \
    } while (0)

#define CHECK_CUBLAS(operation) \
    do { \
        cublas_status = (operation); \
        if (cublas_status != CUBLAS_STATUS_SUCCESS) { \
            code = set_cublas_error(#operation, cublas_status); \
            goto cleanup; \
        } \
    } while (0)

    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUDA(cudaMalloc((void **)&device_left, left_bytes));
    CHECK_CUDA(cudaMalloc((void **)&device_right, right_bytes));
    CHECK_CUDA(cudaMalloc((void **)&device_result, result_bytes));
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&after_h2d));
    CHECK_CUDA(cudaEventCreate(&after_kernel));
    CHECK_CUDA(cudaEventCreate(&after_d2h));

    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpyAsync(device_left, left, left_bytes, cudaMemcpyHostToDevice, 0));
    CHECK_CUDA(cudaMemcpyAsync(device_right, right, right_bytes, cudaMemcpyHostToDevice, 0));
    CHECK_CUDA(cudaEventRecord(after_h2d, 0));

    /* Row-major C = A * B is column-major C^T = B^T * A^T. */
    CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        (int)cols,
        (int)rows,
        (int)inner,
        &alpha,
        device_right,
        (int)cols,
        device_left,
        (int)inner,
        &beta,
        device_result,
        (int)cols));
    CHECK_CUDA(cudaEventRecord(after_kernel, 0));
    CHECK_CUDA(cudaMemcpyAsync(result, device_result, result_bytes, cudaMemcpyDeviceToHost, 0));
    CHECK_CUDA(cudaEventRecord(after_d2h, 0));
    CHECK_CUDA(cudaEventSynchronize(after_d2h));

    CHECK_CUDA(cudaEventElapsedTime(h2d_ms, start, after_h2d));
    CHECK_CUDA(cudaEventElapsedTime(kernel_ms, after_h2d, after_kernel));
    CHECK_CUDA(cudaEventElapsedTime(d2h_ms, after_kernel, after_d2h));
    CHECK_CUDA(cudaEventElapsedTime(device_total_ms, start, after_d2h));

cleanup:
    if (after_d2h != NULL) cudaEventDestroy(after_d2h);
    if (after_kernel != NULL) cudaEventDestroy(after_kernel);
    if (after_h2d != NULL) cudaEventDestroy(after_h2d);
    if (start != NULL) cudaEventDestroy(start);
    if (device_result != NULL) cudaFree(device_result);
    if (device_right != NULL) cudaFree(device_right);
    if (device_left != NULL) cudaFree(device_left);
    if (handle != NULL) cublasDestroy(handle);
    return code;

#undef CHECK_CUDA
#undef CHECK_CUBLAS
}