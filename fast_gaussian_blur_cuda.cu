// ================================================================
// CUDA VERSION: FAST GAUSSIAN BLUR VỚI SONG SONG HÓA CUDA
// ================================================================
// Implementation CUDA của Fast Gaussian Blur
// Dựa trên thuật toán OpenMP nhưng được triển khai trên GPU
//
#include "fast_gaussian_blur_cuda.h"
#include "fast_gaussian_blur_template.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cstring>

// ================================================================
// CUDA KERNELS
// ================================================================

//! \brief CUDA kernel để thực hiện horizontal blur với chính sách extend
//! Mỗi thread xử lý một pixel trong một hàng
//! Sử dụng sliding window accumulator để đạt O(n) thay vì O(n*r)
template<int C>
__global__ void horizontal_blur_extend_kernel(
    const unsigned char* __restrict__ in,
    unsigned char* __restrict__ out,
    int w, int h, int r)
{
    // Tính chỉ số hàng mà thread này xử lý
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Kiểm tra bounds
    if (row >= h) return;
    
    // Tính chỉ số bắt đầu của hàng trong buffer 1D
    int begin = row * w;
    int end = begin + w;
    
    // Tính nghịch đảo kích thước kernel
    float iarr = 1.0f / (r + r + 1);
    
    // Khai báo accumulator cho mỗi channel
    int acc[C];
    for (int ch = 0; ch < C; ch++) {
        acc[ch] = 0;
    }
    
    // Lấy giá trị pixel đầu và cuối hàng (cho extend policy)
    int fv[C], lv[C];
    for (int ch = 0; ch < C; ch++) {
        fv[ch] = in[begin * C + ch];
        lv[ch] = in[(end - 1) * C + ch];
    }
    
    // Khởi tạo accumulator: giả sử có (r+1) pixel đầu có giá trị fv
    for (int ch = 0; ch < C; ch++) {
        acc[ch] = (r + 1) * fv[ch];
    }
    
    // Xử lý theo từng pixel trong hàng
    // Mỗi thread xử lý một pixel
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < w) {
        int ti = begin + col;  // target index
        
        // Tính toán accumulator cho pixel này
        // Với sliding window, ta cần tính từ đầu hàng đến vị trí hiện tại
        if (col == 0) {
            // Pixel đầu tiên: khởi tạo accumulator
            for (int j = 0; j <= r && j < w; j++) {
                for (int ch = 0; ch < C; ch++) {
                    acc[ch] += in[(begin + j) * C + ch];
                }
            }
            // Điều chỉnh vì đã khởi tạo với (r+1)*fv
            for (int ch = 0; ch < C; ch++) {
                acc[ch] -= (r + 1) * fv[ch];
            }
        } else {
            // Các pixel tiếp theo: cập nhật accumulator từ pixel trước
            // Đây là phần phức tạp vì cần shared memory hoặc atomic operations
            // Để đơn giản, ta sẽ tính lại cho mỗi pixel (không tối ưu nhất nhưng đúng)
            // Hoặc ta có thể dùng một thread xử lý toàn bộ hàng
        }
    }
}

//! \brief CUDA kernel để xử lý một hàng hoàn chỉnh
//! Mỗi thread block xử lý một hàng
//! Sử dụng shared memory để tối ưu
template<int C>
__global__ void horizontal_blur_extend_row_kernel(
    const unsigned char* __restrict__ in,
    unsigned char* __restrict__ out,
    int w, int h, int r)
{
    // Mỗi block xử lý một hàng
    int row = blockIdx.x;
    
    if (row >= h) return;
    
    int begin = row * w;
    int end = begin + w;
    
    float iarr = 1.0f / (r + r + 1);
    
    // Shared memory cho accumulator (nếu cần)
    // Khai báo accumulator local
    int acc[C];
    int fv[C], lv[C];
    
    // Lấy giá trị pixel đầu và cuối
    for (int ch = 0; ch < C; ch++) {
        fv[ch] = in[begin * C + ch];
        lv[ch] = in[(end - 1) * C + ch];
        acc[ch] = (r + 1) * fv[ch];
    }
    
    // Khởi tạo accumulator: tính tổng kernel đầu tiên
    int ti = begin;
    int li = begin - r - 1;
    int ri = begin + r;
    
    // Tính tổng ban đầu
    for (int j = ti; j < ri && j < end; j++) {
        for (int ch = 0; ch < C; ch++) {
            acc[ch] += in[j * C + ch];
        }
    }
    // Điều chỉnh vì đã khởi tạo với (r+1)*fv
    for (int ch = 0; ch < C; ch++) {
        acc[ch] -= (r + 1) * fv[ch];
    }
    
    // Xử lý từng pixel trong hàng (sequential trong mỗi thread block)
    // TRƯỜNG HỢP 1: Phần trái ngoài, phần phải trong
    for (; ri < end && li < begin; ri++, ti++, li++) {
        for (int ch = 0; ch < C; ch++) {
            acc[ch] += in[ri * C + ch] - fv[ch];
            out[ti * C + ch] = (unsigned char)(acc[ch] * iarr + 0.5f);
        }
    }
    
    // TRƯỜNG HỢP 2: Cả hai phần đều trong
    for (; ri < end; ri++, ti++, li++) {
        for (int ch = 0; ch < C; ch++) {
            acc[ch] += in[ri * C + ch] - in[li * C + ch];
            out[ti * C + ch] = (unsigned char)(acc[ch] * iarr + 0.5f);
        }
    }
    
    // TRƯỜNG HỢP 3: Phần trái trong, phần phải ngoài
    for (; ti < end; ti++, li++) {
        for (int ch = 0; ch < C; ch++) {
            acc[ch] += lv[ch] - in[li * C + ch];
            out[ti * C + ch] = (unsigned char)(acc[ch] * iarr + 0.5f);
        }
    }
}

//! \brief CUDA kernel để transpose ảnh theo block
//! Sử dụng shared memory để tối ưu
template<int C>
__global__ void flip_block_kernel(
    const unsigned char* __restrict__ in,
    unsigned char* __restrict__ out,
    int w, int h)
{
    // Block size cho transpose
    const int TILE_DIM = 32;
    const int BLOCK_ROWS = 8;
    
    __shared__ unsigned char tile[TILE_DIM][TILE_DIM * C + 1]; // +1 để tránh bank conflicts
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Đọc vào shared memory (coalesced)
    for (int i = 0; i < BLOCK_ROWS; i++) {
        int row = y + i * TILE_DIM / BLOCK_ROWS;
        if (row < h && x < w) {
            for (int ch = 0; ch < C; ch++) {
                tile[threadIdx.y + i * TILE_DIM / BLOCK_ROWS][threadIdx.x * C + ch] = 
                    in[row * w * C + x * C + ch];
            }
        }
    }
    
    __syncthreads();
    
    // Ghi ra với transpose (coalesced)
    x = blockIdx.y * TILE_DIM + threadIdx.x;  // Đổi chỗ x và y
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    for (int i = 0; i < BLOCK_ROWS; i++) {
        int row = y + i * TILE_DIM / BLOCK_ROWS;
        if (row < w && x < h) {
            for (int ch = 0; ch < C; ch++) {
                out[row * h * C + x * C + ch] = 
                    tile[threadIdx.x][(threadIdx.y + i * TILE_DIM / BLOCK_ROWS) * C + ch];
            }
        }
    }
}

// ================================================================
// HOST FUNCTIONS (WRAPPERS)
// ================================================================

extern "C" void horizontal_blur_cuda_extend(
    unsigned char* d_in,
    unsigned char* d_out,
    int w, int h, int c, int r)
{
    // Chọn kernel dựa trên số channels
    dim3 blockSize(1);  // Mỗi block xử lý một hàng
    dim3 gridSize(h);   // Một block cho mỗi hàng
    
    switch (c) {
        case 1:
            horizontal_blur_extend_row_kernel<1><<<gridSize, blockSize>>>(
                d_in, d_out, w, h, r);
            break;
        case 3:
            horizontal_blur_extend_row_kernel<3><<<gridSize, blockSize>>>(
                d_in, d_out, w, h, r);
            break;
        case 4:
            horizontal_blur_extend_row_kernel<4><<<gridSize, blockSize>>>(
                d_in, d_out, w, h, r);
            break;
        default:
            printf("Unsupported channel count: %d\n", c);
            return;
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in horizontal_blur_cuda_extend: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void flip_block_cuda(
    unsigned char* d_in,
    unsigned char* d_out,
    int w, int h, int c)
{
    const int TILE_DIM = 32;
    dim3 blockSize(TILE_DIM, 8);
    dim3 gridSize((w + TILE_DIM - 1) / TILE_DIM, (h + TILE_DIM - 1) / TILE_DIM);
    
    switch (c) {
        case 1:
            flip_block_kernel<1><<<gridSize, blockSize>>>(
                d_in, d_out, w, h);
            break;
        case 3:
            flip_block_kernel<3><<<gridSize, blockSize>>>(
                d_in, d_out, w, h);
            break;
        case 4:
            flip_block_kernel<4><<<gridSize, blockSize>>>(
                d_in, d_out, w, h);
            break;
        default:
            printf("Unsupported channel count: %d\n", c);
            return;
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in flip_block_cuda: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void fast_gaussian_blur_cuda(
    unsigned char* d_in,
    unsigned char* d_out,
    int w, int h, int c,
    float sigma, int n,
    Border border)
{
    // Chỉ hỗ trợ extend policy cho CUDA version
    if (border != kExtend) {
        printf("Warning: CUDA version only supports extend border policy, using extend\n");
    }
    
    // Tính box radius cho mỗi pass
    int boxes[10];  // Hỗ trợ tối đa 10 passes
    if (n > 10) n = 10;
    
    // Tính box radius (tương tự như CPU version)
    float wi = sqrtf((12.0f * sigma * sigma / n) + 1.0f);
    int wl = (int)wi;
    if (wl % 2 == 0) wl--;
    int wu = wl + 2;
    float mi = (12.0f * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4.0f * wl - 4.0f);
    int m = (int)(mi + 0.5f);
    
    for (int i = 0; i < n; i++) {
        boxes[i] = ((i < m ? wl : wu) - 1) / 2;
    }
    
    int size = w * h * c;
    
    // Sử dụng local pointers để swap
    unsigned char* current_in = d_in;
    unsigned char* current_out = d_out;
    
    // BƯỚC 1: N lần horizontal blur
    for (int i = 0; i < n; i++) {
        horizontal_blur_cuda_extend(current_in, current_out, w, h, c, boxes[i]);
        // Swap pointers
        unsigned char* temp = current_in;
        current_in = current_out;
        current_out = temp;
    }
    
    // BƯỚC 2: Transpose
    flip_block_cuda(current_in, current_out, w, h, c);
    unsigned char* temp = current_in;
    current_in = current_out;
    current_out = temp;
    
    // BƯỚC 3: N lần horizontal blur trên ảnh đã transpose (thực chất là vertical blur)
    for (int i = 0; i < n; i++) {
        horizontal_blur_cuda_extend(current_in, current_out, h, w, c, boxes[i]);  // w và h đã đổi chỗ
        temp = current_in;
        current_in = current_out;
        current_out = temp;
    }
    
    // BƯỚC 4: Transpose lại
    flip_block_cuda(current_in, current_out, h, w, c);
    
    // Xác định buffer nào chứa kết quả cuối cùng
    // Sau transpose cuối cùng, kết quả nằm trong current_out
    // Nhưng do các lần swap, current_out có thể không phải là d_out
    unsigned char* final_result = current_out;
    
    // Đảm bảo kết quả cuối cùng nằm trong d_out (copy nếu cần)
    if (final_result != d_out) {
        cudaMemcpy(d_out, final_result, size, cudaMemcpyDeviceToDevice);
    }
}

extern "C" unsigned char* cuda_alloc_image(int size)
{
    unsigned char* d_ptr;
    cudaError_t err = cudaMalloc(&d_ptr, size);
    if (err != cudaSuccess) {
        printf("CUDA malloc error: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return d_ptr;
}

extern "C" void cuda_free_image(unsigned char* d_ptr)
{
    if (d_ptr) {
        cudaFree(d_ptr);
    }
}

extern "C" void cuda_copy_to_device(unsigned char* d_dst, const unsigned char* h_src, int size)
{
    cudaError_t err = cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA memcpy H2D error: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void cuda_copy_to_host(unsigned char* h_dst, const unsigned char* d_src, int size)
{
    cudaError_t err = cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA memcpy D2H error: %s\n", cudaGetErrorString(err));
    }
}

