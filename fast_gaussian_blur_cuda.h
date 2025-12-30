// ================================================================
// CUDA VERSION: FAST GAUSSIAN BLUR VỚI SONG SONG HÓA CUDA
// ================================================================
// Header file chứa khai báo các hàm CUDA cho Fast Gaussian Blur
// Dựa trên thuật toán OpenMP nhưng được triển khai trên GPU với CUDA
//
#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

#include "fast_gaussian_blur_template.h"

// ================================================================
// KHAI BÁO CÁC HÀM CUDA
// ================================================================

//! \brief Hàm CUDA để thực hiện horizontal blur với chính sách extend
//! Mỗi thread block xử lý một hàng của ảnh
//! \param[in] d_in      Buffer ảnh nguồn trên GPU (device memory)
//! \param[out] d_out    Buffer ảnh đích trên GPU (device memory)
//! \param[in] w         Chiều rộng ảnh
//! \param[in] h         Chiều cao ảnh
//! \param[in] c         Số kênh màu (channels)
//! \param[in] r         Bán kính box blur
extern "C" void horizontal_blur_cuda_extend(
    unsigned char* d_in,
    unsigned char* d_out,
    int w, int h, int c, int r
);

//! \brief Hàm CUDA để thực hiện transpose (chuyển vị) ảnh
//! Sử dụng shared memory để tối ưu cache
//! \param[in] d_in      Buffer ảnh nguồn trên GPU
//! \param[out] d_out    Buffer ảnh đích trên GPU
//! \param[in] w         Chiều rộng ảnh gốc (sẽ thành chiều cao sau transpose)
//! \param[in] h         Chiều cao ảnh gốc (sẽ thành chiều rộng sau transpose)
//! \param[in] c         Số kênh màu
extern "C" void flip_block_cuda(
    unsigned char* d_in,
    unsigned char* d_out,
    int w, int h, int c
);

//! \brief Hàm chính CUDA để thực hiện Fast Gaussian Blur
//! Tương tự như phiên bản CPU nhưng chạy trên GPU
//! \param[in,out] d_in      Buffer nguồn trên GPU (sẽ được swap với d_out)
//! \param[in,out] d_out     Buffer đích trên GPU (sẽ được swap với d_in)
//! \param[in] w             Chiều rộng ảnh
//! \param[in] h             Chiều cao ảnh
//! \param[in] c             Số kênh màu
//! \param[in] sigma         Độ lệch chuẩn Gaussian
//! \param[in] n             Số lần box blur passes
//! \param[in] border        Chính sách xử lý biên (hiện tại chỉ hỗ trợ kExtend)
//! \note Sau khi gọi hàm, d_in và d_out có thể đã bị swap
extern "C" void fast_gaussian_blur_cuda(
    unsigned char* d_in,
    unsigned char* d_out,
    int w, int h, int c,
    float sigma, int n,
    Border border
);

//! \brief Hàm helper để cấp phát bộ nhớ GPU
extern "C" unsigned char* cuda_alloc_image(int size);

//! \brief Hàm helper để giải phóng bộ nhớ GPU
extern "C" void cuda_free_image(unsigned char* d_ptr);

//! \brief Hàm helper để copy dữ liệu từ CPU sang GPU
extern "C" void cuda_copy_to_device(unsigned char* d_dst, const unsigned char* h_src, int size);

//! \brief Hàm helper để copy dữ liệu từ GPU sang CPU
extern "C" void cuda_copy_to_host(unsigned char* h_dst, const unsigned char* d_src, int size);

