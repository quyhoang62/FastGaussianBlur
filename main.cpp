// ================================================================
// CHƯƠNG TRÌNH DEMO: FAST GAUSSIAN BLUR VỚI SONG SONG HÓA
// ================================================================
// Chương trình này minh họa việc sử dụng thuật toán Fast Gaussian Blur
// được tối ưu với song song hóa (parallelization) sử dụng OpenMP và CUDA.
//
// Chương trình sẽ chạy và so sánh ba phiên bản:
// - Sequential (không song song hóa)
// - OpenMP (song song hóa trên CPU)
// - CUDA (song song hóa trên GPU)
//
// ================================================================

#include <iostream>
#include <chrono>  // Để đo thời gian thực thi
#include <iomanip>  // Để format output
#include <string>
#include <cmath>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

// ================================================================
// THƯ VIỆN XỬ LÝ ẢNH
// ================================================================
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"               // Thư viện dùng để load (đọc) ảnh từ file (header-only)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"         // Thư viện dùng để lưu ảnh ra file (header-only)

// ================================================================
// THUẬT TOÁN FAST GAUSSIAN BLUR
// ================================================================
// Include header với OpenMP bật (mặc định)
#define USE_OPENMP 1
#include "fast_gaussian_blur_template.h"

// Include CUDA version nếu có
#ifdef USE_CUDA
#include "fast_gaussian_blur_cuda.h"
#include <cuda_runtime.h>
#endif

typedef unsigned char uchar;         // Đặt alias uchar = unsigned char (giá trị 0–255)

// ================================================================
// HÀM HIỂN THỊ THỜI GIAN CHI TIẾT
// ================================================================
void print_detailed_time(const std::string& label, 
                         const std::chrono::high_resolution_clock::time_point& start,
                         const std::chrono::high_resolution_clock::time_point& end) {
    auto duration = end - start;
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    
    // Tính toán với độ chính xác cao
    double ms = milliseconds + (microseconds % 1000) / 1000.0;
    double us = microseconds + (nanoseconds % 1000) / 1000.0;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  " << std::setw(35) << std::left << label << ": ";
    std::cout << std::setw(12) << std::right << ms << " ms  ";
    std::cout << "(" << std::setw(12) << std::right << us << " µs)";
    std::cout << std::endl;
}

// ================================================================
// HÀM HIỂN THỊ KẾT QUẢ SO SÁNH
// ================================================================
void print_comparison_table(double time_sequential, double time_omp, double time_cuda = -1.0) {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              KẾT QUẢ SO SÁNH HIỆU NĂNG                                ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Phiên bản                    │  Thời gian (ms)  │  Tốc độ tăng     ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════════════╣\n";
    
    std::cout << std::fixed << std::setprecision(3);
    
    // Tìm thời gian nhanh nhất để tính speedup
    double fastest = time_sequential;
    if (time_omp > 0 && time_omp < fastest) fastest = time_omp;
    if (time_cuda > 0 && time_cuda < fastest) fastest = time_cuda;
    
    std::cout << "║  " << std::setw(27) << std::left << "Sequential (Single-thread)" 
              << "│  " << std::setw(15) << std::right << time_sequential << "  │  " 
              << std::setw(15) << std::right << (time_sequential / fastest) << "x" << "  ║\n";
    
    if (time_omp > 0) {
        std::cout << "║  " << std::setw(27) << std::left << "OpenMP (Multi-thread CPU)" 
                  << "│  " << std::setw(15) << std::right << time_omp << "  │  " 
                  << std::setw(15) << std::right << (time_omp / fastest) << "x" << "  ║\n";
    }
    
    if (time_cuda > 0) {
        std::cout << "║  " << std::setw(27) << std::left << "CUDA (GPU)" 
                  << "│  " << std::setw(15) << std::right << time_cuda << "  │  " 
                  << std::setw(15) << std::right << (time_cuda / fastest) << "x" << "  ║\n";
    }
    
    std::cout << "╠═══════════════════════════════════════════════════════════════════════╣\n";
    
    if (time_omp > 0) {
        double speedup_omp = time_sequential / time_omp;
        double improvement_omp = ((time_sequential - time_omp) / time_sequential) * 100.0;
        std::cout << "║  OpenMP vs Sequential: " << std::setw(8) << std::right << speedup_omp << "x speedup, "
                  << std::setw(6) << std::right << improvement_omp << "% faster  ║\n";
    }
    
    if (time_cuda > 0) {
        double speedup_cuda = time_sequential / time_cuda;
        double improvement_cuda = ((time_sequential - time_cuda) / time_sequential) * 100.0;
        std::cout << "║  CUDA vs Sequential: " << std::setw(9) << std::right << speedup_cuda << "x speedup, "
                  << std::setw(6) << std::right << improvement_cuda << "% faster  ║\n";
    }
    
    if (time_omp > 0 && time_cuda > 0) {
        double speedup_cuda_vs_omp = time_omp / time_cuda;
        std::cout << "║  CUDA vs OpenMP: " << std::setw(12) << std::right << speedup_cuda_vs_omp << "x speedup  ║\n";
    }
    
    std::cout << "╚═══════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

// ================================================================
// HÀM CHÍNH
// ================================================================
int main(int argc, char * argv[])
{   
    // Kiểm tra số lượng tham số truyền vào
    if( argc < 4 )
    {
        // In hướng dẫn sử dụng
        printf("%s [input] [output] [sigma] [order - optional] [border - optional]\n", argv[0]);
        printf("\n");
        printf("- input:  file ảnh input (jpg/png/bmp/...)\n");
        printf("- output: file ảnh output muốn lưu (.png/.jpg/.bmp)\n");
        printf("- sigma:  độ mờ Gaussian (float, > 0)\n");
        printf("- order:  số lần blur (bộ lọc box đa cấp), mặc định = 3\n");
        printf("- border: cách xử lý biên ảnh [mirror, extend, crop, wrap]\n");
        printf("\n");
        exit(1);                     // Thoát chương trình vì thiếu tham số
    }

    // =====================
    // 1) LOAD ẢNH
    // =====================
    int width, height, channels;

    // stbi_load đọc file ảnh và trả về mảng pixel 1D (uchar*)
    uchar * image_data = stbi_load(argv[1], &width, &height, &channels, 0);

    if (!image_data) {
        printf("Lỗi: Không thể load ảnh từ file %s\n", argv[1]);
        exit(1);
    }

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║          FAST GAUSSIAN BLUR - SO SÁNH HIỆU NĂNG                       ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("Source image: %s\n", argv[1]);
    printf("Kích thước: %dx%d pixels (%d channels)\n", width, height, channels);
    printf("Tổng số pixels: %d\n", width * height);
    printf("Tổng kích thước: %.2f MB\n", (width * height * channels) / (1024.0 * 1024.0));
    
#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    printf("OpenMP: Có sẵn (Max threads: %d)\n", max_threads);
#else
    printf("OpenMP: Không có sẵn (sẽ chạy single-threaded)\n");
#endif
    printf("\n");

    // =====================
    // 2) ĐỌC THAM SỐ
    // =====================

    const float sigma = std::atof(argv[3]);   // Độ mờ Gaussian
    const int passes = argc > 4 ? std::atoi(argv[4]) : 3;  // Số lần blur (mặc định 3)

    const std::string policy = argc > 5
                                ? std::string(argv[5])
                                : "mirror";   // Cách xử lý biên

    Border border;
    if (policy == "mirror")         border = Border::kMirror;
    else if (policy == "extend")    border = Border::kExtend;
    else if (policy == "crop")      border = Border::kKernelCrop;
    else if (policy == "wrap")      border = Border::kWrap;
    else                            border = Border::kMirror; // Default

    printf("Tham số xử lý:\n");
    printf("  - Sigma: %.2f\n", sigma);
    printf("  - Passes: %d\n", passes);
    printf("  - Border policy: %s\n", policy.c_str());
    printf("\n");

    // =====================
    // 3) TẠO BỘ ĐỆM (BUFFER) CHO CẢ BA PHIÊN BẢN
    // =====================

    std::size_t size = width * height * channels; // số phần tử pixel tổng cộng

    // Buffer cho phiên bản Sequential (single-threaded)
    uchar * new_image_seq = new uchar[size];
    uchar * old_image_seq = new uchar[size];
    
    // Buffer cho phiên bản OpenMP (multi-threaded CPU)
    uchar * new_image_omp = new uchar[size];
    uchar * old_image_omp = new uchar[size];
    
    // Buffer cho phiên bản CUDA (GPU)
    uchar * d_in_cuda = nullptr;
    uchar * d_out_cuda = nullptr;
    uchar * h_result_cuda = nullptr;  // Host buffer để lưu kết quả
    
#ifdef USE_CUDA
    // Kiểm tra CUDA có sẵn không
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    bool cuda_available = (err == cudaSuccess && deviceCount > 0);
    
    if (cuda_available) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("CUDA: Có sẵn (Device: %s, Compute Capability: %d.%d)\n", 
               prop.name, prop.major, prop.minor);
        d_in_cuda = cuda_alloc_image(size);
        d_out_cuda = cuda_alloc_image(size);
        h_result_cuda = new uchar[size];
        if (!d_in_cuda || !d_out_cuda || !h_result_cuda) {
            printf("CUDA: Không thể cấp phát bộ nhớ, bỏ qua phiên bản CUDA\n");
            if (d_in_cuda) cuda_free_image(d_in_cuda);
            if (d_out_cuda) cuda_free_image(d_out_cuda);
            if (h_result_cuda) delete[] h_result_cuda;
            cuda_available = false;
        }
    } else {
        printf("CUDA: Không có sẵn hoặc không có GPU\n");
    }
    printf("\n");
#else
    bool cuda_available = false;
    printf("CUDA: Không được compile (define USE_CUDA để bật)\n");
    printf("\n");
#endif

    // =====================
    // 4) COPY DỮ LIỆU ẢNH VÀO BUFFER
    // =====================

    for(std::size_t i = 0; i < size; ++i)
    {
        old_image_seq[i] = image_data[i];
        old_image_omp[i] = image_data[i];
    }
    
#ifdef USE_CUDA
    if (cuda_available) {
        cuda_copy_to_device(d_in_cuda, image_data, size);
    }
#endif

    double time_seq_ms = 0.0;
    double time_omp_ms = 0.0;
    double time_cuda_ms = 0.0;

    // =====================
    // 5) CHẠY PHIÊN BẢN SEQUENTIAL (SINGLE-THREADED)
    // =====================
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  PHIÊN BẢN SEQUENTIAL (Single-threaded - Không song song hóa)       ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");
    
    // Đặt số threads = 1 để mô phỏng sequential (vẫn có overhead nhỏ của OpenMP)
    #ifdef _OPENMP
    int old_num_threads = omp_get_max_threads();
    omp_set_num_threads(1);
    printf("Số threads: 1 (single-threaded, OpenMP disabled)\n");
    #else
    printf("Số threads: 1 (single-threaded, không có OpenMP)\n");
    #endif
    printf("\n");
    
    auto start_seq = std::chrono::high_resolution_clock::now();
    
    fast_gaussian_blur(old_image_seq, new_image_seq,
                       width, height, channels,
                       sigma, passes, border);
    
    auto end_seq = std::chrono::high_resolution_clock::now();
    time_seq_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_seq - start_seq).count() / 1000.0;
    
    print_detailed_time("Tổng thời gian xử lý", start_seq, end_seq);
    printf("\n");
    
    // Khôi phục số threads
    #ifdef _OPENMP
    omp_set_num_threads(old_num_threads);
    #endif

    // =====================
    // 6) CHẠY PHIÊN BẢN OPENMP (MULTI-THREADED CPU)
    // =====================
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  PHIÊN BẢN OPENMP (Song song hóa trên CPU - Multi-threaded)          ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");
    
#ifdef _OPENMP
    // Đảm bảo sử dụng tất cả threads có sẵn
    omp_set_num_threads(omp_get_max_threads());
    printf("Số threads: %d\n", omp_get_max_threads());
#else
    printf("OpenMP không có sẵn, chạy single-threaded\n");
#endif
    printf("\n");
    
    auto start_omp = std::chrono::high_resolution_clock::now();
    
    fast_gaussian_blur(old_image_omp, new_image_omp,
                       width, height, channels,
                       sigma, passes, border);
    
    auto end_omp = std::chrono::high_resolution_clock::now();
    time_omp_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_omp - start_omp).count() / 1000.0;
    
    print_detailed_time("Tổng thời gian xử lý", start_omp, end_omp);
    printf("\n");

    // =====================
    // 7) CHẠY PHIÊN BẢN CUDA (GPU)
    // =====================
#ifdef USE_CUDA
    if (cuda_available) {
        printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
        printf("║  PHIÊN BẢN CUDA (Song song hóa trên GPU)                             ║\n");
        printf("╚═══════════════════════════════════════════════════════════════════════╝\n");
        printf("\n");
        
        auto start_cuda = std::chrono::high_resolution_clock::now();
        
        // Copy input lên GPU (đảm bảo dữ liệu mới nhất)
        cuda_copy_to_device(d_in_cuda, image_data, size);
        
        // Thực hiện blur trên GPU
        fast_gaussian_blur_cuda(d_in_cuda, d_out_cuda,
                                width, height, channels,
                                sigma, passes, border);
        
        // Đồng bộ để đảm bảo tất cả operations hoàn thành trước khi copy
        cudaDeviceSynchronize();
        
        // Kết quả cuối cùng nằm trong d_out_cuda (sau các lần swap)
        // Copy về CPU
        cuda_copy_to_host(h_result_cuda, d_out_cuda, size);
        
        auto end_cuda = std::chrono::high_resolution_clock::now();
        time_cuda_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_cuda - start_cuda).count() / 1000.0;
        
        print_detailed_time("Tổng thời gian xử lý (bao gồm transfer)", start_cuda, end_cuda);
        printf("\n");
    }
#endif

    // =====================
    // 8) HIỂN THỊ KẾT QUẢ SO SÁNH
    // =====================
    print_comparison_table(time_seq_ms, time_omp_ms, 
                          cuda_available ? time_cuda_ms : -1.0);

    // =====================
    // 9) COPY KẾT QUẢ BLUR VỀ image_data ĐỂ LƯU FILE
    // (Sử dụng kết quả từ phiên bản OpenMP)
    // =====================

    for(std::size_t i = 0; i < size; ++i)
    {
        image_data[i] = (uchar)(new_image_omp[i]);
    }

    // =====================
    // 9) LƯU ẢNH RA FILE
    // =====================
    std::string file(argv[2]);
    std::string ext = file.substr(file.size()-3);  // Lấy phần .png/.jpg...

    if( ext == "bmp" )
        stbi_write_bmp(argv[2], width, height, channels, image_data);
    else if( ext == "jpg" )
        stbi_write_jpg(argv[2], width, height, channels, image_data, 90); // chất lượng 90%
    else
    {
        // Nếu không phải png thì chuyển về png
        if( ext != "png" )
        {
            printf("Image format '%s' not supported, writing default png\n",
                   ext.c_str()); 
            file = file.substr(0, file.size()-4) + std::string(".png");
        }
        stbi_write_png(file.c_str(), width, height, channels,
                       image_data, channels * width); // stride = width*channels
    }
    
    printf("Đã lưu ảnh kết quả vào: %s\n", argv[2]);
    printf("\n");

    // =====================
    // 10) GIẢI PHÓNG BỘ NHỚ
    // =====================

    stbi_image_free(image_data);   // Giải phóng ảnh load từ file
    delete[] new_image_seq;         // Giải phóng buffer kết quả sequential
    delete[] old_image_seq;          // Giải phóng buffer input sequential
    delete[] new_image_omp;          // Giải phóng buffer kết quả OpenMP
    delete[] old_image_omp;          // Giải phóng buffer input OpenMP
    
#ifdef USE_CUDA
    if (cuda_available) {
        cuda_free_image(d_in_cuda);
        cuda_free_image(d_out_cuda);
        delete[] h_result_cuda;
    }
#endif

    return 0;
}
