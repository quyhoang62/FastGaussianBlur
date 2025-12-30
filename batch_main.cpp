// ================================================================
// BATCH PROCESSING: FAST GAUSSIAN BLUR VỚI SONG SONG HÓA
// ================================================================
// Chương trình này xử lý nhiều ảnh cùng lúc (batch processing)
// và so sánh hiệu năng giữa 3 phiên bản: Sequential, OpenMP, CUDA
//
// Usage: batch_main.exe [input_folder] [output_folder] [sigma] [passes] [border]
//

#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <thread>
#include <future>

#ifdef _OPENMP
#include <omp.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define USE_OPENMP 1
#include "fast_gaussian_blur_template.h"

#ifdef USE_CUDA
#include "fast_gaussian_blur_cuda.h"
#include <cuda_runtime.h>
#endif

typedef unsigned char uchar;

namespace fs = std::filesystem;

// ================================================================
// HÀM ĐỌC TẤT CẢ ẢNH TRONG FOLDER
// ================================================================
std::vector<std::string> get_image_files(const std::string& folder_path) {
    std::vector<std::string> image_files;
    
    if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) {
        std::cerr << "Error: Folder not found: " << folder_path << std::endl;
        return image_files;
    }
    
    // Các extension được hỗ trợ
    std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tga"};
    
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            
            if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                image_files.push_back(entry.path().string());
            }
        }
    }
    
    std::sort(image_files.begin(), image_files.end());
    return image_files;
}

// ================================================================
// HÀM XỬ LÝ MỘT ẢNH (SEQUENTIAL)
// ================================================================
void process_image_sequential(
    const std::string& input_path,
    const std::string& output_path,
    float sigma, int passes, Border border) {
    
    int width, height, channels;
    uchar* image_data = stbi_load(input_path.c_str(), &width, &height, &channels, 0);
    
    if (!image_data) {
        std::cerr << "Error loading: " << input_path << std::endl;
        return;
    }
    
    size_t size = width * height * channels;
    uchar* old_image = new uchar[size];
    uchar* new_image = new uchar[size];
    
    for (size_t i = 0; i < size; ++i) {
        old_image[i] = image_data[i];
    }
    
    // Tắt OpenMP cho sequential
    #ifdef _OPENMP
    int old_threads = omp_get_max_threads();
    omp_set_num_threads(1);
    #endif
    
    fast_gaussian_blur(old_image, new_image, width, height, channels, sigma, passes, border);
    
    #ifdef _OPENMP
    omp_set_num_threads(old_threads);
    #endif
    
    // Copy kết quả
    for (size_t i = 0; i < size; ++i) {
        image_data[i] = new_image[i];
    }
    
    // Lưu ảnh
    std::string ext = fs::path(output_path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == ".bmp") {
        stbi_write_bmp(output_path.c_str(), width, height, channels, image_data);
    } else if (ext == ".jpg" || ext == ".jpeg") {
        stbi_write_jpg(output_path.c_str(), width, height, channels, image_data, 90);
    } else {
        stbi_write_png(output_path.c_str(), width, height, channels, image_data, width * channels);
    }
    
    stbi_image_free(image_data);
    delete[] old_image;
    delete[] new_image;
}

// ================================================================
// HÀM XỬ LÝ MỘT ẢNH (OPENMP)
// ================================================================
void process_image_omp(
    const std::string& input_path,
    const std::string& output_path,
    float sigma, int passes, Border border) {
    
    int width, height, channels;
    uchar* image_data = stbi_load(input_path.c_str(), &width, &height, &channels, 0);
    
    if (!image_data) {
        std::cerr << "Error loading: " << input_path << std::endl;
        return;
    }
    
    size_t size = width * height * channels;
    uchar* old_image = new uchar[size];
    uchar* new_image = new uchar[size];
    
    for (size_t i = 0; i < size; ++i) {
        old_image[i] = image_data[i];
    }
    
    // Tắt OpenMP bên trong - chỉ dùng OpenMP ở level batch
    // Tạm thời disable OpenMP bằng cách set threads = 1
    #ifdef _OPENMP
    int old_threads = omp_get_max_threads();
    omp_set_num_threads(1);
    #endif
    
    fast_gaussian_blur(old_image, new_image, width, height, channels, sigma, passes, border);
    
    #ifdef _OPENMP
    omp_set_num_threads(old_threads);
    #endif
    
    // Copy kết quả
    for (size_t i = 0; i < size; ++i) {
        image_data[i] = new_image[i];
    }
    
    // Lưu ảnh
    std::string ext = fs::path(output_path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == ".bmp") {
        stbi_write_bmp(output_path.c_str(), width, height, channels, image_data);
    } else if (ext == ".jpg" || ext == ".jpeg") {
        stbi_write_jpg(output_path.c_str(), width, height, channels, image_data, 90);
    } else {
        stbi_write_png(output_path.c_str(), width, height, channels, image_data, width * channels);
    }
    
    stbi_image_free(image_data);
    delete[] old_image;
    delete[] new_image;
}

// ================================================================
// HÀM XỬ LÝ MỘT ẢNH (CUDA)
// ================================================================
#ifdef USE_CUDA
void process_image_cuda(
    const std::string& input_path,
    const std::string& output_path,
    float sigma, int passes, Border border) {
    
    int width, height, channels;
    uchar* image_data = stbi_load(input_path.c_str(), &width, &height, &channels, 0);
    
    if (!image_data) {
        std::cerr << "Error loading: " << input_path << std::endl;
        return;
    }
    
    size_t size = width * height * channels;
    
    // Allocate GPU memory
    uchar* d_in = cuda_alloc_image(size);
    uchar* d_out = cuda_alloc_image(size);
    uchar* h_result = new uchar[size];
    
    if (!d_in || !d_out || !h_result) {
        std::cerr << "Error allocating GPU memory for: " << input_path << std::endl;
        stbi_image_free(image_data);
        if (d_in) cuda_free_image(d_in);
        if (d_out) cuda_free_image(d_out);
        if (h_result) delete[] h_result;
        return;
    }
    
    // Copy to GPU
    cuda_copy_to_device(d_in, image_data, size);
    
    // Process on GPU
    fast_gaussian_blur_cuda(d_in, d_out, width, height, channels, sigma, passes, border);
    
    // Synchronize
    cudaDeviceSynchronize();
    
    // Copy back
    cuda_copy_to_host(h_result, d_out, size);
    
    // Lưu ảnh
    std::string ext = fs::path(output_path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == ".bmp") {
        stbi_write_bmp(output_path.c_str(), width, height, channels, h_result);
    } else if (ext == ".jpg" || ext == ".jpeg") {
        stbi_write_jpg(output_path.c_str(), width, height, channels, h_result, 90);
    } else {
        stbi_write_png(output_path.c_str(), width, height, channels, h_result, width * channels);
    }
    
    stbi_image_free(image_data);
    cuda_free_image(d_in);
    cuda_free_image(d_out);
    delete[] h_result;
}
#endif

// ================================================================
// BATCH PROCESSING VỚI PARALLELIZATION
// ================================================================
template<typename ProcessFunc>
double batch_process_parallel(
    const std::vector<std::string>& input_files,
    const std::string& output_folder,
    ProcessFunc process_func,
    int num_threads = 0,
    bool use_omp_inner = false) {
    
    // Tạo output folder nếu chưa có
    fs::create_directories(output_folder);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    #ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    // Tắt nested parallelism nếu không dùng OpenMP bên trong
    if (!use_omp_inner) {
        omp_set_nested(0);
    }
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (size_t i = 0; i < input_files.size(); ++i) {
        std::string input_path = input_files[i];
        std::string filename = fs::path(input_path).filename().string();
        std::string output_path = (fs::path(output_folder) / filename).string();
        
        process_func(input_path, output_path);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    return duration.count();
}

// ================================================================
// MAIN FUNCTION
// ================================================================
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " [input_folder] [output_folder] [sigma] [passes] [border]" << std::endl;
        std::cout << "\n";
        std::cout << "  input_folder:  Folder chứa ảnh input" << std::endl;
        std::cout << "  output_folder: Folder để lưu ảnh output" << std::endl;
        std::cout << "  sigma:         Độ mờ Gaussian (float, > 0)" << std::endl;
        std::cout << "  passes:        Số lần blur passes (mặc định = 3)" << std::endl;
        std::cout << "  border:        Border policy [mirror, extend, crop, wrap] (mặc định = extend)" << std::endl;
        return 1;
    }
    
    std::string input_folder = argv[1];
    std::string output_folder = argv[2];
    float sigma = std::atof(argv[3]);
    int passes = argc > 4 ? std::atoi(argv[4]) : 3;
    
    std::string border_str = argc > 5 ? std::string(argv[5]) : "extend";
    Border border = kExtend;
    if (border_str == "mirror") border = kMirror;
    else if (border_str == "extend") border = kExtend;
    else if (border_str == "crop") border = kKernelCrop;
    else if (border_str == "wrap") border = kWrap;
    
    // Đọc danh sách ảnh
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║          BATCH PROCESSING - FAST GAUSSIAN BLUR                      ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    
    std::vector<std::string> image_files = get_image_files(input_folder);
    
    if (image_files.empty()) {
        std::cerr << "Error: No image files found in " << input_folder << std::endl;
        return 1;
    }
    
    std::cout << "Input folder: " << input_folder << std::endl;
    std::cout << "Output folder: " << output_folder << std::endl;
    std::cout << "Number of images: " << image_files.size() << std::endl;
    std::cout << "Sigma: " << sigma << std::endl;
    std::cout << "Passes: " << passes << std::endl;
    std::cout << "Border: " << border_str << std::endl;
    std::cout << "\n";
    
    #ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    std::cout << "OpenMP: Available (Max threads: " << max_threads << ")\n";
    #else
    std::cout << "OpenMP: Not available\n";
    #endif
    
    #ifdef USE_CUDA
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    bool cuda_available = (err == cudaSuccess && deviceCount > 0);
    if (cuda_available) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "CUDA: Available (Device: " << prop.name << ", Compute Capability: " 
                  << prop.major << "." << prop.minor << ")\n";
    } else {
        std::cout << "CUDA: Not available\n";
    }
    #else
    bool cuda_available = false;
    std::cout << "CUDA: Not compiled\n";
    #endif
    
    std::cout << "\n";
    
    double time_seq = 0.0, time_omp = 0.0, time_cuda = 0.0;
    
    // =====================
    // SEQUENTIAL BATCH
    // =====================
    std::cout << "╔═══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  PHIÊN BẢN SEQUENTIAL (Single-threaded)                              ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "Processing " << image_files.size() << " images...\n";
    
    auto process_seq = [&](const std::string& in, const std::string& out) {
        process_image_sequential(in, out, sigma, passes, border);
    };
    
    time_seq = batch_process_parallel(image_files, output_folder + "/sequential", process_seq, 1, false);
    
    std::cout << "  Total time: " << std::fixed << std::setprecision(3) << time_seq << " ms\n";
    std::cout << "  Average per image: " << (time_seq / image_files.size()) << " ms\n";
    std::cout << "\n";
    
    // =====================
    // OPENMP BATCH
    // =====================
    std::cout << "╔═══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  PHIÊN BẢN OPENMP (Multi-threaded CPU)                                ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "Processing " << image_files.size() << " images in parallel...\n";
    
    auto process_omp = [&](const std::string& in, const std::string& out) {
        process_image_omp(in, out, sigma, passes, border);
    };
    
    // Với OpenMP batch, có thể dùng nested parallelism (mỗi thread xử lý 1 ảnh với OpenMP bên trong)
    // Hoặc tắt nested và chỉ parallelize ở level batch
    time_omp = batch_process_parallel(image_files, output_folder + "/openmp", process_omp, 0, true);
    
    std::cout << "  Total time: " << std::fixed << std::setprecision(3) << time_omp << " ms\n";
    std::cout << "  Average per image: " << (time_omp / image_files.size()) << " ms\n";
    std::cout << "\n";
    
    // =====================
    // CUDA BATCH
    // =====================
    #ifdef USE_CUDA
    if (cuda_available) {
        std::cout << "╔═══════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  PHIÊN BẢN CUDA (GPU)                                                ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════════════╝\n";
        std::cout << "Processing " << image_files.size() << " images on GPU...\n";
        
        auto process_cuda = [&](const std::string& in, const std::string& out) {
            process_image_cuda(in, out, sigma, passes, border);
        };
        
        time_cuda = batch_process_parallel(image_files, output_folder + "/cuda", process_cuda, 0);
        
        std::cout << "  Total time: " << std::fixed << std::setprecision(3) << time_cuda << " ms\n";
        std::cout << "  Average per image: " << (time_cuda / image_files.size()) << " ms\n";
        std::cout << "\n";
    }
    #endif
    
    // =====================
    // SO SÁNH KẾT QUẢ
    // =====================
    std::cout << "╔═══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              KẾT QUẢ SO SÁNH HIỆU NĂNG                                ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Phiên bản                    │  Tổng thời gian (ms)  │  Tốc độ tăng ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════════════╣\n";
    
    double fastest = time_seq;
    if (time_omp > 0 && time_omp < fastest) fastest = time_omp;
    if (time_cuda > 0 && time_cuda < fastest) fastest = time_cuda;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "║  " << std::setw(27) << std::left << "Sequential (Single-thread)" 
              << "│  " << std::setw(20) << std::right << time_seq << "  │  " 
              << std::setw(12) << std::right << (time_seq / fastest) << "x  ║\n";
    
    if (time_omp > 0) {
        std::cout << "║  " << std::setw(27) << std::left << "OpenMP (Multi-thread CPU)" 
                  << "│  " << std::setw(20) << std::right << time_omp << "  │  " 
                  << std::setw(12) << std::right << (time_omp / fastest) << "x  ║\n";
    }
    
    if (time_cuda > 0) {
        std::cout << "║  " << std::setw(27) << std::left << "CUDA (GPU)" 
                  << "│  " << std::setw(20) << std::right << time_cuda << "  │  " 
                  << std::setw(12) << std::right << (time_cuda / fastest) << "x  ║\n";
    }
    
    std::cout << "╠═══════════════════════════════════════════════════════════════════════╣\n";
    
    if (time_omp > 0) {
        double speedup_omp = time_seq / time_omp;
        double improvement_omp = ((time_seq - time_omp) / time_seq) * 100.0;
        std::cout << "║  OpenMP vs Sequential: " << std::setw(8) << std::right << speedup_omp << "x speedup, "
                  << std::setw(6) << std::right << improvement_omp << "% faster  ║\n";
    }
    
    if (time_cuda > 0) {
        double speedup_cuda = time_seq / time_cuda;
        double improvement_cuda = ((time_seq - time_cuda) / time_seq) * 100.0;
        std::cout << "║  CUDA vs Sequential: " << std::setw(9) << std::right << speedup_cuda << "x speedup, "
                  << std::setw(6) << std::right << improvement_cuda << "% faster  ║\n";
    }
    
    if (time_omp > 0 && time_cuda > 0) {
        double speedup_cuda_vs_omp = time_omp / time_cuda;
        std::cout << "║  CUDA vs OpenMP: " << std::setw(12) << std::right << speedup_cuda_vs_omp << "x speedup  ║\n";
    }
    
    std::cout << "╚═══════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    
    std::cout << "Output folders:\n";
    std::cout << "  - " << output_folder << "/sequential\n";
    std::cout << "  - " << output_folder << "/openmp\n";
    if (time_cuda > 0) {
        std::cout << "  - " << output_folder << "/cuda\n";
    }
    std::cout << "\n";
    
    return 0;
}

