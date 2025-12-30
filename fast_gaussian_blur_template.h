// Copyright (C) 2017-2022 Basile Fraboni
// Copyright (C) 2014 Ivan Kutskir (for the original fast blur implementation)
// All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the MIT license. For further details please refer 
// to : https://mit-license.org/
//
#pragma once

// ================================================================
// MACRO ĐIỀU KHIỂN OPENMP
// ================================================================
// Định nghĩa macro này trước khi include header để bật/tắt OpenMP
// Mặc định: nếu không định nghĩa, sẽ sử dụng OpenMP nếu có sẵn
#ifndef USE_OPENMP
    #ifdef _OPENMP
        #define USE_OPENMP 1
    #else
        #define USE_OPENMP 0
    #endif
#endif

// Macro helper để điều kiện hóa pragma OpenMP
#if USE_OPENMP
    #define OMP_PARALLEL_FOR _Pragma("omp parallel for")
    #define OMP_PARALLEL_FOR_COLLAPSE_2 _Pragma("omp parallel for collapse(2)")
#else
    #define OMP_PARALLEL_FOR
    #define OMP_PARALLEL_FOR_COLLAPSE_2
#endif

// ================================================================
// TỔNG QUAN VỀ SONG SONG HÓA (PARALLELIZATION) TRONG CODE NÀY
// ================================================================
//
// Code này sử dụng OpenMP (Open Multi-Processing) để song song hóa
// các phần tính toán tốn kém nhất của thuật toán Fast Gaussian Blur.
//
// CÁC ĐIỂM SONG SONG HÓA CHÍNH:
// 
// 1. HORIZONTAL BLUR (Blur theo chiều ngang):
//    - Vị trí: Trong các hàm horizontal_blur_* (extend, mirror, crop, wrap)
//    - Cách thức: #pragma omp parallel for trước vòng lặp for(int i=0; i<h; i++)
//    - Chi tiết: Mỗi hàng (row) của ảnh được xử lý độc lập bởi một thread riêng
//    - Lợi ích: 
//      * Mỗi hàng có thể xử lý độc lập, không có dependency
//      * Dữ liệu mỗi hàng nằm liên tiếp trong bộ nhớ (cache-friendly)
//      * Tự động load balancing giữa các threads
//      * Tăng tốc tuyến tính với số lượng CPU cores
//
// 2. TRANSPOSE/FLIP BLOCK (Chuyển vị ảnh theo block):
//    - Vị trí: Trong hàm flip_block
//    - Cách thức: #pragma omp parallel for collapse(2) cho 2 vòng lặp lồng nhau
//    - Chi tiết: 
//      * Chia ảnh thành các block nhỏ (256/C pixels)
//      * Mỗi block được transpose độc lập bởi một thread
//      * collapse(2) làm phẳng 2 vòng lặp để tạo nhiều tasks hơn
//    - Lợi ích:
//      * Block-based approach giữ cache locality tốt
//      * Nhiều tasks hơn = load balancing tốt hơn
//      * Transpose là operation tốn kém, song song hóa tăng tốc đáng kể
//
// CÁCH SỬ DỤNG SONG SONG HÓA:
// 
// 1. Compile với OpenMP:
//    - GCC/Clang: g++ -fopenmp -O3 main.cpp -o fastblur
//    - MSVC: cl /openmp /O2 main.cpp
//
// 2. Set số threads (tùy chọn):
//    - Mặc định: OpenMP sử dụng tất cả CPU cores có sẵn
//    - Để set thủ công: export OMP_NUM_THREADS=4 (Linux/Mac) hoặc set OMP_NUM_THREADS=4 (Windows)
//
// 3. Kiểm tra song song hóa đang hoạt động:
//    - Chạy chương trình và quan sát CPU usage (nên ~100% trên tất cả cores)
//    - Đo thời gian: song song hóa nên giảm thời gian xử lý đáng kể
//
// LƯU Ý QUAN TRỌNG:
// - Song song hóa chỉ hiệu quả khi ảnh đủ lớn (thường > 100x100 pixels)
// - Với ảnh nhỏ, overhead của thread creation có thể lớn hơn lợi ích
// - Mỗi thread cần đủ bộ nhớ cache để tránh cache thrashing
// - Không có race condition vì mỗi thread xử lý vùng nhớ riêng biệt
//
// ================================================================

//!
//! \file fast_gaussian_blur_template.h
//! \author Basile Fraboni
//! \date 2017 - 2022
//!
//! \brief File này chứa implementation C++ của thuật toán Fast Gaussian Blur chạy trong thời gian tuyến tính O(n)
//!
//! Buffer ảnh được giả định có kích thước `w * h * c`, trong đó:
//!   - `h` là chiều cao của ảnh
//!   - `w` là chiều rộng của ảnh  
//!   - `c` là số kênh màu (channels) - ví dụ: 1 (grayscale), 3 (RGB), 4 (RGBA)
//!
//! Implementation mặc định chỉ hỗ trợ tối đa 4 kênh, nhưng có thể dễ dàng mở rộng để hỗ trợ nhiều kênh hơn
//! bằng cách thêm các trường hợp template cụ thể hoặc hàm generic nhận số kênh làm tham số.
//!
//! Thuật toán Fast Blur được thực hiện bằng cách áp dụng nhiều lần box blur passes (bộ lọc box đa cấp).
//! Bộ lọc này hội tụ về một Gaussian blur thật sự sau nhiều lần passes nhờ vào định lý Central Limit.
//! Trong thực tế, 3 passes (biquadratic filter) là đủ để cho kết quả chất lượng tốt.
//!
//! Tài liệu tham khảo:
//!     - http://blog.ivank.net/fastest-gaussian-blur.html
//!     - https://www.peterkovesi.com/papers/FastGaussianSmoothing.pdf
//!     - https://github.com/bfraboni/FastGaussianBlur
//!
//! **Lưu ý quan trọng về độ chính xác tại biên ảnh:**
//! Thuật toán fast gaussian blur không chính xác 100% tại các vùng biên của ảnh. 
//! Nó thực hiện việc khuếch tán tín hiệu qua nhiều passes, mỗi pass phụ thuộc vào kết quả của pass trước.
//! Một phần tín hiệu khuếch tán bị mất gần các biên và dẫn đến mất độ chính xác nhẹ cho pass tiếp theo.
//! Vấn đề này có thể được giải quyết bằng cách tăng kích thước hỗ trợ của ảnh thêm một nửa 
//! phạm vi kernel ở mỗi pass. Phần padding thêm này sẽ bắt được tín hiệu khuếch tán và làm pass tiếp theo chính xác hơn.
//! Ngược lại, true Gaussian blur không gặp vấn đề này vì toàn bộ quá trình khuếch tán được thực hiện trong một pass duy nhất.
//! Phần padding thêm không được thực hiện trong implementation này, tuy nhiên chúng ta cung cấp nhiều 
//! chính sách xử lý biên (border policies) khác nhau dẫn đến các xấp xỉ và độ chính xác khác nhau.
//! 

//!
//! \brief Enum định nghĩa các chính sách xử lý biên (border policies) cho bộ lọc.
//! 
//! Các chính sách xử lý biên xác định cách xử lý các pixel nằm ngoài phạm vi ảnh khi tính toán blur.
//! Chi tiết về các border policies có thể tham khảo tại:
//! - https://en.wikipedia.org/wiki/Kernel_(image_processing)#Edge_Handling
//! - https://www.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/top/volume-2-image-processing/filtering-functions-2/user-defined-border-types.html
//! - https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5
//! - http://iihm.imag.fr/Docs/java/jai1_0guide/Image-enhance.doc.html
//!
enum Border
{
    kExtend,        // Mở rộng (Extend): Lấy giá trị pixel gần nhất tại biên
    kKernelCrop,    // Cắt kernel (Kernel Crop): Cắt kernel để chỉ tính trên phần nằm trong ảnh
    kMirror,        // Phản chiếu (Mirror): Phản chiếu ảnh tại biên như gương
    kWrap,          // Bao quanh (Wrap): Bao quanh ảnh như torus (ảnh quấn quanh)
};

//! Hàm helper để tính toán chỉ số mảng (array indices) cho các chính sách xử lý biên khác nhau.
//! Khi kernel blur cần truy cập pixel nằm ngoài phạm vi ảnh, hàm này sẽ ánh xạ lại chỉ số
//! để lấy giá trị pixel phù hợp dựa trên chính sách biên được chọn.
//!
//! \param begin: Chỉ số bắt đầu của dải hợp lệ (thường là đầu hàng/hàng trong ảnh)
//! \param end: Chỉ số kết thúc của dải hợp lệ (không bao gồm)
//! \param index: Chỉ số muốn truy cập (có thể nằm ngoài phạm vi [begin, end))
//! \return: Chỉ số đã được ánh xạ lại nằm trong phạm vi hợp lệ [begin, end)
template<Border P>
inline int remap_index(const int begin, const int end, const int index)
{
    // Kiểm tra xem chỉ số có nằm trong phạm vi hợp lệ không
    const bool inside = index >= begin && index < end;
    if (!inside)  // Nếu nằm ngoài phạm vi, cần ánh xạ lại
    {
        if constexpr (P == kWrap)  // Chính sách Wrap: bao quanh như torus
        {
            const int length = end-begin;  // Độ dài của dải hợp lệ
            // Tính số lần cần lặp lại để đưa index vào phạm vi dương
            const int repeat = std::abs(index / length)+1;
            const int value = index + repeat * length;  // Đưa index vào phạm vi dương
            return begin+(value%length);  // Lấy phần dư để bao quanh
        }
        else if constexpr (P == kMirror)  // Chính sách Mirror: phản chiếu như gương
        {
            //! FIXME: is there a better way ?
            const int length = end-begin, last = end-1, slength = length-1;
            // Tính chỉ số tương đối từ đầu dải
            const int pindex = index < begin ? last-index+slength : index-begin;
            // Tính số lần phản chiếu (chẵn hay lẻ)
            const int repeat = pindex / slength;
            const int mod = pindex % slength;
            // Nếu repeat lẻ: phản chiếu ngược lại, nếu chẵn: giữ nguyên hướng
            return repeat%2 ? slength-mod+begin : mod+begin;
        }
        else if constexpr (P == kExtend)  // Chính sách Extend: lấy giá trị pixel gần nhất tại biên
        {
            // Clamp giá trị về phạm vi [begin, end-1]
            return std::min(end-1, std::max(begin, index));
        }
    }
    return index;  // Nếu đã nằm trong phạm vi, trả về chỉ số gốc
};

//!
//! Khi thực hiện sliding kernel accumulation (tích lũy kernel trượt), có 4 trường hợp:
//! 1. left side out and right side in:   Phần trái của kernel nằm ngoài ảnh, phần phải nằm trong
//! 2. left side in and right side in:    Cả hai phần đều nằm trong ảnh (trường hợp trung tâm)
//! 3. left side in and right side out:   Phần trái nằm trong ảnh, phần phải nằm ngoài
//! 4. left side out and right side out:  Cả hai phần đều nằm ngoài ảnh (kernel lớn hơn ảnh)
//!
//! Phân loại kích thước kernel:
//! - Small (S): kernel có bán kính < chiều rộng; r < w
//! - Mid   (M): kernel có kích thước < chiều rộng; 2r+1 < w (tức r < w/2 không đúng, nhưng r < w)
//! - Large (L): kernel có bán kính > chiều rộng; r >= w
//!
//! Các phiên bản tối ưu cho từng loại:
//! - Phiên bản fast cho (S) sử dụng 3 vòng lặp cho các trường hợp 1, 2 và 3
//! - Phiên bản fast cho (M) sử dụng 3 vòng lặp cho các trường hợp 1, 4 và 3
//! - Phiên bản fast cho (L) sử dụng 1 vòng lặp cho trường hợp 4
//!
enum Kernel
{
    kSmall,  // Kernel nhỏ: r < w/2 (thường là trường hợp phổ biến nhất)
    kMid,    // Kernel trung bình: w/2 <= r < w
    kLarge,  // Kernel lớn: r >= w (kernel lớn hơn hoặc bằng chiều rộng ảnh)
};

//!
//! \brief Hàm helper để tính giá trị làm tròn cho kiểu dữ liệu được chỉ định.
//! Hàm này được sử dụng để giảm thiểu lỗi làm tròn với các kiểu số nguyên gây ra hiện tượng làm tối ảnh.
//! Khi làm tròn số nguyên, việc cộng thêm 0.5f trước khi cast giúp làm tròn đúng (round to nearest).
//! Với số thực (float), không cần cộng thêm vì phép toán đã chính xác.
//!
template<typename T>
constexpr float round_v() { return std::is_integral_v<T> ? 0.5f : 0.f; }

//!
//! \brief Hàm này thực hiện một lần box blur theo chiều ngang (horizontal) với chính sách biên extend (mở rộng).
//! Hàm này được template hóa theo kiểu dữ liệu buffer T và số kênh màu C.
//! Đây là phiên bản tối ưu cho các kernel nhỏ hơn hoặc bằng chiều rộng ảnh (r <= w).
//!
//! \param[in] in           Buffer ảnh nguồn (source buffer)
//! \param[in,out] out      Buffer ảnh đích (target buffer) - sẽ được ghi kết quả blur
//! \param[in] w            Chiều rộng ảnh (image width)
//! \param[in] h            Chiều cao ảnh (image height)
//! \param[in] r            Bán kính box blur (box radius/dimension)
//!
template<typename T, int C, Kernel kernel = kSmall>
inline void horizontal_blur_extend(const T * in, T * out, const int w, const int h, const int r)
{
    // Thay đổi kiểu biến local dựa trên kiểu template để tính toán nhanh hơn
    // Nếu T là số nguyên (int, uchar...) thì dùng int để tính, nếu là float thì dùng float
    using calc_type = std::conditional_t<std::is_integral_v<T>, int, float>;

    // Tính nghịch đảo của kích thước box kernel để chuẩn hóa kết quả
    // Box kernel có kích thước = r + r + 1 (ví dụ: r=3 thì kernel = [x x x x x x x] = 7 pixels)
    const float iarr = 1.f / (r+r+1);
    
    // ================================================================
    // PHẦN SONG SONG HÓA (PARALLELIZATION) - CHI TIẾT:
    // ================================================================
    // #pragma omp parallel for: Đây là directive của OpenMP để song song hóa vòng lặp
    // 
    // Cách hoạt động:
    // 1. OpenMP sẽ tự động chia vòng lặp for(int i=0; i<h; i++) thành nhiều phần
    // 2. Mỗi thread (luồng) sẽ xử lý một nhóm các hàng (rows) khác nhau
    // 3. Ví dụ: Nếu có 4 threads và h=1000, có thể chia thành:
    //    - Thread 0: xử lý hàng 0-249
    //    - Thread 1: xử lý hàng 250-499
    //    - Thread 2: xử lý hàng 500-749
    //    - Thread 3: xử lý hàng 750-999
    //
    // Lý do song song hóa theo hàng:
    // - Mỗi hàng có thể được xử lý độc lập, không phụ thuộc vào kết quả của hàng khác
    // - Dữ liệu mỗi hàng nằm liên tiếp trong bộ nhớ (cache-friendly)
    // - Việc chia theo hàng tận dụng tốt cache locality
    //
    // Lợi ích:
    // - Tăng tốc đáng kể trên CPU đa nhân (multi-core)
    // - Tự động load balancing giữa các threads
    // - Không cần code phức tạp để quản lý threads thủ công
    //
    // Lưu ý: Cần compile với flag -fopenmp (GCC/Clang) hoặc /openmp (MSVC)
    //        và link với thư viện OpenMP
    // ================================================================
    OMP_PARALLEL_FOR
    for(int i=0; i<h; i++)  // Duyệt qua từng hàng của ảnh 
    {
        // Tính chỉ số bắt đầu và kết thúc của hàng hiện tại trong buffer 1D
        // Buffer được lưu dạng row-major: pixel[y][x][c] = buffer[y*w*C + x*C + c]
        const int begin = i*w;           // Chỉ số bắt đầu hàng i
        const int end = begin+w;         // Chỉ số kết thúc hàng i (không bao gồm)
        
        // Khai báo các biến tích lũy cho thuật toán sliding window:
        calc_type fv[C];  // first value: giá trị pixel đầu tiên của hàng (dùng cho extend)
        calc_type lv[C];  // last value: giá trị pixel cuối cùng của hàng (dùng cho extend)
        calc_type acc[C]; // accumulator: bộ tích lũy trượt (sliding accumulator) cho mỗi kênh màu

        // Khởi tạo fv, lv, acc bằng cách mở rộng (extend) ra ngoài buffer ảnh
        // Chính sách extend: lấy giá trị pixel gần nhất tại biên
        for(int ch=0; ch<C; ++ch)
        {
            fv[ch] =  in[begin*C+ch];        // Lấy pixel đầu tiên của hàng
            lv[ch] =  in[(end-1)*C+ch];      // Lấy pixel cuối cùng của hàng
            acc[ch] = (r+1)*fv[ch];          // Khởi tạo accumulator: giả sử có (r+1) pixel đầu có giá trị fv[ch]
        }

        // Xử lý theo từng loại kernel (Small, Mid, Large) để tối ưu hiệu năng
        if constexpr(kernel == kLarge)  // Kernel lớn: r >= w (kernel lớn hơn hoặc bằng chiều rộng ảnh)
        {
            // Khởi tạo tích lũy ban đầu cho kernel lớn
            // Với kernel lớn, hầu hết pixels đều có giá trị như nhau (lv hoặc fv)
            for(int j=0; j<r; j++)
            for(int ch=0; ch<C; ++ch)
            {
                // Nếu pixel nằm trong phạm vi ảnh (j < w), lấy giá trị thực
                // Nếu nằm ngoài, dùng giá trị pixel cuối (last value - extend policy)
                acc[ch] += j < w ? in[(begin+j)*C+ch] : lv[ch];
            }

            // Với kernel lớn, tất cả pixels trong hàng có cùng giá trị trung bình
            // Không cần trượt window phức tạp, chỉ cần tính trung bình của toàn bộ hàng
            for(int ti = begin; ti < end; ti++)  // ti = target index (chỉ số pixel đích)
            for(int ch=0; ch<C; ++ch)
            {
                // Với kernel lớn, accumulator không thay đổi khi trượt (vì tất cả đều extend)
                acc[ch] += lv[ch] - fv[ch];  // Điều chỉnh accumulator (thực tế không đổi nhiều)
                // assert(acc[ch] >= 0);  // Kiểm tra debug: accumulator phải >= 0
                
                // Tính giá trị blur: nhân với nghịch đảo kích thước kernel và làm tròn
                out[ti*C+ch] = acc[ch]*iarr + round_v<T>();
            }
        }
        else if constexpr(kernel == kMid)  // Kernel trung bình: w/2 <= r < w
        {
            // Khởi tạo các chỉ số cho sliding window:
            int ti = begin;        // target index: chỉ số pixel đang được tính toán (vị trí trung tâm kernel)
            int li = begin-r-1;    // left index: chỉ số pixel bên trái của kernel (sẽ bị loại bỏ khi trượt)
            int ri = begin+r;      // right index: chỉ số pixel bên phải của kernel (sẽ được thêm vào khi trượt)

            // Khởi tạo tích lũy ban đầu: tính tổng của kernel đầu tiên
            // Kernel đầu tiên bao gồm các pixel từ ti đến ri (chưa bao gồm ri)
            for(int j=ti; j<ri; j++)
            for(int ch=0; ch<C; ++ch)
            {
                acc[ch] += in[j*C+ch];  // Cộng dồn giá trị pixel vào accumulator
            }

            // TRƯỜNG HỢP 1: Phần trái của kernel nằm ngoài ảnh, phần phải nằm trong
            // (li < begin: pixel bên trái ngoài ảnh, ri < end: pixel bên phải trong ảnh)
            // Đây là phần đầu của hàng, kernel đang "trượt vào" từ bên trái
            for(; ri<end; ri++, ti++, li++)  // Trượt kernel sang phải
            for(int ch=0; ch<C; ++ch)
            {
                // Cập nhật accumulator: thêm pixel bên phải mới, trừ pixel đầu tiên (extend policy)
                acc[ch] += in[ri*C+ch] - fv[ch];  // fv[ch] = giá trị pixel đầu hàng (extend)
                // assert(acc[ch] >= 0);
                // Tính và lưu giá trị blur cho pixel hiện tại
                out[ti*C+ch] = acc[ch]*iarr + round_v<T>();
            }

            // TRƯỜNG HỢP 4: Cả phần trái và phải đều nằm ngoài ảnh (kernel lớn hơn ảnh)
            // (li < begin: phần trái ngoài, ri >= end: phần phải ngoài)
            // Với kernel mid, có thể xảy ra khi kernel gần bằng chiều rộng ảnh
            for(; li<begin; ti++, li++)  // Tiếp tục trượt, nhưng cả hai phía đều extend
            for(int ch=0; ch<C; ++ch)
            {
                // Với kernel mid trong trường hợp này, accumulator không đổi nhiều
                acc[ch] += lv[ch] - fv[ch];  // Cả hai phía đều dùng giá trị biên
                // assert(acc[ch] >= 0);
                out[ti*C+ch] = acc[ch]*iarr + round_v<T>();
            }

            // TRƯỜNG HỢP 3: Phần trái nằm trong ảnh, phần phải nằm ngoài
            // (li >= begin: pixel bên trái trong ảnh, ri >= end: pixel bên phải ngoài ảnh)
            // Đây là phần cuối của hàng, kernel đang "trượt ra" bên phải
            for(; ti<end; ti++, li++)  // Tiếp tục trượt đến cuối hàng
            for(int ch=0; ch<C; ++ch)
            {
                // Cập nhật accumulator: thêm giá trị pixel cuối (extend), trừ pixel bên trái
                acc[ch] += lv[ch] - in[li*C+ch];  // lv[ch] = giá trị pixel cuối hàng (extend)
                // assert(acc[ch] >= 0);
                out[ti*C+ch] = acc[ch]*iarr + round_v<T>();
            }
        }
        else if constexpr(kernel == kSmall)  // Kernel nhỏ: r < w/2 (trường hợp phổ biến nhất)
        {
            // Khởi tạo các chỉ số cho sliding window:
            int ti = begin;        // target index: chỉ số pixel đang được tính toán (vị trí trung tâm kernel)
            int li = begin-r-1;    // left index: chỉ số pixel bên trái của kernel (sẽ bị loại bỏ khi trượt)
            int ri = begin+r;      // right index: chỉ số pixel bên phải của kernel (sẽ được thêm vào khi trượt)

            // Khởi tạo tích lũy ban đầu: tính tổng của kernel đầu tiên
            // Kernel đầu tiên bao gồm các pixel từ ti đến ri (chưa bao gồm ri)
            for(int j=ti; j<ri; j++)
            for(int ch=0; ch<C; ++ch)
            {
                acc[ch] += in[j*C+ch];  // Cộng dồn giá trị pixel vào accumulator
            }

            // TRƯỜNG HỢP 1: Phần trái của kernel nằm ngoài ảnh, phần phải nằm trong
            // (li < begin: pixel bên trái ngoài ảnh, ri < end: pixel bên phải trong ảnh)
            // Đây là phần đầu của hàng, kernel đang "trượt vào" từ bên trái
            // Thuật toán sliding window: O(n) thay vì O(n*r) bằng cách cộng/trừ thay vì tính lại tổng
            for(; li<begin; ri++, ti++, li++)  // Trượt kernel sang phải
            for(int ch=0; ch<C; ++ch)
            {
                // Cập nhật accumulator: thêm pixel bên phải mới, trừ pixel đầu tiên (extend policy)
                // Đây là điểm mấu chốt của sliding window: không tính lại toàn bộ, chỉ cập nhật
                acc[ch] += in[ri*C+ch] - fv[ch];  // fv[ch] = giá trị pixel đầu hàng (extend)
                // assert(acc[ch] >= 0);
                // Tính và lưu giá trị blur cho pixel hiện tại
                out[ti*C+ch] = acc[ch]*iarr + round_v<T>();
            }

            // TRƯỜNG HỢP 2: Cả phần trái và phải đều nằm trong ảnh (trường hợp trung tâm)
            // (li >= begin: pixel bên trái trong ảnh, ri < end: pixel bên phải trong ảnh)
            // Đây là phần giữa của hàng, kernel hoàn toàn nằm trong ảnh - phần hiệu quả nhất
            for(; ri<end; ri++, ti++, li++)  // Trượt kernel sang phải
            for(int ch=0; ch<C; ++ch)
            {
                // Cập nhật accumulator: thêm pixel bên phải mới, trừ pixel bên trái cũ
                // Không cần extend vì cả hai pixel đều nằm trong ảnh
                acc[ch] += in[ri*C+ch] - in[li*C+ch];  // Cập nhật O(1) thay vì tính lại O(r)
                // assert(acc[ch] >= 0);
                out[ti*C+ch] = acc[ch]*iarr + round_v<T>();
            }

            // TRƯỜNG HỢP 3: Phần trái nằm trong ảnh, phần phải nằm ngoài
            // (li >= begin: pixel bên trái trong ảnh, ri >= end: pixel bên phải ngoài ảnh)
            // Đây là phần cuối của hàng, kernel đang "trượt ra" bên phải
            for(; ti<end; ti++, li++)  // Tiếp tục trượt đến cuối hàng
            for(int ch=0; ch<C; ++ch)
            {
                // Cập nhật accumulator: thêm giá trị pixel cuối (extend), trừ pixel bên trái
                acc[ch] += lv[ch] - in[li*C+ch];  // lv[ch] = giá trị pixel cuối hàng (extend)
                // assert(acc[ch] >= 0);
                out[ti*C+ch] = acc[ch]*iarr + round_v<T>();
            }
        }
    }
}

//!
//! \brief Hàm này thực hiện một lần box blur theo chiều ngang với chính sách biên kernel crop (cắt kernel).
//! Với chính sách này, khi kernel nằm gần biên, chỉ tính trên phần kernel nằm trong ảnh.
//! Hàm được template hóa theo kiểu dữ liệu buffer T và số kênh màu C.
//!
//! \param[in] in           Buffer ảnh nguồn (source buffer)
//! \param[in,out] out      Buffer ảnh đích (target buffer)
//! \param[in] w            Chiều rộng ảnh (image width)
//! \param[in] h            Chiều cao ảnh (image height)
//! \param[in] r            Bán kính box blur (box radius/dimension)
//!
template<typename T, int C, Kernel kernel = kSmall>
inline void horizontal_blur_kernel_crop(const T * in, T * out, const int w, const int h, const int r)
{
    // Thay đổi kiểu biến local dựa trên kiểu template để tính toán nhanh hơn
    using calc_type = std::conditional_t<std::is_integral_v<T>, int, float>;

    // Tính nghịch đảo kích thước kernel chuẩn (khi kernel hoàn toàn trong ảnh)
    const float iarr = 1.f / (r+r+1);
    // Tính nghịch đảo chiều rộng ảnh (dùng khi kernel lớn hơn ảnh)
    const float iwidth = 1.f / w;
    
    // ================================================================
    // SONG SONG HÓA: Tương tự như horizontal_blur_extend
    // Mỗi thread xử lý một nhóm hàng độc lập, tăng tốc trên CPU đa nhân
    // ================================================================
    OMP_PARALLEL_FOR
    for(int i=0; i<h; i++)  // Duyệt qua từng hàng của ảnh
    {
        const int begin = i*w;
        const int end = begin+w;
        calc_type acc[C] = { 0 };

        if constexpr(kernel == kLarge)
        {
            // initial acucmulation
            for(int j=begin; j<end; j++)
            for(int ch=0; ch < C; ++ch)
            {
                acc[ch] += in[j*C+ch];
            }

            // this is constant
            for(int j=begin; j<end; j++)
            for(int ch=0; ch < C; ++ch)
            {
                out[j*C+ch] = acc[ch]*iwidth + round_v<T>();
            }
        }
        else if constexpr(kernel == kMid)
        {
            // current index, left index, right index
            int ti = begin, li = begin-r-1, ri = begin+r;   
            
            // initial acucmulation
            for(int j=ti; j<ri; j++)
            for(int ch=0; ch<C; ++ch)
            {
                acc[ch] += in[j*C+ch];
            }

            // 1. left side out and right side in
            for(; ri<end; ri++, ti++, li++)
            for(int ch=0; ch<C; ++ch)
            { 
                acc[ch] += in[ri*C+ch];
                // assert(acc[ch] >= 0);
                const float inorm = 1.f / float(ri+1-begin);
                out[ti*C+ch] = acc[ch]*inorm + round_v<T>();
            }

            // 4. left side out and right side out
            for(; li<begin; ti++, li++)
            for(int ch=0; ch<C; ++ch)
            { 
                out[ti*C+ch] = acc[ch]*iwidth + round_v<T>();
            }

            // 3. left side in and right side out
            for(; ti<end; ti++, li++)
            for(int ch=0; ch<C; ++ch)
            { 
                acc[ch] -= in[li*C+ch];
                // assert(acc[ch] >= 0);
                const float inorm = 1.f / float(end-li-1);
                out[ti*C+ch] = acc[ch]*inorm + round_v<T>();
            }
        }
        else if constexpr(kernel == kSmall)
        {
            // current index, left index, right index
            int ti = begin, li = begin-r-1, ri = begin+r;

            // initial acucmulation
            for(int j=ti; j<ri; j++)
            for(int ch=0; ch<C; ++ch)
            {
                acc[ch] += in[j*C+ch];
            }

            // 1. left side out and right side in
            for(; li<begin; ri++, ti++, li++)
            for(int ch=0; ch<C; ++ch)
            { 
                acc[ch] += in[ri*C+ch];
                // assert(acc[ch] >= 0);
                const float inorm = 1.f / float(ri+1-begin);
                out[ti*C+ch] = acc[ch]*inorm + round_v<T>();
            }

            // 2. left side in and right side in
            for(; ri<end; ri++, ti++, li++)
            for(int ch=0; ch<C; ++ch)
            { 
                acc[ch] += in[ri*C+ch] - in[li*C+ch];
                // assert(acc[ch] >= 0);
                out[ti*C+ch] = acc[ch]*iarr + round_v<T>();
            }

            // 3. left side in and right side out
            for(; ti<end; ti++, li++)
            for(int ch=0; ch<C; ++ch)
            { 
                acc[ch] -= in[li*C+ch];
                // assert(acc[ch] >= 0);
                const float inorm = 1.f / float(end-li-1);
                out[ti*C+ch] = acc[ch]*inorm + round_v<T>();
            }
        }
    }
}

//!
//! \brief Hàm này thực hiện một lần box blur theo chiều ngang với chính sách biên mirror (phản chiếu).
//! Với chính sách này, các pixel ngoài biên được lấy bằng cách phản chiếu ảnh như gương.
//! Hàm được template hóa theo kiểu dữ liệu buffer T và số kênh màu C.
//!
//! \param[in] in           Buffer ảnh nguồn (source buffer)
//! \param[in,out] out      Buffer ảnh đích (target buffer)
//! \param[in] w            Chiều rộng ảnh (image width)
//! \param[in] h            Chiều cao ảnh (image height)
//! \param[in] r            Bán kính box blur (box radius/dimension)
//!
//! \todo Rework this one at some point.
template<typename T, int C, Kernel kernel = kSmall>
inline void horizontal_blur_mirror(const T* in, T* out, const int w, const int h, const int r)
{
    // Thay đổi kiểu biến local dựa trên kiểu template để tính toán nhanh hơn
    using calc_type = std::conditional_t<std::is_integral_v<T>, int, float>;

    // Tính nghịch đảo kích thước kernel để chuẩn hóa kết quả
    const double iarr = 1.f/(r+r+1);
    
    // ================================================================
    // SONG SONG HÓA: Mỗi hàng được xử lý song song bởi các threads khác nhau
    // Thread pool được quản lý tự động bởi OpenMP runtime
    // ================================================================
    OMP_PARALLEL_FOR
    for (int i = 0; i < h; i++)  // Duyệt qua từng hàng của ảnh
    {
        const int begin = i*w;
        const int end = begin+w;
        calc_type acc[C] = { 0 };

        // current index, left index, right index
        int ti = begin, li = begin-r-1, ri = begin+r;

        if constexpr(kernel == kLarge) // generic but slow
        {
            // initial acucmulation
            for(int j=li; j<ri; j++) 
            for(int ch=0; ch<C; ++ch)
            {
                const int id = remap_index<kMirror>(begin, end, j);
                acc[ch] += in[id*C+ch];
            }

            // perform filtering
            for(int j=0; j<w; j++, ri++, ti++, li++) 
            for(int ch=0; ch<C; ++ch)
            { 
                const int rid = remap_index<kMirror>(begin, end, ri);
                const int lid = remap_index<kMirror>(begin, end, li);
                acc[ch] += in[rid*C+ch] - in[lid*C+ch];
                out[ti*C+ch] = acc[ch]*iarr + round_v<T>();
            }
        }
        else if constexpr(kernel == kMid)
        {
            for(int j=li; j<begin; j++) 
            for(int ch=0; ch<C; ++ch)
            {
                const int lid = 2 * begin - j; // mirrored id
                acc[ch] += in[lid*C+ch];
            }

            for(int j=begin; j<ri; j++) 
            for(int ch=0; ch<C; ++ch)
            {
                acc[ch] += in[j*C+ch];
            }

            // 1. left side out and right side in
            for(; ri<end; ri++, ti++, li++)
            for(int ch=0; ch<C; ++ch)
            { 
                const int lid = 2 * begin - li; // left mirrored id
                acc[ch] += in[ri*C+ch] - in[lid*C+ch];
                // assert(acc[ch] >= 0);
                out[ti*C+ch] = acc[ch]*iarr + round_v<T>();
            }

            // 4. left side out and right side out
            for(; li<begin; ri++, ti++, li++)
            for(int ch=0; ch<C; ++ch)
            { 
                const int rid = 2 * end - 2 - ri;   // right mirrored id
                const int lid = 2 * begin - li;     // left mirrored id
                acc[ch] += in[rid*C+ch] - in[lid*C+ch];
                // assert(acc[ch] >= 0);
                out[ti*C+ch] = acc[ch]*iarr + round_v<T>();
            }

            // 3. left side in and right side out
            for(; ti<end; ri++, ti++, li++)
            for(int ch=0; ch<C; ++ch)
            {
                const int rid = 2*end-2-ri; // right mirrored id
                acc[ch] += in[rid*C+ch] - in[li*C+ch];
                // assert(acc[ch] >= 0);
                out[ti*C+ch] = acc[ch]*iarr + round_v<T>();
            }
        }
        else if constexpr(kernel == kSmall)
        {
            for(int j=li; j<begin; j++) 
            for(int ch=0; ch<C; ++ch)
            {
                const int lid = 2 * begin - j; // mirrored id
                acc[ch] += in[lid*C+ch];
            }

            for(int j=begin; j<ri; j++) 
            for(int ch=0; ch<C; ++ch)
            {
                acc[ch] += in[j*C+ch];
            }

            // 1. left side out and right side in
            for(; li<begin; ri++, ti++, li++)
            for(int ch=0; ch<C; ++ch)
            { 
                const int lid = 2 * begin - li; // left mirrored id
                acc[ch] += in[ri*C+ch] - in[lid*C+ch];
                // assert(acc[ch] >= 0);
                out[ti*C+ch] = acc[ch]*iarr + round_v<T>();
            }

            // 2. left side in and right side in
            for(; ri<end; ri++, ti++, li++) 
            for(int ch=0; ch<C; ++ch)
            { 
                acc[ch] += in[ri*C+ch] - in[li*C+ch];
                // assert(acc[ch] >= 0);
                out[ti*C+ch] = acc[ch]*iarr + round_v<T>();
            }

            // 3. left side in and right side out
            for(; ti<end; ri++, ti++, li++)
            for(int ch=0; ch<C; ++ch)
            {
                const int rid = 2*end-2-ri; // right mirrored id
                acc[ch] += in[rid*C+ch] - in[li*C+ch];
                // assert(acc[ch] >= 0);
                out[ti*C+ch] = acc[ch]*iarr + round_v<T>();
            }
        }
    }
}

//!
//! \brief Hàm này thực hiện một lần box blur theo chiều ngang với chính sách biên wrap (bao quanh).
//! Với chính sách này, ảnh được coi như một torus - pixel ngoài biên trái quấn về phía phải và ngược lại.
//! Hàm được template hóa theo kiểu dữ liệu buffer T và số kênh màu C.
//! Đây là phiên bản generic cho tất cả kích thước kernel.
//!
//! \param[in] in           Buffer ảnh nguồn (source buffer)
//! \param[in,out] out      Buffer ảnh đích (target buffer)
//! \param[in] w            Chiều rộng ảnh (image width)
//! \param[in] h            Chiều cao ảnh (image height)
//! \param[in] r            Bán kính box blur (box radius/dimension)
//!
//! \todo Make a faster version for small kernels.
template<typename T, int C>
inline void horizontal_blur_wrap(const T* in, T* out, const int w, const int h, const int r)
{
    // Thay đổi kiểu biến local dựa trên kiểu template để tính toán nhanh hơn
    using calc_type = std::conditional_t<std::is_integral_v<T>, int, float>;

    // Tính nghịch đảo kích thước kernel để chuẩn hóa kết quả
    const float iarr = 1.f / (r+r+1);
    
    // ================================================================
    // SONG SONG HÓA: Xử lý song song các hàng của ảnh
    // Mỗi thread xử lý một tập hợp hàng độc lập, không có race condition
    // vì mỗi hàng được ghi vào vùng nhớ riêng biệt
    // ================================================================
    OMP_PARALLEL_FOR
    for(int i=0; i<h; i++)  // Duyệt qua từng hàng của ảnh 
    {
        const int begin = i*w;
        const int end = begin+w;
        int ti = begin, li = begin-r-1, ri = begin+r;   // current index, left index, right index
        calc_type acc[C] = { 0 };                       // sliding accumulator

        // initial acucmulation
        for(int j=li; j<ri; j++) 
        for(int ch=0; ch<C; ++ch)
        {
            const int id = remap_index<kWrap>(begin, end, j);
            acc[ch] += in[id*C+ch];
        }

        // perform filtering
        for(int j=0; j<w; j++, ri++, ti++, li++) 
        for(int ch=0; ch<C; ++ch)
        {
            const int rid = remap_index<kWrap>(begin, end, ri);
            const int lid = remap_index<kWrap>(begin, end, li);
            acc[ch] += in[rid*C+ch] - in[lid*C+ch];
            out[ti*C+ch] = acc[ch]*iarr + round_v<T>();
        }
    }
}

//!
//! \brief Hàm dispatcher template cho horizontal_blur.
//! Template hóa theo kiểu dữ liệu buffer T, số kênh màu C, và border policy P.
//! Hàm này tự động chọn phiên bản tối ưu dựa trên border policy và kích thước kernel.
//!
//! \param[in] in           Buffer ảnh nguồn (source buffer)
//! \param[in,out] out      Buffer ảnh đích (target buffer)
//! \param[in] w            Chiều rộng ảnh (image width)
//! \param[in] h            Chiều cao ảnh (image height)
//! \param[in] r            Bán kính box blur (box dimension/radius)
//!
template<typename T, int C, Border P = kMirror>
inline void horizontal_blur(const T * in, T * out, const int w, const int h, const int r)
{
    // Dispatch theo border policy (compile-time) và kích thước kernel (runtime)
    if constexpr(P == kExtend)  // Chính sách Extend
    {
        // Chọn phiên bản tối ưu dựa trên kích thước kernel so với chiều rộng ảnh
        if( r < w/2 )       horizontal_blur_extend<T,C,Kernel::kSmall>(in, out, w, h, r);  // Kernel nhỏ
        else if( r < w )    horizontal_blur_extend<T,C,Kernel::kMid  >(in, out, w, h, r);  // Kernel trung bình
        else                horizontal_blur_extend<T,C,Kernel::kLarge>(in, out, w, h, r);  // Kernel lớn
    }
    else if constexpr(P == kKernelCrop)  // Chính sách Kernel Crop
    {
        if( r < w/2 )       horizontal_blur_kernel_crop<T,C,Kernel::kSmall>(in, out, w, h, r);
        else if( r < w )    horizontal_blur_kernel_crop<T,C,Kernel::kMid  >(in, out, w, h, r);
        else                horizontal_blur_kernel_crop<T,C,Kernel::kLarge>(in, out, w, h, r);
    }
    else if constexpr(P == kMirror)  // Chính sách Mirror
    {
        if( r < w/2 )       horizontal_blur_mirror<T,C,Kernel::kSmall>(in, out, w, h, r);
        else if( r < w )    horizontal_blur_mirror<T,C,Kernel::kMid  >(in, out, w, h, r);
        else                horizontal_blur_mirror<T,C,Kernel::kLarge>(in, out, w, h, r);
    }
    else if constexpr(P == kWrap)  // Chính sách Wrap (chỉ có 1 phiên bản generic)
    {
        horizontal_blur_wrap<T,C>(in, out, w, h, r);
    }
}

//!
//! \brief Hàm dispatcher template cho horizontal_blur. Template hóa theo kiểu dữ liệu T và border policy P.
//! Hàm này dispatch theo số kênh màu c để gọi phiên bản template tối ưu tương ứng.
//!
//! \param[in] in           Buffer ảnh nguồn (source buffer)
//! \param[in,out] out      Buffer ảnh đích (target buffer)
//! \param[in] w            Chiều rộng ảnh (image width)
//! \param[in] h            Chiều cao ảnh (image height)
//! \param[in] c            Số kênh màu (image channels)
//! \param[in] r            Bán kính box blur (box dimension/radius)
//!
template<typename T, Border P = kMirror>
inline void horizontal_blur(const T * in, T * out, const int w, const int h, const int c, const int r)
{
    // Dispatch theo số kênh màu để gọi phiên bản template tối ưu
    // Việc này giúp compiler có thể unroll loops và optimize tốt hơn
    switch(c)
    {
        case 1: horizontal_blur<T,1,P>(in, out, w, h, r); break;  // Grayscale
        case 2: horizontal_blur<T,2,P>(in, out, w, h, r); break;  // 2 channels
        case 3: horizontal_blur<T,3,P>(in, out, w, h, r); break;  // RGB
        case 4: horizontal_blur<T,4,P>(in, out, w, h, r); break;  // RGBA
        default: printf("horizontal_blur over %d channels is not supported yet. Add a specific case if possible or fall back to the generic version.\n", c); break;
        // default: horizontal_blur<T>(in, out, w, h, c, r); break;
    }
}

//!
//! \brief Hàm này thực hiện chuyển vị (transpose) 2D của ảnh.
//! Việc chuyển vị được thực hiện theo từng khối (block) để giảm số lần cache miss 
//! và cải thiện cache coherency cho các buffer ảnh lớn.
//! 
//! Cache coherency: Thay vì transpose toàn bộ ảnh một lúc (gây nhiều cache miss),
//! ta chia ảnh thành các block nhỏ và transpose từng block, giữ dữ liệu trong cache.
//!
//! Hàm được template hóa theo kiểu dữ liệu buffer T và số kênh màu C.
//!
//! \param[in] in           Buffer ảnh nguồn (source buffer) - dạng row-major
//! \param[in,out] out      Buffer ảnh đích (target buffer) - sẽ chứa ảnh đã transpose (column-major)
//! \param[in] w            Chiều rộng ảnh gốc (image width) - trở thành chiều cao sau transpose
//! \param[in] h            Chiều cao ảnh gốc (image height) - trở thành chiều rộng sau transpose
//!
template<typename T, int C>
inline void flip_block(const T * in, T * out, const int w, const int h)
{
    // Kích thước block để transpose: 256/C pixels
    // Block size được chọn để vừa với cache L1 (thường ~32KB)
    // Chia cho C vì mỗi pixel có C channels
    constexpr int block = 256/C;
    
    // ================================================================
    // PHẦN SONG SONG HÓA (PARALLELIZATION) - CHI TIẾT:
    // ================================================================
    // #pragma omp parallel for collapse(2): Song song hóa 2 vòng lặp lồng nhau
    //
    // Cách hoạt động của collapse(2):
    // 1. OpenMP sẽ "làm phẳng" 2 vòng lặp lồng nhau thành 1 vòng lặp lớn
    // 2. Sau đó chia đều các iteration cho các threads
    // 3. Ví dụ: w=1000, h=800, block=256
    //    - Vòng lặp ngoài: x = 0, 256, 512, 768 (4 iterations)
    //    - Vòng lặp trong: y = 0, 256, 512, 768 (4 iterations)
    //    - Tổng: 4*4 = 16 iterations (block combinations)
    //    - Với 4 threads, mỗi thread xử lý ~4 block combinations
    //
    // Tại sao dùng collapse(2):
    // - Nếu chỉ parallelize vòng lặp ngoài, có thể không đủ work cho nhiều threads
    // - Collapse(2) tạo nhiều tasks hơn, load balancing tốt hơn
    // - Đặc biệt hiệu quả khi số blocks theo một chiều nhỏ
    //
    // Lợi ích của song song hóa transpose:
    // - Transpose là operation tốn kém (O(w*h)), song song hóa tăng tốc đáng kể
    // - Mỗi block được xử lý độc lập, không có race condition
    // - Block-based approach giữ cache locality tốt
    //
    // Lưu ý: Cần compile với -fopenmp và link với OpenMP library
    // ================================================================
    OMP_PARALLEL_FOR_COLLAPSE_2
    for(int x= 0; x < w; x+= block)     // Duyệt theo block theo chiều ngang
    for(int y= 0; y < h; y+= block)     // Duyệt theo block theo chiều dọc
    {
        // Tính con trỏ bắt đầu của block trong buffer nguồn (in) và buffer đích (out)
        // in: row-major format - pixel tại (y, x) = in[y*w*C + x*C]
        const T * p = in + y*w*C + x*C;
        // out: column-major format (sau transpose) - pixel tại (x, y) = out[y*C + x*h*C]
        T * q = out + y*C + x*h*C;
        
        // Tính kích thước thực tế của block (có thể nhỏ hơn block size ở biên)
        const int blockx= std::min(w, x+block) - x;  // Chiều rộng block (theo x)
        const int blocky= std::min(h, y+block) - y;  // Chiều cao block (theo y)
        
        // Transpose block: copy từng pixel từ row-major sang column-major
        for(int xx= 0; xx < blockx; xx++)  // Duyệt theo chiều ngang trong block
        {
            for(int yy= 0; yy < blocky; yy++)  // Duyệt theo chiều dọc trong block
            {
                // Copy tất cả channels của pixel hiện tại
                for(int k= 0; k < C; k++)
                    q[k]= p[k];
                // Di chuyển con trỏ: p sang hàng tiếp theo trong ảnh gốc
                p+= w*C;
                // Di chuyển con trỏ: q sang cột tiếp theo trong ảnh đã transpose
                q+= C;
            }
            // Sau khi copy xong một cột trong block, di chuyển sang cột tiếp theo
            // Quay lại đầu cột và di chuyển sang cột kế
            p+= -blocky*w*C + C;  // Quay lại đầu cột hiện tại, di chuyển sang pixel tiếp theo trong hàng
            q+= -blocky*C + h*C;  // Quay lại đầu cột hiện tại trong output, di chuyển sang cột tiếp theo
        }
    }
}
//!
//! \brief Hàm dispatcher template cho flip_block. Template hóa theo kiểu dữ liệu buffer T.
//! Hàm này chọn phiên bản flip_block phù hợp dựa trên số kênh màu c.
//! Việc dùng switch case giúp compiler optimize tốt hơn so với template đệ quy.
//!
//! \param[in] in           Buffer ảnh nguồn (source buffer)
//! \param[in,out] out      Buffer ảnh đích (target buffer)
//! \param[in] w            Chiều rộng ảnh (image width)
//! \param[in] h            Chiều cao ảnh (image height)
//! \param[in] c            Số kênh màu (image channels)
//!
template<typename T>
inline void flip_block(const T * in, T * out, const int w, const int h, const int c)
{
    // Dispatch theo số kênh màu để gọi phiên bản template tối ưu tương ứng
    switch(c)
    {
        case 1: flip_block<T,1>(in, out, w, h); break;  // Grayscale (1 channel)
        case 2: flip_block<T,2>(in, out, w, h); break;  // 2 channels (ví dụ: grayscale + alpha)
        case 3: flip_block<T,3>(in, out, w, h); break;  // RGB (3 channels)
        case 4: flip_block<T,4>(in, out, w, h); break;  // RGBA (4 channels)
        default: printf("flip_block over %d channels is not supported yet. Add a specific case if possible or fall back to the generic version.\n", c); break;
        // default: flip_block<T>(in, out, w, h, c); break;
    }
}

//!
//! \brief Hàm này chuyển đổi độ lệch chuẩn (standard deviation) của Gaussian blur 
//! thành bán kính box (box radius) cho mỗi lần box blur pass.
//! Trả về giá trị sigma xấp xỉ đạt được với N lần box blur passes.
//!
//! Thuật toán dựa trên công thức từ paper của Peter Kovesi để xấp xỉ Gaussian blur
//! bằng nhiều lần box blur. Việc sử dụng các box có kích thước khác nhau giúp
//! xấp xỉ Gaussian tốt hơn so với dùng cùng một kích thước.
//!
//! Tài liệu tham khảo:
//! - https://www.peterkovesi.com/papers/FastGaussianSmoothing.pdf
//!
//! \param[out] boxes   Mảng chứa bán kính box cho mỗi pass (kernel sizes = 2*boxes[i]+1)
//! \param[in] sigma    Độ lệch chuẩn Gaussian mong muốn
//! \param[in] n        Số lần box blur passes
//! \return             Giá trị sigma xấp xỉ đạt được
//!
inline float sigma_to_box_radius(int boxes[], const float sigma, const int n)
{
    // Tính kích thước filter lý tưởng (ideal filter width) từ sigma và số passes
    // Công thức: wi = sqrt(12*sigma^2/n + 1)
    // Đây là kích thước box filter tối ưu để xấp xỉ Gaussian với n passes
    float wi = std::sqrt((12*sigma*sigma/n)+1);
    
    // Làm tròn xuống đến số lẻ gần nhất (box filter cần kích thước lẻ)
    int wl = wi;  // không cần std::floor vì cast tự động làm tròn xuống
    if(wl%2==0) wl--;  // Đảm bảo số lẻ (ví dụ: 4 -> 3, 5 -> 5)
    int wu = wl+2;     // Kích thước lớn hơn tiếp theo (cũng là số lẻ)
                
    // Tính số lượng passes nên dùng box nhỏ (wl) và box lớn (wu)
    // Sử dụng công thức tối ưu để minimize error giữa tổng box blurs và Gaussian
    float mi = (12*sigma*sigma - n*wl*wl - 4*n*wl - 3*n)/(-4*wl - 4);
    int m = mi+0.5f;  // Làm tròn (tránh std::round bằng cách cộng 0.5f rồi cast)
                
    // Phân bổ: m passes đầu dùng box nhỏ (wl), (n-m) passes sau dùng box lớn (wu)
    for(int i=0; i<n; i++)
        boxes[i] = ((i < m ? wl : wu) - 1) / 2;  // Chuyển từ width sang radius

    // Trả về sigma xấp xỉ thực tế đạt được với cấu hình này
    return std::sqrt((m*wl*wl+(n-m)*wu*wu-n)/12.f);
}

//!
//! \brief Hàm này thực hiện Fast Gaussian Blur. Được template hóa theo kiểu dữ liệu T và số passes N.
//!
//! Áp dụng nhiều lần box blur sẽ hội tụ về một Gaussian blur thật sự (nhờ định lý Central Limit).
//! Ba passes là đủ để có kết quả tốt. Buffer input cũng được dùng làm temporary và bị sửa đổi
//! trong quá trình xử lý nên không thể là const.
//!
//! Thông thường quá trình nên luân phiên giữa horizontal và vertical passes
//! với số lần tương ứng với số box blur passes mong muốn. Tuy nhiên nhờ tính chất của box blur,
//! các passes có thể tách biệt (separable) có thể thực hiện theo bất kỳ thứ tự nào mà không thay đổi kết quả.
//! Vì lý do hiệu năng, thuật toán được tối ưu như sau:
//! - Áp dụng N lần horizontal blur (horizontal passes)
//! - Chuyển vị buffer ảnh (transposition) - biến horizontal thành vertical
//! - Áp dụng N lần horizontal blur (thực chất là vertical passes trên ảnh đã transpose)
//! - Chuyển vị lại buffer ảnh (transposition) - trả về dạng ban đầu
//!
//! Có hai phiên bản của hàm:
//! - Generic N passes (dùng nhiều std::swap hơn)
//! - Specialized 3 passes (tối ưu riêng cho 3 passes)
//!
//! \param[in,out] in       Con trỏ tham chiếu đến buffer nguồn (sẽ bị sửa đổi)
//! \param[in,out] out      Con trỏ tham chiếu đến buffer đích (sẽ bị sửa đổi)
//! \param[in] w            Chiều rộng ảnh
//! \param[in] h            Chiều cao ảnh
//! \param[in] c            Số kênh màu
//! \param[in] sigma        Độ lệch chuẩn Gaussian
//!
template<typename T, unsigned int N, Border P>
inline void fast_gaussian_blur(T *& in, T *& out, const int w, const int h, const int c, const float sigma) 
{
    // Tính toán kích thước box kernel cho mỗi pass
    // Sử dụng công thức tối ưu để xấp xỉ Gaussian với N passes
    int boxes[N];
    sigma_to_box_radius(boxes, sigma, N);

    // ================================================================
    // BƯỚC 1: THỰC HIỆN N LẦN HORIZONTAL BLUR PASSES
    // ================================================================
    // Mỗi pass sẽ được song song hóa bên trong horizontal_blur
    // Các passes này xử lý blur theo chiều ngang của ảnh
    for(int i = 0; i < N; ++i)
    {
        // Thực hiện horizontal blur với box radius boxes[i]
        // Hàm này sử dụng OpenMP để song song hóa theo từng hàng
        horizontal_blur<T,P>(in, out, w, h, c, boxes[i]);
        // Hoán đổi con trỏ: output của pass này trở thành input của pass tiếp theo
        std::swap(in, out);
    }   

    // ================================================================
    // BƯỚC 2: CHUYỂN VỊ (TRANSPOSE) BUFFER ẢNH
    // ================================================================
    // Transpose biến ảnh từ dạng row-major sang column-major
    // Sau transpose: blur ngang trên ảnh gốc = blur dọc trên ảnh đã transpose
    // Hàm flip_block cũng được song song hóa với collapse(2)
    flip_block(in, out, w, h, c);
    std::swap(in, out);  // Hoán đổi con trỏ sau transpose
    
    // ================================================================
    // BƯỚC 3: THỰC HIỆN N LẦN HORIZONTAL BLUR TRÊN ẢNH ĐÃ TRANSPOSE
    // ================================================================
    // Vì ảnh đã được transpose, blur ngang trên ảnh transpose = blur dọc trên ảnh gốc
    // Chú ý: w và h đã đổi chỗ sau transpose (w_old = h_new, h_old = w_new)
    for(int i = 0; i < N; ++i)
    {
        // Horizontal blur trên ảnh đã transpose (thực chất là vertical blur trên ảnh gốc)
        horizontal_blur<T,P>(in, out, h, w, c, boxes[i]);
        std::swap(in, out);
    }   
    
    // ================================================================
    // BƯỚC 4: CHUYỂN VỊ LẠI BUFFER ẢNH
    // ================================================================
    // Transpose lại để trả về dạng ban đầu (row-major)
    // Chú ý: w và h vẫn đổi chỗ vì ta đang transpose lại
    flip_block(in, out, h, w, c);
}

// Phiên bản chuyên biệt cho 3 passes (biquadratic filter) - tối ưu hơn phiên bản generic
// Tối ưu bằng cách giảm số lần swap và có thể được compiler optimize tốt hơn
template<typename T, Border P>
inline void fast_gaussian_blur(T *& in, T *& out, const int w, const int h, const int c, const float sigma) 
{
    // Tính toán kích thước box kernel cho 3 passes
    int boxes[3];
    sigma_to_box_radius(boxes, sigma, 3);

    // ================================================================
    // BƯỚC 1: THỰC HIỆN 3 LẦN HORIZONTAL BLUR PASSES
    // ================================================================
    // Luân phiên sử dụng in và out để tránh copy không cần thiết
    horizontal_blur<T,P>(in, out, w, h, c, boxes[0]);  // Pass 1: in -> out
    horizontal_blur<T,P>(out, in, w, h, c, boxes[1]);  // Pass 2: out -> in (đảo ngược)
    horizontal_blur<T,P>(in, out, w, h, c, boxes[2]);  // Pass 3: in -> out
    
    // ================================================================
    // BƯỚC 2: CHUYỂN VỊ (TRANSPOSE) BUFFER ẢNH
    // ================================================================
    // Chuyển vị ảnh: out (chứa kết quả 3 passes ngang) -> in (sẽ làm input cho passes dọc)
    flip_block(out, in, w, h, c);
    
    // ================================================================
    // BƯỚC 3: THỰC HIỆN 3 LẦN HORIZONTAL BLUR TRÊN ẢNH ĐÃ TRANSPOSE
    // ================================================================
    // Blur ngang trên ảnh transpose = blur dọc trên ảnh gốc
    // Chú ý: w và h đã đổi chỗ (w_old = h_new, h_old = w_new)
    horizontal_blur<T,P>(in, out, h, w, c, boxes[0]);  // Pass 1 (dọc): in -> out
    horizontal_blur<T,P>(out, in, h, w, c, boxes[1]);  // Pass 2 (dọc): out -> in
    horizontal_blur<T,P>(in, out, h, w, c, boxes[2]);  // Pass 3 (dọc): in -> out
    
    // ================================================================
    // BƯỚC 4: CHUYỂN VỊ LẠI BUFFER ẢNH
    // ================================================================
    // Transpose lại để trả về dạng ban đầu
    flip_block(out, in, h, w, c);
    
    // Hoán đổi con trỏ để kết quả cuối cùng nằm trong buffer out
    std::swap(in, out);    
}

//!
//! \brief Hàm dispatcher template cho fast_gaussian_blur. Template hóa theo kiểu dữ liệu T và border policy P.
//! Đây là hàm chính được expose và nên được sử dụng trong các chương trình.
//!
//! Hàm này chọn phiên bản fast_gaussian_blur phù hợp dựa trên số passes n.
//! Việc dispatch này giúp compiler có thể inline và optimize tốt hơn.
//!
//! \param[in,out] in       Con trỏ tham chiếu đến buffer nguồn (sẽ bị sửa đổi)
//! \param[in,out] out      Con trỏ tham chiếu đến buffer đích (sẽ bị sửa đổi)
//! \param[in] w            Chiều rộng ảnh (image width)
//! \param[in] h            Chiều cao ảnh (image height)
//! \param[in] c            Số kênh màu (image channels)
//! \param[in] sigma        Độ lệch chuẩn Gaussian (Gaussian standard deviation)
//! \param[in] n            Số lần passes, nên > 0 (number of passes, should be > 0)
//!
template<typename T, Border P = kMirror>
void fast_gaussian_blur(T *& in, T *& out, const int w, const int h, const int c, const float sigma, const uint32_t n)
{
    // Dispatch theo số passes để gọi phiên bản template tối ưu tương ứng
    switch(n)
    {
        case 1: fast_gaussian_blur<T,1,P>(in, out, w, h, c, sigma); break;
        case 2: fast_gaussian_blur<T,2,P>(in, out, w, h, c, sigma); break;
        case 3: fast_gaussian_blur<T,  P>(in, out, w, h, c, sigma); break; // Phiên bản chuyên biệt cho 3 passes (tối ưu nhất)
        case 4: fast_gaussian_blur<T,4,P>(in, out, w, h, c, sigma); break;
        case 5: fast_gaussian_blur<T,5,P>(in, out, w, h, c, sigma); break;
        case 6: fast_gaussian_blur<T,6,P>(in, out, w, h, c, sigma); break;
        case 7: fast_gaussian_blur<T,7,P>(in, out, w, h, c, sigma); break;
        case 8: fast_gaussian_blur<T,8,P>(in, out, w, h, c, sigma); break;
        case 9: fast_gaussian_blur<T,9,P>(in, out, w, h, c, sigma); break;
        case 10: fast_gaussian_blur<T,10,P>(in, out, w, h, c, sigma); break;
        default: printf("fast_gaussian_blur with %d passes is not supported yet. Add a specific case if possible or fall back to the generic version.\n", n); break;
        // default: fast_gaussian_blur<T,10>(in, out, w, h, c, sigma, n); break;
    }
}

//!
//! \brief Hàm dispatcher template chính cho fast_gaussian_blur. Template hóa theo kiểu dữ liệu buffer T.
//! Đây là hàm chính được expose và nên được sử dụng trong các chương trình.
//!
//! Hàm này dispatch theo border policy để gọi hàm fast_gaussian_blur tương ứng.
//! Đây là entry point cuối cùng, nhận tất cả các tham số bao gồm border policy.
//!
//! \param[in,out] in       Con trỏ tham chiếu đến buffer nguồn (sẽ bị sửa đổi)
//! \param[in,out] out      Con trỏ tham chiếu đến buffer đích (sẽ bị sửa đổi)
//! \param[in] w            Chiều rộng ảnh (image width)
//! \param[in] h            Chiều cao ảnh (image height)
//! \param[in] c            Số kênh màu (image channels)
//! \param[in] sigma        Độ lệch chuẩn Gaussian (Gaussian standard deviation)
//! \param[in] n            Số lần passes, mặc định = 3 (number of passes, default = 3)
//! \param[in] p            Chính sách xử lý biên: {kExtend, kMirror, kKernelCrop, kWrap}, mặc định = kExtend
//!
template<typename T>
void fast_gaussian_blur(
    T *& in,
    T *& out,
    const int w,
    const int h,
    const int c,
    const float sigma,
    const uint32_t n = 3,
    const Border p = kExtend)
{
    // Dispatch theo border policy để gọi hàm fast_gaussian_blur tương ứng
    switch(p)
    {
        case kExtend:       fast_gaussian_blur<T, kExtend>       (in, out, w, h, c, sigma, n); break;
        case kMirror:       fast_gaussian_blur<T, kMirror>       (in, out, w, h, c, sigma, n); break;
        case kKernelCrop:   fast_gaussian_blur<T, kKernelCrop>   (in, out, w, h, c, sigma, n); break;
        case kWrap:         fast_gaussian_blur<T, kWrap>         (in, out, w, h, c, sigma, n); break;
    }
}