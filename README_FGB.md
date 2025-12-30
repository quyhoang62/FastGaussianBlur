# LUỒNG HOẠT ĐỘNG CỦA CHƯƠNG TRÌNH FAST GAUSSIAN BLUR

Tài liệu này giải thích chi tiết luồng hoạt động từ đầu đến cuối của chương trình Fast Gaussian Blur với song song hóa.

---

## TỔNG QUAN

Chương trình thực hiện Fast Gaussian Blur trên ảnh bằng cách:
1. Đọc ảnh từ file (sử dụng `stb_image.h`)
2. Xử lý blur với song song hóa (sử dụng `fast_gaussian_blur_template.h`)
3. Lưu ảnh đã blur ra file (sử dụng `stb_image_write.h`)

---

## LUỒNG HOẠT ĐỘNG CHI TIẾT

### BƯỚC 1: KHỞI TẠO VÀ KIỂM TRA THAM SỐ

```
File: main.cpp
Hàm: main(int argc, char * argv[])
```

**Mục đích:** Kiểm tra và parse các tham số từ command line.

**Tham số cần thiết:**
- `argv[1]`: Đường dẫn file ảnh input (ví dụ: `test.jpg`)
- `argv[2]`: Đường dẫn file ảnh output (ví dụ: `result.png`)
- `argv[3]`: Độ mờ Gaussian - sigma (float, > 0, ví dụ: `5.0`)
- `argv[4]`: Số lần blur passes (optional, mặc định = 3)
- `argv[5]`: Border policy (optional, mặc định = "mirror")

**Luồng:**
1. Kiểm tra `argc < 4` → In hướng dẫn và thoát nếu thiếu tham số
2. Parse các tham số tùy chọn (passes, border policy)
3. Chuyển đổi string border policy thành enum `Border`

---

### BƯỚC 2: ĐỌC ẢNH TỪ FILE

```
File: main.cpp (sử dụng stb_image.h)
Hàm: stbi_load()
```

**Mục đích:** Đọc và giải mã (decode) ảnh từ file thành mảng pixel trong bộ nhớ.

**Luồng:**
1. Gọi `stbi_load(argv[1], &width, &height, &channels, 0)`
   - `argv[1]`: đường dẫn file input
   - `width, height, channels`: con trỏ để nhận kích thước và số kênh (output)
   - `0`: giữ nguyên số kênh như trong file
   
2. **Bên trong stbi_load():**
   - Mở file và đọc header để detect định dạng (JPEG, PNG, BMP, ...)
   - Giải mã (decode) ảnh thành raw pixel data
   - Chuyển đổi về định dạng chuẩn: `unsigned char*` với giá trị 0-255
   - Pixel được lưu theo **row-major order** (từ trái sang phải, từ trên xuống)
   - Mỗi pixel có `channels` bytes (1=grayscale, 3=RGB, 4=RGBA)
   - Trả về con trỏ đến mảng pixel, hoặc NULL nếu lỗi

3. **Kết quả:**
   - `image_data`: con trỏ đến mảng pixel 1D
   - `width`: chiều rộng ảnh (pixels)
   - `height`: chiều cao ảnh (pixels)
   - `channels`: số kênh màu (1, 3, hoặc 4)
   - Tổng số phần tử: `width * height * channels`

**Lưu ý:** Bộ nhớ được cấp phát động, phải giải phóng bằng `stbi_image_free()` sau này.

---

### BƯỚC 3: TẠO BUFFER VÀ COPY DỮ LIỆU

```
File: main.cpp
```

**Mục đích:** Tạo các buffer để xử lý blur (input và output buffers riêng biệt).

**Luồng:**
1. Tính kích thước buffer: `size = width * height * channels`

2. Cấp phát bộ nhớ cho 2 buffers:
   - `old_image`: buffer chứa ảnh gốc (input)
   - `new_image`: buffer chứa ảnh đã blur (output)
   
   **Tùy chọn:**
   - Mặc định: `unsigned char` (8-bit, 0-255)
   - Nếu define `USE_FLOAT`: `float` (0.0-1.0)

3. Copy dữ liệu từ `image_data` vào `old_image`:
   ```cpp
   for(i = 0; i < size; i++) {
       old_image[i] = image_data[i];  // Copy trực tiếp hoặc normalize nếu dùng float
   }
   ```

**Tại sao cần 2 buffers riêng?**
- Thuật toán blur cần swap giữa input và output nhiều lần
- Tránh overwrite dữ liệu đang xử lý
- Cho phép song song hóa an toàn

---

### BƯỚC 4: THỰC HIỆN GAUSSIAN BLUR (PHẦN CORE)

```
File: fast_gaussian_blur_template.h
Hàm: fast_gaussian_blur()
```

**Mục đích:** Áp dụng Fast Gaussian Blur với song song hóa.

#### 4.1. TÍNH TOÁN BOX RADIUS

```
Hàm: sigma_to_box_radius()
```

**Luồng:**
1. Nhận `sigma` (độ mờ Gaussian mong muốn) và `n` (số passes)
2. Tính toán bán kính box tối ưu cho mỗi pass:
   - Công thức: `wi = sqrt(12*sigma^2/n + 1)`
   - Làm tròn về số lẻ gần nhất (box filter cần kích thước lẻ)
   - Phân bổ: một số passes dùng box nhỏ, số còn lại dùng box lớn
   - Mục đích: xấp xỉ Gaussian tốt nhất với N lần box blur

3. Kết quả: mảng `boxes[N]` chứa bán kính box cho mỗi pass

**Lý thuyết:**
- Nhiều lần box blur → hội tụ về Gaussian (định lý Central Limit)
- 3 passes thường đủ cho chất lượng tốt
- Box blur rất nhanh vì có thể dùng sliding window (O(n) thay vì O(n*r))

---

#### 4.2. BLUR THEO CHIỀU NGANG (N LẦN)

```
Hàm: horizontal_blur() (gọi N lần)
```

**Luồng cho mỗi pass:**

1. **Gọi `horizontal_blur()`:**
   ```cpp
   horizontal_blur<T,P>(in, out, w, h, c, boxes[i]);
   ```

2. **Bên trong `horizontal_blur()`:**
   
   **a) Dispatch theo border policy và kích thước kernel:**
   - Chọn hàm phù hợp: `horizontal_blur_extend/mirror/crop/wrap`
   - Chọn loại kernel: `kSmall` (r < w/2), `kMid` (w/2 <= r < w), `kLarge` (r >= w)
   
   **b) Song song hóa - OpenMP:**
   ```cpp
   #pragma omp parallel for
   for(int i = 0; i < h; i++) {  // Duyệt qua từng hàng
       // Xử lý hàng i
   }
   ```
   
   **Cách hoạt động song song hóa:**
   - OpenMP tự động chia `h` hàng thành nhiều nhóm
   - Mỗi thread (CPU core) xử lý một nhóm hàng
   - Ví dụ: 1000 hàng, 4 threads → mỗi thread xử lý ~250 hàng
   - Các hàng độc lập → không có race condition
   - Load balancing tự động
   
   **c) Xử lý mỗi hàng với sliding window:**
   
   ```cpp
   // Khởi tạo accumulator cho kernel đầu tiên
   for(j = ti; j < ri; j++) {
       acc[ch] += in[j*C+ch];  // Tích lũy giá trị pixel
   }
   
   // Trượt kernel sang phải (sliding window)
   for(mỗi pixel trong hàng) {
       // Cập nhật accumulator: + pixel mới, - pixel cũ
       acc[ch] += in[ri*C+ch] - in[li*C+ch];
       // Tính giá trị blur: trung bình
       out[ti*C+ch] = acc[ch] * iarr + round_v<T>();
   }
   ```
   
   **Tại sao sliding window nhanh?**
   - Thông thường: O(n*r) - tính lại tổng cho mỗi pixel
   - Với sliding window: O(n) - chỉ cập nhật O(1) mỗi pixel
   - Ví dụ: kernel r=10, ảnh 1000x1000
     - Thông thường: 1000*1000*10 = 10 triệu phép tính
     - Sliding window: 1000*1000 + 1000*10 = ~1 triệu phép tính
   
   **d) Xử lý biên (border handling):**
   - **Extend**: Lấy giá trị pixel gần nhất tại biên
   - **Mirror**: Phản chiếu ảnh như gương
   - **Crop**: Chỉ tính trên phần kernel nằm trong ảnh
   - **Wrap**: Bao quanh như torus

3. **Hoán đổi buffers:**
   ```cpp
   std::swap(in, out);  // Output của pass này → input của pass tiếp theo
   ```

4. **Lặp lại N lần** (thường N=3)

**Kết quả sau N passes ngang:** Ảnh đã được blur theo chiều ngang N lần.

---

#### 4.3. CHUYỂN VỊ (TRANSPOSE) ẢNH

```
Hàm: flip_block()
```

**Mục đích:** Transpose ảnh để biến blur ngang thành blur dọc.

**Luồng:**

1. **Gọi `flip_block()`:**
   ```cpp
   flip_block(in, out, w, h, c);
   ```

2. **Song song hóa với collapse(2):**
   ```cpp
   #pragma omp parallel for collapse(2)
   for(int x = 0; x < w; x += block) {      // Vòng lặp ngoài
       for(int y = 0; y < h; y += block) {  // Vòng lặp trong
           // Transpose block tại (x, y)
       }
   }
   ```
   
   **Cách hoạt động collapse(2):**
   - OpenMP làm phẳng 2 vòng lặp thành 1 vòng lặp lớn
   - Ví dụ: w=1000, h=800, block=256
     - Vòng ngoài: 4 iterations (x = 0, 256, 512, 768)
     - Vòng trong: 4 iterations (y = 0, 256, 512, 768)
     - Tổng: 4*4 = 16 block combinations
     - Với 4 threads: mỗi thread xử lý ~4 blocks
   
   **Tại sao transpose theo block?**
   - Cache-friendly: mỗi block nhỏ vừa với cache
   - Giảm cache miss so với transpose toàn bộ một lúc
   - Song song hóa hiệu quả hơn với nhiều tasks nhỏ

3. **Transpose từng block:**
   - Copy pixel từ vị trí (y, x) trong input → vị trí (x, y) trong output
   - Chuyển từ row-major → column-major (hoặc ngược lại)

4. **Hoán đổi buffers:**
   ```cpp
   std::swap(in, out);
   ```

**Kết quả:** Ảnh đã được transpose, blur ngang giờ = blur dọc trên ảnh gốc.

---

#### 4.4. BLUR THEO CHIỀU DỌC (THỰC CHẤT LÀ BLUR NGANG TRÊN ẢNH ĐÃ TRANSPOSE)

```
Hàm: horizontal_blur() (gọi N lần trên ảnh đã transpose)
```

**Luồng:**

1. **Lặp lại N lần blur ngang:**
   ```cpp
   for(int i = 0; i < N; i++) {
       horizontal_blur<T,P>(in, out, h, w, c, boxes[i]);
       // Chú ý: w và h đã đổi chỗ vì ảnh đã transpose
       std::swap(in, out);
   }
   ```

2. **Song song hóa tương tự như bước 4.2**

3. **Kết quả:** Ảnh đã được blur theo cả 2 chiều (ngang và dọc)

---

#### 4.5. TRANSPOSE LẠI ĐỂ TRỞ VỀ DẠNG BAN ĐẦU

```
Hàm: flip_block()
```

**Luồng:**
1. Transpose lại ảnh (w và h đã đổi chỗ)
2. Hoán đổi buffers nếu cần để kết quả cuối cùng nằm trong buffer đúng

**Kết quả:** Ảnh đã được blur hoàn chỉnh, trở về dạng row-major ban đầu.

---

### BƯỚC 5: ĐO THỜI GIAN XỬ LÝ

```
File: main.cpp
```

**Luồng:**
1. Lấy thời gian bắt đầu: `auto start = std::chrono::system_clock::now()`
2. Thực hiện blur (Bước 4)
3. Lấy thời gian kết thúc: `auto end = std::chrono::system_clock::now()`
4. Tính thời gian: `elapsed = duration_cast<milliseconds>(end-start).count()`
5. In ra: `printf("Time %.4fms\n", elapsed)`

**Mục đích:** Đánh giá hiệu năng của thuật toán và song song hóa.

---

### BƯỚC 6: COPY KẾT QUẢ VỀ image_data

```
File: main.cpp
```

**Mục đích:** Copy ảnh đã blur từ buffer về `image_data` để lưu file.

**Luồng:**
```cpp
for(i = 0; i < size; i++) {
    image_data[i] = (uchar)(new_image[i]);  // Copy trực tiếp
    // Hoặc nếu dùng float: image_data[i] = (uchar)(new_image[i] * 255.f);
}
```

**Lưu ý:** 
- `new_image` chứa kết quả blur cuối cùng
- Cần convert nếu dùng float (0.0-1.0) về uchar (0-255)

---

### BƯỚC 7: LƯU ẢNH RA FILE

```
File: main.cpp (sử dụng stb_image_write.h)
Hàm: stbi_write_png/bmp/jpg()
```

**Mục đích:** Lưu ảnh đã blur ra file.

**Luồng:**

1. **Xác định định dạng file từ extension:**
   ```cpp
   std::string ext = file.substr(file.size()-3);  // Lấy 3 ký tự cuối (.png, .jpg, .bmp)
   ```

2. **Gọi hàm write tương ứng:**
   
   **PNG:**
   ```cpp
   stbi_write_png(filename, width, height, channels, image_data, stride);
   ```
   - `stride = channels * width`: khoảng cách giữa các hàng (bytes)
   - Hỗ trợ nén lossless
   - Hỗ trợ alpha channel
   
   **BMP:**
   ```cpp
   stbi_write_bmp(filename, width, height, channels, image_data);
   ```
   - Không nén (uncompressed)
   - Không hỗ trợ alpha trong output
   
   **JPEG:**
   ```cpp
   stbi_write_jpg(filename, width, height, channels, image_data, quality);
   ```
   - Nén lossy (mất mát dữ liệu)
   - `quality`: 1-100 (cao hơn = tốt hơn nhưng file lớn hơn)
   - Bỏ qua alpha channel
   
   **Mặc định:** Nếu extension không được hỗ trợ, chuyển sang PNG.

3. **Bên trong hàm write:**
   - Mở file để ghi
   - Encode pixel data theo định dạng file
   - Nén (nếu có: PNG deflate, JPEG DCT)
   - Ghi header và data vào file
   - Đóng file
   - Trả về 0 nếu lỗi, non-zero nếu thành công

---

### BƯỚC 8: GIẢI PHÓNG BỘ NHỚ

```
File: main.cpp
```

**Mục đích:** Giải phóng tất cả bộ nhớ đã cấp phát để tránh memory leak.

**Luồng:**
```cpp
stbi_image_free(image_data);  // Giải phóng ảnh từ stbi_load()
delete[] new_image;           // Giải phóng buffer output
delete[] old_image;           // Giải phóng buffer input
```

**Lưu ý:** Rất quan trọng để tránh memory leak, đặc biệt khi xử lý nhiều ảnh.

---

## SƠ ĐỒ LUỒNG HOẠT ĐỘNG TỔNG QUÁT

```
┌─────────────────────────────────────────────────────────────┐
│ 1. KHỞI TẠO                                                 │
│    - Parse command line arguments                           │
│    - Validate parameters                                    │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. ĐỌC ẢNH (stb_image.h)                                    │
│    stbi_load() → decode image → pixel array                 │
│    - width, height, channels                                │
│    - image_data: unsigned char*                             │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. TẠO BUFFER                                                │
│    - Allocate old_image (input buffer)                      │
│    - Allocate new_image (output buffer)                     │
│    - Copy image_data → old_image                            │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. GAUSSIAN BLUR (CORE - VỚI SONG SONG HÓA)                 │
│    ┌────────────────────────────────────────────────────┐   │
│    │ 4.1. Tính box radius từ sigma                     │   │
│    └──────────────────┬─────────────────────────────────┘   │
│                       │                                      │
│                       ▼                                      │
│    ┌────────────────────────────────────────────────────┐   │
│    │ 4.2. BLUR NGANG (N lần)                            │   │
│    │      #pragma omp parallel for                      │   │
│    │      - Mỗi hàng xử lý bởi 1 thread                 │   │
│    │      - Sliding window: O(n)                        │   │
│    └──────────────────┬─────────────────────────────────┘   │
│                       │                                      │
│                       ▼                                      │
│    ┌────────────────────────────────────────────────────┐   │
│    │ 4.3. TRANSPOSE                                     │   │
│    │      #pragma omp parallel for collapse(2)          │   │
│    │      - Mỗi block xử lý bởi 1 thread                │   │
│    │      - Block-based: cache-friendly                 │   │
│    └──────────────────┬─────────────────────────────────┘   │
│                       │                                      │
│                       ▼                                      │
│    ┌────────────────────────────────────────────────────┐   │
│    │ 4.4. BLUR DỌC (N lần - thực chất blur ngang        │   │
│    │           trên ảnh đã transpose)                   │   │
│    │      - Song song hóa tương tự 4.2                  │   │
│    └──────────────────┬─────────────────────────────────┘   │
│                       │                                      │
│                       ▼                                      │
│    ┌────────────────────────────────────────────────────┐   │
│    │ 4.5. TRANSPOSE LẠI                                 │   │
│    │      - Trở về dạng ban đầu                         │   │
│    └──────────────────┬─────────────────────────────────┘   │
└───────────────────────┼──────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. ĐO THỜI GIAN                                              │
│    - Tính elapsed time                                      │
│    - Print: "Time X.XXXXms"                                 │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. COPY KẾT QUẢ                                             │
│    - Copy new_image → image_data                            │
│    - Convert nếu cần (float → uchar)                        │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. LƯU ẢNH (stb_image_write.h)                              │
│    stbi_write_png/bmp/jpg() → encode → save file            │
│    - Detect format từ extension                             │
│    - Compress (nếu có)                                      │
│    - Write to disk                                           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. GIẢI PHÓNG BỘ NHỚ                                        │
│    - stbi_image_free(image_data)                            │
│    - delete[] new_image                                     │
│    - delete[] old_image                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## CHI TIẾT VỀ SONG SONG HÓA

### 1. SONG SONG HÓA HORIZONTAL BLUR

**Cách thức:**
- `#pragma omp parallel for` trước vòng lặp `for(int i = 0; i < h; i++)`
- Mỗi hàng được xử lý độc lập bởi một thread

**Ví dụ:**
```
Ảnh: 1920x1080 pixels
CPU: 8 cores
Threads: 8

Phân chia:
- Thread 0: Hàng 0-134
- Thread 1: Hàng 135-269
- Thread 2: Hàng 270-404
- ...
- Thread 7: Hàng 945-1079
```

**Lợi ích:**
- Tăng tốc ~8x trên CPU 8 cores (lý thuyết)
- Cache-friendly: mỗi hàng nằm liên tiếp trong bộ nhớ
- Không có race condition: mỗi thread ghi vào vùng nhớ riêng

---

### 2. SONG SONG HÓA TRANSPOSE

**Cách thức:**
- `#pragma omp parallel for collapse(2)` cho 2 vòng lặp lồng nhau
- Mỗi block được transpose bởi một thread

**Ví dụ:**
```
Ảnh: 1000x800 pixels
Block size: 256
CPU: 4 cores

Blocks:
- X: 0, 256, 512, 768 (4 blocks)
- Y: 0, 256, 512, 768 (4 blocks)
- Tổng: 4*4 = 16 blocks

Phân chia:
- Thread 0: Blocks 0-3
- Thread 1: Blocks 4-7
- Thread 2: Blocks 8-11
- Thread 3: Blocks 12-15
```

**Lợi ích:**
- Block-based approach giữ cache locality tốt
- Nhiều tasks hơn → load balancing tốt hơn
- Tăng tốc đáng kể so với sequential transpose

---

## ĐỘ PHỨC TẠP THỜI GIAN

### Sequential (không song song hóa):
- Horizontal blur: O(w * h * r * N) = O(pixels * r * N)
- Transpose: O(w * h)
- **Tổng:** O(pixels * r * N) với sliding window = O(pixels * N)

### Parallel (với song song hóa):
- Horizontal blur: O(w * h * N / num_threads) = O(pixels * N / cores)
- Transpose: O(w * h / num_threads) = O(pixels / cores)
- **Tổng:** O(pixels * N / cores)

**Tăng tốc lý thuyết:** ~num_cores lần

**Ví dụ thực tế:**
- Ảnh 2000x2000 pixels, sigma=5, N=3
- CPU: 8 cores
- Sequential: ~50ms
- Parallel: ~7ms
- **Tăng tốc: ~7x**

---

## LƯU Ý QUAN TRỌNG

1. **Memory Management:**
   - Phải giải phóng tất cả buffers
   - `stbi_image_free()` cho ảnh từ stb_image
   - `delete[]` cho buffers tự cấp phát

2. **Thread Safety:**
   - Mỗi thread xử lý vùng nhớ riêng → không có race condition
   - OpenMP đảm bảo thread safety

3. **Compilation:**
   - Phải compile với `-fopenmp` (GCC/Clang) hoặc `/openmp` (MSVC)
   - Link với OpenMP library

4. **Performance:**
   - Song song hóa chỉ hiệu quả với ảnh đủ lớn (>100x100 pixels)
   - Với ảnh nhỏ, overhead có thể lớn hơn lợi ích

5. **Border Handling:**
   - Các chính sách border khác nhau cho kết quả khác nhau ở biên ảnh
   - Mirror thường cho kết quả tốt nhất

---

## KẾT LUẬN

Chương trình Fast Gaussian Blur sử dụng:
- **Thuật toán:** Nhiều lần box blur để xấp xỉ Gaussian (O(n) với sliding window)
- **Song song hóa:** OpenMP để tận dụng CPU đa nhân
- **Thư viện:** stb_image để I/O ảnh (header-only, dễ sử dụng)

Kết quả: Xử lý blur rất nhanh, phù hợp cho ứng dụng real-time với ảnh độ phân giải hợp lý.


