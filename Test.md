# HƯỚNG DẪN TEST 1 ẢNH CHO CẢ 3 TRƯỜNG HỢP

## CÁC CÂU LỆNH TEST

### Cách 1: Dùng Script Tự Động (Khuyến nghị)

**PowerShell:**
```powershell
.\test_single_image.ps1
```

**Hoặc với tham số tùy chỉnh:**
```powershell
.\test_single_image.ps1 test.jpg output 5.0 3 extend
```

**Batch (CMD):**
```cmd
test_single_image.bat
```

**Hoặc với tham số:**
```cmd
test_single_image.bat test.jpg output 5.0 3 extend
```

### Cách 2: Chạy Trực Tiếp

#### Test với fastblur.exe (Sequential + OpenMP)
```cmd
.\fastblur.exe test.jpg output_comparison.png 5.0 3 extend
```

Chương trình này sẽ tự động:
- Chạy Sequential (single-threaded)
- Chạy OpenMP (multi-threaded)
- So sánh kết quả

#### Test với fastblur_cuda.exe (CUDA)
```cmd
.\fastblur_cuda.exe test.jpg output_cuda.png 5.0 3 extend
```

Chương trình này sẽ tự động:
- Chạy Sequential (single-threaded)
- Chạy OpenMP (multi-threaded)
- Chạy CUDA (GPU)
- So sánh cả 3 phiên bản

### Cách 3: Chạy Từng Phiên Bản Riêng

Nếu muốn test từng phiên bản riêng:

**Sequential only:**
```cmd
set OMP_NUM_THREADS=1
.\fastblur.exe test.jpg output_seq.png 5.0 3 extend
```

**OpenMP only:**
```cmd
.\fastblur.exe test.jpg output_omp.png 5.0 3 extend
```

**CUDA only:**
```cmd
.\fastblur_cuda.exe test.jpg output_cuda.png 5.0 3 extend
```

## THAM SỐ

- `test.jpg`: File ảnh input
- `output.png`: File ảnh output
- `5.0`: Sigma (độ mờ Gaussian)
- `3`: Số passes (mặc định = 3)
- `extend`: Border policy (mirror, extend, crop, wrap)

## VÍ DỤ ĐẦY ĐỦ

```cmd
REM Test với ảnh test.jpg, sigma=5.0, 3 passes, border=extend
.\fastblur.exe test.jpg output1.png 5.0 3 extend

REM Test với ảnh khác, sigma=10.0, 5 passes, border=mirror
.\fastblur.exe myimage.jpg output2.png 10.0 5 mirror

REM Test CUDA version
.\fastblur_cuda.exe test.jpg output_cuda.png 5.0 3 extend
```

## KẾT QUẢ

Sau khi chạy, bạn sẽ thấy:
- Bảng so sánh thời gian xử lý
- Tốc độ tăng (speedup) giữa các phiên bản
- File ảnh output đã được blur

## LƯU Ý

1. **fastblur.exe** so sánh Sequential vs OpenMP
2. **fastblur_cuda.exe** so sánh cả 3: Sequential, OpenMP, và CUDA
3. Đảm bảo đã build các executable trước khi test:
   ```cmd
   make fastblur
   make fastblur_cuda
   ```

