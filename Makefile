# Makefile cho Fast Gaussian Blur với hỗ trợ OpenMP và CUDA

# Compiler và flags
CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -std=c++17
OMPFLAGS = -fopenmp
CUDAFLAGS = -O3 -std=c++17 -arch=sm_60 -Xcompiler -fopenmp

# Default target
all: fastblur

# Build với OpenMP (không có CUDA)
fastblur: main.cpp fast_gaussian_blur_template.h
	$(CXX) main.cpp -o fastblur $(CXXFLAGS) $(OMPFLAGS)

# Build với CUDA và OpenMP
fastblur_cuda: main.cpp fast_gaussian_blur_template.h fast_gaussian_blur_cuda.cu fast_gaussian_blur_cuda.h
	$(NVCC) main.cpp fast_gaussian_blur_cuda.cu -o fastblur_cuda $(CUDAFLAGS) -DUSE_CUDA -lcudart

# Build batch processing version (OpenMP)
batch_main: batch_main.cpp fast_gaussian_blur_template.h
	$(CXX) batch_main.cpp -o batch_main $(CXXFLAGS) $(OMPFLAGS)

# Build batch processing version (CUDA)
batch_main_cuda: batch_main.cpp fast_gaussian_blur_template.h fast_gaussian_blur_cuda.cu fast_gaussian_blur_cuda.h
	$(NVCC) batch_main.cpp fast_gaussian_blur_cuda.cu -o batch_main_cuda $(CUDAFLAGS) -DUSE_CUDA -lcudart

# Build debug version
debug: main.cpp fast_gaussian_blur_template.h
	$(CXX) main.cpp -o fastblur -Og -g -std=c++17

# Build single-threaded (không OpenMP)
single: main.cpp fast_gaussian_blur_template.h
	$(CXX) main.cpp -o fastblur -O3 -std=c++17

# Clean
clean:
	rm -f fastblur fastblur_cuda *.o

.PHONY: all clean debug single fastblur fastblur_cuda
