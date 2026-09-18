NVCC = nvcc
NVCCFLAGS = -O3 -std=c++17 -Iinclude -Wno-deprecated-gpu-targets
LDFLAGS = -lcublas

TARGET = sgemm_benchmark
SRC_DIR = src

all: $(TARGET)

# Ensure all your .cu and .cpp files are listed here
$(TARGET): $(SRC_DIR)/benchmark.cu $(SRC_DIR)/sgemm_cpu.cpp $(SRC_DIR)/sgemm_naive.cu $(SRC_DIR)/sgemm_shared.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)