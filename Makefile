NVCC = nvcc
NVCCFLAGS = -O3 -std=c++17 -Iinclude -Wno-deprecated-gpu-targets

TARGET = sgemm_benchmark
SRC_DIR = src

all: $(TARGET)

$(TARGET): $(SRC_DIR)/benchmark.cu $(SRC_DIR)/sgemm_cpu.cpp
	$(NVCC) $(NVCCFLAGS) $^ -o $@

clean:
	rm -f $(TARGET)