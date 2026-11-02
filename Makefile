# Les commandes commencent par '>' au lieu d'une tabulation (robuste au copier-coller)
.RECIPEPREFIX = >

NVCC ?= nvcc

# Architecture GPU : sm_75 = T4. Surcharger : make ARCH=sm_86
ARCH ?= sm_75

NVCCFLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Iinclude -Wno-deprecated-gpu-targets
LDFLAGS   = -lcublas

SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj

TARGET = sgemm_benchmark

SRCS_CU  = $(SRC_DIR)/benchmark.cu $(SRC_DIR)/sgemm_naive.cu $(SRC_DIR)/sgemm_shared.cu
SRCS_CPP = $(SRC_DIR)/sgemm_cpu.cpp

OBJS = $(patsubst $(SRC_DIR)/%.cu,  $(OBJ_DIR)/%.o, $(SRCS_CU)) \
       $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS_CPP))

# Tout header modifié => recompilation de tous les objets
HEADERS = $(wildcard $(INC_DIR)/*.cuh)

all: $(TARGET)

$(TARGET): $(OBJS)
> $(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS) | $(OBJ_DIR)
> $(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(OBJ_DIR)
> $(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR):
> mkdir -p $@

clean:
> rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean
