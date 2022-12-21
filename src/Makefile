
SRC = $(wildcard *.cpp)
OBJ = $(SRC:.cpp=.o)
EXE = shibuya

ROCM_PATH ?= /opt/rocm
HIPCC      = ${ROCM_PATH}/bin/hipcc

OFFLOAD_ARCH ?= $(shell ${ROCM_PATH}/bin/rocminfo | grep -m 1 -o -P 'gfx.{0,4}')
OFFLOAD_ARCH := $(if $(OFFLOAD_ARCH),$(OFFLOAD_ARCH),gfx906,gfx908,gfx90a)

CXXFLAGS  = -O3 -std=c++11 -mavx
CXXFLAGS += --offload-arch=${OFFLOAD_ARCH}

CUDA_PATH  ?= /usr/local/cuda
NVCC        = ${CUDA_PATH}/bin/nvcc
CUDA_FLAGS  = --x cu

LIB = -lnuma

rocm: $(OBJ)
	$(HIPCC) $(CXXFLAGS) $(OBJ) $(LIB) -o $(EXE)

.cpp.o:
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

cuda:
	$(NVCC) $(CUDA_FLAGS) $(SRC) $(LIB) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE) *.pdf
