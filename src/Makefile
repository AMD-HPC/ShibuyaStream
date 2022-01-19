
SRC = $(wildcard *.cpp)
OBJ = $(SRC:.cpp=.o)
EXE = shibuya

ROCM_PATH ?= /opt/rocm
HIPCC      = ${ROCM_PATH}/bin/hipcc
ROCM_INC   = -I${ROCM_PATH}/include
ROCM_LIB   = -L${ROCM_PATH}/lib
ROCM_FLAGS = --amdgpu-target=gfx906,gfx908

CUDA_PATH ?= /usr/local/cuda
NVCC       = ${CUDA_PATH}/bin/nvcc
NV_FLAGS   = --x cu

CXXFLAGS  = -O3 -std=c++11 -mavx
LIBS      = -lnuma

rocm: $(OBJ)
	$(HIPCC) $(CXXFLAGS) $(ROCM_FLAGS) \
	$(ROCM_INC) $(ROCM_LIB) $(LIBS) \
	$(OBJ) -o $(EXE)

cuda: $(OBJ)
	$(NVCC)  $(CXXFLAGS) \
	$(LIBS) \
	$(OBJ) -o $(EXE)

.cpp.o:
	$(HIPCC) $(CXXFLAGS) $(INC) -c $< -o $@
# 	$(NVCC) $(CXXFLAGS) $(NV_FLAGS) $(INC) -c $< -o $@

doc:
	pandoc README.md -o README.pdf

clean:
	rm -rf $(OBJ) $(EXE) *.pdf
