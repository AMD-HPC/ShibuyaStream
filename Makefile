
ROCM_PATH ?= /opt/rocm
CUDA_PATH ?= /usr/local/cuda
HIPCC      = ${ROCM_PATH}/bin/hipcc
NVCC       = ${CUDA_PATH}/bin/nvcc
HIP_INC    = -I${ROCM_PATH}/include
HIP_LIB    = -L${ROCM_PATH}/lib
CXXFLAGS   = -O3 -std=c++11
ROC_FLAGS  = --amdgpu-target=gfx906,gfx908
NV_FLAGS   = --x cu
LIBS       = -lnuma
SRC        = shibuya.cpp
EXE        = shibuya

rocm:
	$(HIPCC) \
    $(CXXFLAGS) $(ROC_FLAGS) \
    $(HIP_INC) $(HIP_LIB) $(LIBS) \
    $(SRC) -o $(EXE)

cuda:
	$(NVCC) \
	$(CXXFLAGS) $(NV_FLAGS) \
	$(LIBS) \
    $(SRC) -o $(EXE)

run:
	./shibuya

doc:
	pandoc README.md -o README.pdf

clean:
	rm -rf $(EXE) *.pdf
