NVCC ?= nvcc
CUDA_GENCODE ?= -arch=native
NVCCFLAGS := $(CUDA_GENCODE) -O2 -std=c++17

gpc_query: gpc_query.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f gpc_query

.PHONY: clean
