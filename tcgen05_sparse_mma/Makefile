NVCC ?= nvcc
CUDA_GENCODE ?= -gencode arch=compute_100a,code=sm_100a
NVCCFLAGS ?= -std=c++17 $(CUDA_GENCODE)

MODE ?= throughput
INSTRUCTION ?= mma
KIND ?= f16
A_TYPE ?= f16
B_TYPE ?= f16
SPARSITY ?= 0
AB_LAYOUT ?= ss
CTA_GROUP ?= 1
MMA_M ?= 128
MMA_N ?= 64
MMA_K ?= 32
MMA_DEPTH ?= 256
SCALE_TYPE ?= ue8m0

MODE_ID := 0
ifeq ($(MODE),latency)
MODE_ID := 1
endif

INSTRUCTION_ID := 0
ifeq ($(INSTRUCTION),ws)
INSTRUCTION_ID := 1
endif

AB_LAYOUT_ID := 0
ifeq ($(AB_LAYOUT),ts)
AB_LAYOUT_ID := 1
endif

KIND_ID := 1
ifeq ($(KIND),tf32)
KIND_ID := 0
endif
ifeq ($(KIND),f16)
KIND_ID := 1
endif
ifeq ($(KIND),f8f6f4)
KIND_ID := 2
endif
ifeq ($(KIND),i8)
KIND_ID := 3
endif
ifeq ($(KIND),mxf8f6f4)
KIND_ID := 4
endif
ifeq ($(KIND),mxf4)
KIND_ID := 5
endif
ifeq ($(KIND),mxf4nvf4)
KIND_ID := 6
endif

SCALE_TYPE_ID := 1
ifeq ($(SCALE_TYPE),ue4m3)
SCALE_TYPE_ID := 0
endif

A_TYPE_CODE := 0
A_BITS := 16
ifeq ($(A_TYPE),tf32)
A_TYPE_CODE := 2
A_BITS := 32
endif
ifeq ($(A_TYPE),f16)
A_TYPE_CODE := 0
A_BITS := 16
endif
ifeq ($(A_TYPE),bf16)
A_TYPE_CODE := 1
A_BITS := 16
endif
ifeq ($(A_TYPE),e4m3)
A_TYPE_CODE := 0
A_BITS := 8
endif
ifeq ($(A_TYPE),e5m2)
A_TYPE_CODE := 1
A_BITS := 8
endif
ifeq ($(A_TYPE),e2m3)
A_TYPE_CODE := 3
A_BITS := 6
endif
ifeq ($(A_TYPE),e3m2)
A_TYPE_CODE := 4
A_BITS := 6
endif
ifeq ($(A_TYPE),e2m1)
A_TYPE_CODE := 5
A_BITS := 4
endif
ifeq ($(A_TYPE),u8)
A_TYPE_CODE := 0
A_BITS := 8
endif
ifeq ($(A_TYPE),s8)
A_TYPE_CODE := 1
A_BITS := 8
endif
ifeq ($(A_TYPE),mxf4)
A_TYPE_CODE := 1
A_BITS := 4
endif
ifeq ($(A_TYPE),nvf4)
A_TYPE_CODE := 1
A_BITS := 4
endif

B_TYPE_CODE := 0
B_BITS := 16
ifeq ($(B_TYPE),tf32)
B_TYPE_CODE := 2
B_BITS := 32
endif
ifeq ($(B_TYPE),f16)
B_TYPE_CODE := 0
B_BITS := 16
endif
ifeq ($(B_TYPE),bf16)
B_TYPE_CODE := 1
B_BITS := 16
endif
ifeq ($(B_TYPE),e4m3)
B_TYPE_CODE := 0
B_BITS := 8
endif
ifeq ($(B_TYPE),e5m2)
B_TYPE_CODE := 1
B_BITS := 8
endif
ifeq ($(B_TYPE),e2m3)
B_TYPE_CODE := 3
B_BITS := 6
endif
ifeq ($(B_TYPE),e3m2)
B_TYPE_CODE := 4
B_BITS := 6
endif
ifeq ($(B_TYPE),e2m1)
B_TYPE_CODE := 5
B_BITS := 4
endif
ifeq ($(B_TYPE),u8)
B_TYPE_CODE := 0
B_BITS := 8
endif
ifeq ($(B_TYPE),s8)
B_TYPE_CODE := 1
B_BITS := 8
endif
ifeq ($(B_TYPE),mxf4)
B_TYPE_CODE := 1
B_BITS := 4
endif
ifeq ($(B_TYPE),nvf4)
B_TYPE_CODE := 1
B_BITS := 4
endif

DEFINES := -DBENCH_MODE=$(MODE_ID) -DINSTRUCTION_ID=$(INSTRUCTION_ID) \
           -DKIND_ID=$(KIND_ID) -DA_TYPE_CODE=$(A_TYPE_CODE) -DB_TYPE_CODE=$(B_TYPE_CODE) \
           -DA_BITS=$(A_BITS) -DB_BITS=$(B_BITS) -DSPARSITY=$(SPARSITY) \
           -DAB_LAYOUT_ID=$(AB_LAYOUT_ID) -DCTA_GROUP=$(CTA_GROUP) \
           -DMMA_M=$(MMA_M) -DMMA_N=$(MMA_N) -DMMA_K=$(MMA_K) \
           -DMMA_DEPTH=$(MMA_DEPTH) -DSCALE_TYPE_ID=$(SCALE_TYPE_ID)

all: tcgen05_sparse_mma.out

tcgen05_sparse_mma.out: tcgen05_sparse_mma.cu
	$(NVCC) $(NVCCFLAGS) $(DEFINES) -o $@ $<

ptx: tcgen05_sparse_mma.cu
	$(NVCC) $(NVCCFLAGS) $(DEFINES) -ptx -o tcgen05_sparse_mma.ptx $<

run: tcgen05_sparse_mma.out
	./tcgen05_sparse_mma.out

report:
	latexmk -pdf -interaction=nonstopmode REPORT.tex

clean:
	rm -f *.out *.ptx *.o

clean-report:
	latexmk -c REPORT.tex

.PHONY: all ptx run report clean clean-report
