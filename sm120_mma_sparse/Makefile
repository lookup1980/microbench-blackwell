NVCC ?= nvcc
CUDA_GENCODE ?= -gencode arch=compute_120,code=sm_120
NVCCFLAGS ?= -std=c++17 -O3 $(CUDA_GENCODE)

MODE ?= throughput
FAMILY ?= f16
A_TYPE ?= f16
B_TYPE ?= f16
SPARSITY ?= 1
ORDERED_METADATA ?= 0
SATFINITE ?= 0
MMA_K ?= 16
ITERATIONS ?= 4096
UNROLL ?= 8
BLOCKS ?= 108
WARPS_PER_BLOCK ?= 8

MODE_ID := 0
ifeq ($(MODE),latency)
MODE_ID := 1
endif

FAMILY_ID := 0
ifeq ($(FAMILY),f16)
FAMILY_ID := 0
endif
ifeq ($(FAMILY),bf16)
FAMILY_ID := 1
endif
ifeq ($(FAMILY),tf32)
FAMILY_ID := 2
endif
ifeq ($(FAMILY),fp8)
FAMILY_ID := 3
endif
ifeq ($(FAMILY),int8)
FAMILY_ID := 4
endif
ifeq ($(FAMILY),int4)
FAMILY_ID := 5
endif
ifeq ($(FAMILY),f16f16)
FAMILY_ID := 6
endif

ATYPE_ID := 0
ifeq ($(A_TYPE),f16)
ATYPE_ID := 0
endif
ifeq ($(A_TYPE),bf16)
ATYPE_ID := 1
endif
ifeq ($(A_TYPE),tf32)
ATYPE_ID := 2
endif
ifeq ($(A_TYPE),e4m3)
ATYPE_ID := 3
endif
ifeq ($(A_TYPE),e5m2)
ATYPE_ID := 4
endif
ifeq ($(A_TYPE),u8)
ATYPE_ID := 5
endif
ifeq ($(A_TYPE),s8)
ATYPE_ID := 6
endif
ifeq ($(A_TYPE),u4)
ATYPE_ID := 7
endif
ifeq ($(A_TYPE),s4)
ATYPE_ID := 8
endif

BTYPE_ID := 0
ifeq ($(B_TYPE),f16)
BTYPE_ID := 0
endif
ifeq ($(B_TYPE),bf16)
BTYPE_ID := 1
endif
ifeq ($(B_TYPE),tf32)
BTYPE_ID := 2
endif
ifeq ($(B_TYPE),e4m3)
BTYPE_ID := 3
endif
ifeq ($(B_TYPE),e5m2)
BTYPE_ID := 4
endif
ifeq ($(B_TYPE),u8)
BTYPE_ID := 5
endif
ifeq ($(B_TYPE),s8)
BTYPE_ID := 6
endif
ifeq ($(B_TYPE),u4)
BTYPE_ID := 7
endif
ifeq ($(B_TYPE),s4)
BTYPE_ID := 8
endif

DEFINES := -DBENCH_MODE=$(MODE_ID) -DFAMILY_ID=$(FAMILY_ID) \
           -DA_TYPE_ID=$(ATYPE_ID) -DB_TYPE_ID=$(BTYPE_ID) \
           -DSPARSITY=$(SPARSITY) -DORDERED_METADATA=$(ORDERED_METADATA) \
           -DSATFINITE=$(SATFINITE) -DMMA_K=$(MMA_K) \
           -DITERATIONS=$(ITERATIONS) -DTHROUGHPUT_UNROLL=$(UNROLL) \
           -DBLOCKS=$(BLOCKS) -DWARPS_PER_BLOCK=$(WARPS_PER_BLOCK)

all: sm120_mma_sparse.out

sm120_mma_sparse.out: sm120_mma_sparse.cu
	$(NVCC) $(NVCCFLAGS) $(DEFINES) -o $@ $<

ptx: sm120_mma_sparse.cu
	$(NVCC) $(NVCCFLAGS) $(DEFINES) -ptx -o sm120_mma_sparse.ptx $<

run: sm120_mma_sparse.out
	./sm120_mma_sparse.out

report:
	latexmk -pdf -interaction=nonstopmode REPORT.tex

summary:
	python3 summarize_results.py results/sm120_mma_full.csv --out-dir results

clean:
	rm -f *.out *.ptx *.o

clean-report:
	latexmk -c REPORT.tex

.PHONY: all ptx run report summary clean clean-report
