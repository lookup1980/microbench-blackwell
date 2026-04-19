NVCC       ?= nvcc
CUDA_GENCODE ?= -arch=native
CXXFLAGS   := -O2 -std=c++17

TARGET     := l2_pointer_chase
SRC        := l2_pointer_chase.cu

.PHONY: all clean run plot

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(CUDA_GENCODE) -o $@ $<

run: $(TARGET)
	mkdir -p results
	./$(TARGET) > results/distance.csv

plot: results/distance.csv
	uv run python3 plot_distance.py results/distance.csv

clean:
	rm -f $(TARGET)
	rm -rf results/
