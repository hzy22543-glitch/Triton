NVCC ?= nvcc
CUDA_ARCH ?= 89
NVCCFLAGS ?= -O2 -lineinfo -std=c++17 -arch=sm_$(CUDA_ARCH)
PYTHON ?= python

.PHONY: all test
all: build/vector_add_check build/vector_bench

build:
	mkdir -p build

build/vector_add_check: src/vector_add.cu | build
	$(NVCC) $(NVCCFLAGS) $< -o $@

build/vector_bench: src/vector_bench.cu | build
	$(NVCC) $(NVCCFLAGS) $< -o $@

test: all
	./build/vector_add_check
	./build/vector_bench 1000 2 5
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

