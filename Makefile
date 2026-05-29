# Makefile for Conservation Spectral SDK — PTX-Native Kernels
# Requires nvcc 11.5+, sm_75+ GPU

NVCC = nvcc
ARCH = -arch=sm_75
NVFLAGS = $(ARCH) -O2 -lineinfo
INCLUDES = -Iinclude

.PHONY: all driver inline test bench clean

all: driver inline test

driver: src/driver.cu src/conservation_kernels.ptx include/conservation_types.h
	$(NVCC) $(NVFLAGS) $(INCLUDES) -o driver src/driver.cu -lcuda
	@echo "Built: driver (PTX driver API)"

inline: src/inline_ptx.cu include/conservation_types.h
	$(NVCC) $(NVFLAGS) $(INCLUDES) -o inline_ptx src/inline_ptx.cu
	@echo "Built: inline_ptx (CUDA C++ with inline PTX)"

test: tests/test_chord.cu include/conservation_types.h
	$(NVCC) $(NVFLAGS) $(INCLUDES) -o test_chord tests/test_chord.cu
	@echo "Built: test_chord"
	./test_chord

bench: inline
	./inline_ptx

clean:
	rm -f driver inline_ptx test_chord
