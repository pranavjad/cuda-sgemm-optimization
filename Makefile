CC=nvcc
CUDA_ARCH ?= sm_89
CFLAGS=-O3 -lineinfo -arch=$(CUDA_ARCH)

all: launch

launch: launch.cu
	$(CC) $(CFLAGS) -o build/launch launch.cu -lcublas

clean:
	rm -f build/launch
