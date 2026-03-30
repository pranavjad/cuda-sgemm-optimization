CC=nvcc
CUDA_ARCH ?= sm_89
CFLAGS=-O3 -lineinfo -arch=$(CUDA_ARCH)
INCLUDE=-I./cutlass/include -I./cutlass/tools/util/include

all: launch

launch: launch.cu
	$(CC) $(CFLAGS) -o build/launch launch.cu -lcublas $(INCLUDE) 

clean:
	rm -f build/launch
