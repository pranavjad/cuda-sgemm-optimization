CC=nvcc
CUDA_ARCH ?= sm_89
CFLAGS=-O3 -lineinfo -arch=$(CUDA_ARCH)
INCLUDE=-I./cutlass/include -I./cutlass/tools/util/include

all: launch bench

launch: launch.cu
	$(CC) $(CFLAGS) -o build/launch launch.cu -lcublas $(INCLUDE) 

bench:
	./build/launch

clean:
	rm -f build/launch
