CC=nvcc
CFLAGS=-lineinfo

all: launch

launch: launch.cu
	$(CC) $(CFLAGS) -o build/launch launch.cu -lcublas

clean:
	rm -f launch
