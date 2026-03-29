## SGEMM:
C = a * A * B + b * C
where A, B, C are matrices
where a, b are scalars

A = M by K
B = K by N
sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);

## Performance Calculations:
For the GEMM we have to do a matmul of two matrices followed by an addition to a matrix.
If all matrices are 4096x4096, we have:
1. Total Flops: 2 * 4096^3 + 4096^2 = 137 GFLOPS
- For every element in C (4096^2) we have to do 4096 Multiply and Add operations to compute
the dot product a row in A and col in C.
- Then we have to do 4096^2 add operations to do element wise addition to old C.
2. Total data to read: 3 * 4096^2 * 4B = 201 MB
- 3 matrices of 4 byte floats
3. Total data to write: 4096^2 * 4B = 67 MB

Minimum memory transfer to/from global mem: 268 MB
RTX 4090 Peak FP32 TFLOPS: 82.6
RTX 4090 Peak bandwidth: 1008 GB/sec

Theoretical minimum compute time: 137e9 / 82.6e12 = 1.65 ms
Theoretical minimum memory time: 268 MB / 1008 GB = 0.265 ms
Compute time is ~5x memory transfer time, so the kernel is compute bound.

## Worklog
lower bound: 0.00192446865 s = 1.92 ms
naive.cu: 0.2163s, 
naive.cu coalesced: 0.0385s
smem_block.cu: 0.0227s
warp_tiling_1d.cu: 0.0247s
warp_tiling_2d.cu: 0.0165s
warp_tiling_2d_vec.cu: 0.0148s

## Notes
[4090 specs](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
### Using cublas to check correctness
```c
cublasSgemm(
    handle,
    CUBLAS_OP_N,
    CUBLAS_OP_N,
    N, M, K,
    &alpha,
    d_B, N,
    d_A, K,
    &beta,
    d_C, N
);
```
1. cublas handle
2. operation for matrix A. OP_N = no transpose, OP_T = transpose
3. operation for matrix B
4. N, M, K => rows of A, columns of B, and inner dimension
5. alpha
6. pointer to A, and leading dim of A
7. pointer to B, and leading dim of B
8. beta
9. pointer to C, leading dim of C
the leading dim is the number of elements to move by +1 on axis 0. If the layout is
row major, that would be the number of elements in a row.


### Kernel 2 - global memory coalescing.
Threads are organized in groups of 32 called warps. Warps are assigned to warp schedulers
which are the physical cores that fetch and issue instructions to gorups of 32 threads.
When a warp stalls (waiting for global memory), the warp scheduler switches to another
available warp. That is why it's a good idea to have a lot of available warps to hide latency.

"Before the Volta architecture, it used to be the case that all threads of a warp were fed from the same instruction stream. On a branch, the threads that didn’t take the branch were inactived using the so-called active mask. However, since Volta, it’s no longer a good idea to rely on this ‘warp-synchronous’ behaviour, as instructions from different branches may be interleaved even for the same threads within a warp."

Global memory coalescing - sequential accesses in global memory by threads in the same warp
can be coalesced into one transaction. The largest transaction is 128 bytes, so if all threads
in a warp access floats that map to a single chunk of 32 floats in memory, that is 1 transaction.

In `naive.cu` we assign x and y like this:
```c
const uint x = blockIdx.x * blockDim.x + threadIdx.x;
const uint y = blockIdx.y * blockDim.y + threadIdx.y;
```
Since our code fetches the x-th row of A and y-th column of B, threads 0 and 1 in the first warp (threadIdx.x = 0, 1) will access
two consecutive rows of A. This is bad because memory layout is row major, so the locations that they access will be offset by K.
We would rather have these threads access the same row in A and consecutive columns in B.

First we switch from a 2d to 1d threadblock: `dim3 block(32, 32)` to `dim3 block(32 * 32)`.
Then we change our assignment to this:
```c
const uint x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE)
const uint y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE)
```
Note that `threadIdx.x / BLOCKSIZE` is equivalent to `threadIdx.y` with 2d threadblocks,
and `threadIdx.x % BLOCKSIZE` is equivalent to `threadIdx.x`.

### Kernel 3 - shared memory cache-blocking
Each block has access to shared memory that it's threads can use to communicate with each other.
Each block in the 4090 has 128KB of shared memory.
With this improvement, each thread still calculates one element of the output C.
We load chunks of A and B into shared memory, do as much work as we can and then do the next one.
For A this chunk slides horizontally, for B it slides vertically. That way each threadblock
ends up calculating a 32x32 chunk of the output.

This gives us a slight improvement. However, looking at the ptx of the inner loop we see this:
```s
ld.shared.f32 	%f10, [%r8];
ld.shared.f32 	%f11, [%r7];
fma.rn.f32 	%f12, %f11, %f10, %f109;
ld.shared.f32 	%f13, [%r8+128];
ld.shared.f32 	%f14, [%r7+4];
fma.rn.f32 	%f15, %f14, %f13, %f12;
ld.shared.f32 	%f16, [%r8+256];
ld.shared.f32 	%f17, [%r7+8];
```
Lots of ld instructions. If we look at warp states:
```
Warp State (All Cycles)
Metric,Current
Stall MIO Throttle,24.09
Stall Long Scoreboard,5.55
Stall Barrier,4.11
Stall Not Selected,3.43
Stall Wait,1.83
Selected,1.00
```
We can see that MIO Throttle is the high meaning our kernel is frequently waiting for shared memory instructions. Another interesting point is that we have a decent amount of Stall not selected which means that the warp was eligible to be scheduled, but a different one got scheduled instead. This means that occupancy isn't really an issue because we have many warps waiting.

Arithmetic intensity: FLOPs/byte of memory loaded.
- high AI = compute bound. We don't need much occupancy to hide memory access latency since each thread does so much computation.
- low AI = memory bound. We need a lot of occupancy to hide memory access latency since each thread doesn't do much computation.

In kernel 3, each issues loads for an entire row of A and col of B.
When each thread calculates multiple values, it can use data already loaded from gmem to smem multiple times.
For example, to calculate a partial 8x8 tile, we just need 16 values from A and B.
Whereas for 1 partial, we need 1 value from a and 1 value from b. 1:2 vs 4:1.

### Kernel 4 - 1d blocktiling for multiple results per thread
If we have each thread compute more output elements in C, then we can rely more on registers and less on SMEM.

This kernel is similar to the one above. We are going to have each block load
a chunk from A and a chunk from B, and be responsible for calculating a chunk from C. The chunk from A will slide horizontally, and from B will slide vertically as before. However now instead of the chunks being square 32x32, they will have dimensions (BM, BK) for the block from A and (BK, BN) for the block from B.

Now each block is responsible for a 64x64 chunk in C. Each thread is responsible for an 8x1 stripe of those values.

Each thread will now be resposible for calculating TM rows in one column of it's threadblock's output block in C. Therefore each warp will calculate TMx32 chunk of the output block.

Let's look at the warp state statistics now:
```
Warp State (All Cycles)
Metric,Current
Stall MIO Throttle,6.76
Stall Long Scoreboard,4.06
Stall Barrier,3.30
Stall Not Selected,1.85
Stall Wait,1.28
Stall Short Scoreboard,1.22
Selected,1.00
```
Significantly less MIO throttling. 
In fact, we can keep increasing our arithmetic intensity as long as our kernel is memory bound.
Right now each thread is computing a vertical stripe of our result, why not have it compute a
square tile of the result. This is the most arithmetically intense way to do it.

### Kernel 5 - Increasing arithmetic intensity via 2d Blocktiling
Note:
A matrix multiplication C = AB Can be thought of as C[i, j] = dot(ith row of A, jth col of B)
or as sum of K inner MxN matrices where each one is the outer product of the kth col of A and
the kth row of B where k goes from 0..K-1.

The basic idea is to compute a grid of 8x8 elements in C per thread.
We will achieve this as follows:
1. The outer loop doesn't change. We still use the threads to load a block from A and B
into SMEM. We slide this block horizontally in A, and vertically in B.
2. We maintain 3 arrays in register file
- float thread_results[TM * TN] - results for this thread
- float tmp_A[TM] - column from A block
- float tmp_N[TM] - row from B block
3. Then in the inner loop we loop over BK the dot product dimension, then we load tmp_A and tmp_B and
accumulate the outer product of tmp_A and tmp_B into thread_results.
Here is the warp state statistics now:
```
Warp State (All Cycles)
Metric,Current
Stall Not Selected,1.63
Stall MIO Throttle,1.29
Selected,1.00
Stall Dispatch Stall,0.86
Stall Short Scoreboard,0.64
Stall Barrier,0.54
Stall Long Scoreboard,0.37
Stall Wait,0.27
Stall Math Pipe Throttle,0.14
```
As you can see, much less MIO Throttling. And now not selected is higher even.
### Kernel 6 - Vectorizing loads to A
Currently, when we load the column from A_shared for the outer product it looks like this:
```c
// Load column from A shared memory
for (int i = 0; i < TM; i++) {
    tmp_A[i] = A_shared[(thread_row_C * TM + i) * BK + dot_idx];
}
// Load row from B shared memory
for (int i = 0; i < TN; i++) {
    tmp_B[i] = B_shared[dot_idx * BN + (thread_col_C * TN + i)];
}
```
The tmp_A loading leads to multiple LDS SASS instructions, while the tmp_B loading is just
two LDS.128 instructions. Ideally we would like the tmp_A loading to use vectorized loads
as well. We can do this by transposing A in shared memory so consecutive positions in memory
are being loaded in this loop.

### Kernel 6 - warp tiling
Now we explicitly express all levels of parallelism

Tiling hierarchy
- blocktiles: each block computes a tile of C (BM*BN)
- warptiles: 4 warps compute one blocktile (WM*WN)
- warp subtiles: warptiles are divided into 4 warp subtiles (WSUBM * WSUBN)
4 warp subtiles are collectively handled by 1 warp! Each thread will touch all 4 subtiles.
- threadtiles: warp subtiles are divided into 32 threadtiles (TM * TN)
each thread handles 4 threadtiles (one in each warp subtile)

1. Loop over K, loading blocks of A and B from gmem --> smem
2. Loop over BK, loading chunks of the A block and B block from smem -> registers
3. Compute the outer product of size TMxTN, 4 times (one for each warp subtile) 
4. Write the results back to gmem.



