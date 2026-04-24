# Optimizing a MatMul with CuTe

If you like CUDA kernels, you may be aware of the [canonical blogpost](https://siboehm.com/articles/22/CUDA-MMM) by Simon Boemh in which he iteratively optimizes a matrix multiplication kernel. In this post, I implement these kernels using [CuTe](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html) which is a header only library with useful primitives for expressing tensor layouts and indexing. The CuTe docs are great, but I found myself struggling to understand the inner workings and design choices of CuTe until I implemented Simon's kernels in CuTe. I found that forcing myself to implement Simon's kernels down to the same access/computation pattern but using CuTe's idioms helped me deeply understand the framework, and this post is an attempt to distill and convey that.

My goal with this article is not to explain the matmuls themselves, Simon's article already does a great job at that. Rather, I will go through a few of the most important optimizations and how they would be implemented using CuTe which is sufficient to showcase the framework and learn its core concepts.

## Table of Contents

1. Preliminaries
2. 2D blocktiling
3. 2D blocktiling with vectorized loads
4. Warp tiling

## Preliminaries

The goal with each of these kernels is to do an SGEMM operation with the following operands:

- A: (M, K)
- B: (K, N)

The result is 

- C = alpha * (A @ B) + beta * C, where C is (M, N).

## 2D Blocktiling

This is kernel 5 in Simon's blog, and the first kernel that achieves serious performance. Since the rest of the article assumes understanding of this, let's take a minute to review. The 2D blocktiling kernel can be summarized as follows.

Computation pattern:

- block: calculates a (BM, BN) blocktile of C using a (BM, BK) blocktile from A and a (BK, BN) blocktile from B
- thread: thread calculates a (TM, TN) threadtile of the C blocktile via a mamtul of (TM, BK) @ (BK, TN) threadtiles from A and B respectively.
    - the matmul is done via summing outer product of (TM, 1) slices from A and (1, TN) slices from B

Memory access pattern:

- block: each block loads a blocktiles from A/B from global memory (gmem) to a shared memory buffer (smem)
- thread: each thread loads threadtile slices from A/B blocktiles from gmem to register memory (rmem)

1. threads cooperatively load A, B blocktiles from gmem --> smem
2. each thread loads the threadtile slice from the A/B blocktiles from smem --> rmem
3. each thread does one iteration of outer product accumulation into it's (TM, TN) threadtile
4. Repeat steps 2-3 advancing the slices along BK dimension
5. Repeat steps 1-4 advancing the blocktiles along K dimension
6. Write results from rmem --> gmem

Now, let's implement it using CuTe! The first thing we need to do is declare the kernel. Simon's looks like this:

```cpp
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
```

With CuTe, we need a few more things here and we will quickly see why.

```cpp
template <
    const int BM, const int BN, const int BK, const int TM, const int TN
    class ATiler, class BTiler, class CTiler,
    class AStride, class ASmemLayout, class AThreadLayout,
    class BStride, class BSmemLayout, class BThreadLayout,
    class CStride, class CThreadLayout
>
__global__ void sgemm_2d_block_tiling_cute(
    int M, int N, int K,
    float alpha, const float *A, const float *B, float beta, float *C,
    AStride A_strides, BStride B_strides, CStride C_strides,
    ATiler A_tiler, BTiler B_tiler, CTiler C_tiler,
    ASmemLayout A_shared_layout, BSmemLayout B_shared_layout,
    AThreadLayout A_thread_layout, BThreadLayout B_thread_layout, CThreadLayout C_thread_layout
) {
```

The added template types can be ignored, they exist to capture the complex types CuTe would create so that we don't have to type them out ourselves. For the function arguments, we have a few extra. Namely, we have strides, tilers, and a thread_layout for each matrix. With CuTe, we can use these objects to extract the portions of our inputs and outputs and each thread will read and write to. You will see that overarching theme again and again: CuTe gives us a powerful language to describe shapes, layouts, and access patterns along with an ergenomic set of utilies to extract a given thread's workload from that description. Fundamentally, that's all we really need to write matmuls.

The next part of our reference kernel
1. defines some bookkeeping variables used for indexing calculations later
2. allocates shared memory to load the blocktiles into
3. calculates which blocktile this block handles
```cpp
const uint cRow = blockIdx.y;
const uint cCol = blockIdx.x;

// BN/TN are the number of threads to span a column
const int threadCol = threadIdx.x % (BN / TN);
const int threadRow = threadIdx.x / (BN / TN);

// allocate space for the current blocktile in smem
__shared__ float As[BM * BK];
__shared__ float Bs[BK * BN];

// Move blocktile to beginning of A's row and B's column
A += cRow * BM * K;
B += cCol * BN;
C += cRow * BM * N + cCol * BN;
```
Here's what it looks like with CuTe. A few things to note:
1. Less indexing variables like cRow, threadCol in CuTe. Instead of indexing via complex expressions involving these variables, we defer that calculation to a Layout on a Tensor.
2. Everything we work with is a Tensor. 
```cpp
// create Tensors from data + Layout
Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), A_strides);
Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(K, N), B_strides);
Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), C_strides);

// shared tiles
__shared__ float A_smem[cosize_v<ASmemLayout>];
__shared__ float B_smem[cosize_v<BSmemLayout>];
Tensor sA = make_tensor(make_smem_ptr(A_smem), A_shared_layout);
Tensor sB = make_tensor(make_smem_ptr(B_smem), B_shared_layout);

// global tiles
Tensor gA = local_tile(mA, A_tiler, make_coord(blockIdx.y, _)); // (BM, BK, k)
Tensor gB = local_tile(mB, B_tiler, make_coord(_, blockIdx.x)); // (BK, BN, k)
Tensor gC = local_tile(mC, C_tiler, make_coord(blockIdx.y, blockIdx.x)); // (BM, BN)
```
### Tensors and Layouts
This the first snippet where we see some real CuTe operations. Let's break them down.
- Tensor: a Tensor is data with some Layout
- Layout: a Layout is a pair of shape:stride. The Shape defines the logical dimensions of the Layout's coordinate system. The strides define how many elements you have to skip in the buffer to get to the next element along that dimension. Fundmantally a Layout is just a function that maps a coordinate like (0, 1) to an offset like 1. For example, a row major 2x4 Tensor would have the layout (2, 4):(4, 1). Again, that layout is a function which you can call:
```cpp
// layout - (2, 4):(4, 1)
layout(0, 0) = 0; layout(0, 1) = 1; layout(0, 2) = 2; layout(0, 3) = 3;
layout(1, 0) = 4; layout(1, 1) = 5; layout(1, 2) = 6; layout(1, 3) = 7;
...
```
Or we could visualize a 2D layout like this. Then shows us what 2D coordinate maps to what offset more intuitively.
```
0 1 2 3
4 5 6 7
```
A Tensor is a data buffer with a layout that tells you how tuple coordinates map to offsets in the buffer. So to index a Tensor, CuTe simply uses the Layout to calculate what offset to find the data at instead of us doing it manually.
Now, let's return to the CuTe snippet. Given the context on Tensors and Layouts, we can now understand this portion:
```cpp
// create Tensors from data + Layout
Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), A_strides);
Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(K, N), B_strides);
Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), C_strides);

// shared tiles
__shared__ float A_smem[cosize_v<ASmemLayout>];
__shared__ float B_smem[cosize_v<BSmemLayout>];
Tensor sA = make_tensor(make_smem_ptr(A_smem), A_shared_layout);
Tensor sB = make_tensor(make_smem_ptr(B_smem), B_shared_layout);
```
First we declare `mA, mB, mC` Tensors backed by data buffers A/B/C, and layouts specified by the next 2 arguments giving it's shape and stride. In this case the stride is row major since the last mode has a stride of 1 (mode means corresponding positions in the shape:stride pair). Next we declare `sA, sB` Tensors which will hold the blocktiles in shared memory. This time they are backed by smem pointers, and we specify the layout as a single argument created via make_layout.

Now let's look at that last part of our CuTe snippet.
```cpp
// global tiles
Tensor gA = local_tile(mA, A_tiler, make_coord(blockIdx.y, _)); // (BM, BK, k)
Tensor gB = local_tile(mB, B_tiler, make_coord(_, blockIdx.x)); // (BK, BN, k)
Tensor gC = local_tile(mC, C_tiler, make_coord(blockIdx.y, blockIdx.x)); // (BM, BN)
```
This is where we determine what blocktiles this block actually needs to read. Remember, Simon's kernel achieves this by advancing the A/B/C pointers to the start of the correct blocktile row/col and then incrementing them. With CuTe we don't need to do that manually, we get a nice utility `local_tile` for pulling the part of A/B/C that this block needs to see. This is the CuTe idiom again, declare useful layouts on data and use them to pull this thread/block/warp's workload with the handy utilities.
The local_tile signature looks something like this: `local_tile(tensor, shape, coordinate)`. The function will tile the `tensor` into chunks of shape `shape` and let you yank out one of those tiles (elaboration: local_tile results in some data shaped like this: ((tile_w, tile_h), (rest_w, rest_h)) and the coord indexes into the second "rest" mode).
For C, we use `make_coord(blockIdx.y, blockIdx.x)` to grab the output blocktile this block should compute. For A/B we use `_` in the cordinate to grab all blocktiles along that row/col which we will need to calculate the one output tile. The resulting tensor shapes are mentioned in the comments.

Great, so we've achieved everything our reference kernel does so far. Let's see the next snippets.


