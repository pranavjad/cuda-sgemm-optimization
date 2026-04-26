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
The local_tile signature looks something like this: `local_tile(tensor, shape, coordinate)`. The function will tile the `tensor` into chunks of shape `shape` and let you yank out one of those tiles (elaboration: local_tile results in some data shaped like this: `((tile_w, tile_h), (rest_w, rest_h))` and the coord indexes into the second "rest" mode, TODO: add vis).
For C, we use `make_coord(blockIdx.y, blockIdx.x)` to grab the output blocktile this block should compute. For A/B we use `_` in the cordinate to grab all blocktiles along that row/col which we will need to calculate the one output tile. This results in a 3D shape for A/B where the first 2 dimensions represent the shape of one blocktile, and the last dimension represents the number of blocktiles along K. The resulting tensor shapes are mentioned in the comments.

Great, so we've achieved everything our reference kernel does so far. Let's see the next snippet from the reference kernel.
1. calculate each thread's share of the blocktile it should load from gmem to smem
2. declare register memory for storing computation results, and for staging parts of the threadtiles for computation.
```cpp
// calculating the indices that this thread will load into SMEM
const uint innerRowA = threadIdx.x / BK;
const uint innerColA = threadIdx.x % BK;
// calculates the number of rows of As that are being loaded in a single step
// by a single block
const uint strideA = numThreadsBlocktile / BK;
const uint innerRowB = threadIdx.x / BN;
const uint innerColB = threadIdx.x % BN;
// for both As and Bs we want each load to span the full column-width, for
// better GMEM coalescing (as opposed to spanning full row-width and iterating
// across columns)
const uint strideB = numThreadsBlocktile / BN;

// allocate thread-local cache for results in registerfile
float threadResults[TM * TN] = {0.0};
// register caches for As and Bs
float regM[TM] = {0.0};
float regN[TN] = {0.0};
```

Here's what is looks like with CuTe. In CuTe we actually do a little more up front work, rather than doing complex indexing later which will become clear as we see the main loops of each kernel.
```cpp
// part of gA each thread loads to sA
Tensor gA_to_r = local_partition(gA, A_thread_layout, threadIdx.x);
Tensor sA_to_w = local_partition(sA, A_thread_layout, threadIdx.x);

// part of gB each thread loads to sB
Tensor gB_to_r = local_partition(gB, B_thread_layout, threadIdx.x);
Tensor sB_to_w = local_partition(sB, B_thread_layout, threadIdx.x);

// part of sA, sB each thread reads for computation
auto thread_row_C = threadIdx.x / (BN / TN);
auto thread_col_C = threadIdx.x % (BN / TN);
auto A_col_shape = make_shape(TM, 1);
auto B_row_shape = make_shape(1, TN);
Tensor sA_to_r = local_tile(sA, A_col_shape, make_coord(thread_row_C, _)); // (TM, 1, BK)
Tensor sB_to_r = local_tile(sB, B_row_shape, make_coord(_, thread_col_C)); // (1, TN, BK)

// part of gC each thread writes results
Tensor gC_to_w = local_tile(gC, shape(C_thread_layout), make_coord(thread_row_C, thread_col_C)); // (TM, TN)

// rmem
Tensor thread_results = make_tensor_like(gC_to_w);
clear(thread_results);
Tensor tmp_A = make_tensor_like<float>(make_layout(make_shape(TM)));
Tensor tmp_B = make_tensor_like<float>(make_layout(make_shape(TN)));
```
Again at a high level we achieve the same thing:
1. Calculate which part of gmem blocktiles each thread should load to smem.
2. declare rmem for storing computation results.
But additionally we do the following up front, which Simon's kernel does implicitly later via indexing expressions:
3. Calculate which parts of the smem blocktiles the thread needs to read in order to calculate it's threadtile.
4. Calculate which parts of global memory this thread should write it's results to.
In Simon's kernel we also do this, it just shows up as an indexing calculation later on in the hot loop, like `sA[calculation]`. Since CuTe is about avoiding these manual indexing calculations, and doing everything with layouts we do this work up front.


Let's take a closer look.
```cpp
// part of gA each thread loads to sA
Tensor gA_to_r = local_partition(gA, A_thread_layout, threadIdx.x);
Tensor sA_to_w = local_partition(sA, A_thread_layout, threadIdx.x);

// part of gB each thread loads to sB
Tensor gB_to_r = local_partition(gB, B_thread_layout, threadIdx.x);
Tensor sB_to_w = local_partition(sB, B_thread_layout, threadIdx.x);
```
The `local_partition` function is similar to the `local_tile` function, but rather than letting you pick out a tile by coordinate, it lets you pick out an element of each tile. Formally, `local_partition(tensor, layout, index)` tiles `tensor` according to `shape(layout)` producing a tensor of shape `((tile_w, tile_h), (rest_w, rest_h))` as we saw before. However instead of indexing into mode 1 which would be grabbing one tile, it indexes into mode 0. For example `local_partition(tensor, layout, 0)` grabs the first element from each tile giving you a tensor of shape `(rest_w, rest_h)` (TODO: add vis here).

One more thing you may have noticed. In order to decide which element of the tile to grab, we give it a 1D index. But the tile shape is usually ND. How does cute convert between the two? That is a design choice of the framework, and CuTe chooses to do it via colexicographic ordering, which they explain in detail in the docs [here](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#coordinate-mapping). Basically, its column major so for a 2x2 tensor the 1D to 2D coordinate mapping would look like `0 -> (0, 0), 1 -> (1, 0), 2 -> (0, 1), 3 -> (1, 1)`.

Now that we understand `local_partition`, it is clear what this code does. For gA, and sA it simply tiles the tensors into tiles equal to the size that all the thread can load in a go, and gives each thread an element in each of those tiles. So looping over `gA_to_r` for example would yield all elements that a given thread is responsible for loading. We will see that this ends up with a much cleaner loop definition later on, whereas in the reference kernel we have to deal with a lot more manual bookkeeping such as `strideA` and `strideB` to know much we can load at once, and how to advance the iterator, etc.

One more thing to note. The shape of `gA` is 3D as we saw before, the first 2D being the blocktile shape and the last dimension being the number of blocktiles in a row. `local_tile` only operates on the first two modes and preserves the last meaning that the shape of `gA_to_r` is also 3D.

Next up.
```cpp
// part of sA, sB each thread reads for computation
auto thread_row_C = threadIdx.x / (BN / TN);
auto thread_col_C = threadIdx.x % (BN / TN);
auto A_col_shape = make_shape(TM, 1);
auto B_row_shape = make_shape(1, TN);
Tensor sA_to_r = local_tile(sA, A_col_shape, make_coord(thread_row_C, _)); // (TM, 1, BK)
Tensor sB_to_r = local_tile(sB, B_row_shape, make_coord(_, thread_col_C)); // (1, TN, BK)

// part of gC each thread writes results
Tensor gC_to_w = local_tile(gC, shape(C_thread_layout), make_coord(thread_row_C, thread_col_C)); // (TM, TN)
```
Now since we want to grab contiguous blocks, we're back to using local tile (TODO: insert vis here). Here we grab the slices of rows/cols from A/B that each thread needs to calculate its threadtile. Additionally we grab the tile of gC the thread should write it's results to.

Finally declaring the register memory in CuTe looks a little different but is functionally the same.
```cpp
Tensor thread_results = make_tensor_like(gC_to_w);
clear(thread_results);
Tensor tmp_A = make_tensor_like<float>(make_layout(make_shape(TM)));
Tensor tmp_B = make_tensor_like<float>(make_layout(make_shape(TN)));
```
The `make_tensor_like` function declares a buffer in rmem and then makes a tensor backed by that buffer with a given layout.

Great, we have the entire preamble of our reference kernel implemented in CuTe. Now we can look at the interesting part, the hot loop.
Here it is from our reference kernel. Conceptually this does:
1. load blocktile gmem -> smem
2. load chunk of blocktile this thread needs for it's calculations from smem -> rmem
3. acculumate one outer product into rmem
4. repeat 2-3 BK times, resulting in the matmul for one blocktile done.
4. repeat 1-4 sliding blocktiles along K.
```cpp
// outer-most loop over block tiles
for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
        As[(innerRowA + loadOffset) * BK + innerColA] =
            A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
        Bs[(innerRowB + loadOffset) * BN + innerColB] =
            B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
        // block into registers
        for (uint i = 0; i < TM; ++i) {
            regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
        }
        for (uint i = 0; i < TN; ++i) {
            regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
        }
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                threadResults[resIdxM * TN + resIdxN] +=
                regM[resIdxM] * regN[resIdxN];
            }
        }
    }
    __syncthreads();
}
```
Here it is in CuTe.
```cpp
auto max_tile_idx = shape<2>(gA);
for (int tile_idx = 0; tile_idx < max_tile_idx; tile_idx++) {
    // load tiles
    Tensor gA_tile = gA_to_r(_, _, tile_idx);
    CUTE_UNROLL
    for (int i = 0; i < size(gA_tile); i++) {
        sA_to_w(i) = gA_tile(i);
    }
    Tensor gB_tile = gB_to_r(_, _, tile_idx);
    CUTE_UNROLL
    for (int i = 0; i < size(gB_tile); i++) {
        sB_to_w(i) = gB_tile(i);
    }

    __syncthreads();

    // compute partial results for this tile
    CUTE_UNROLL
    for (int dot_idx = 0; dot_idx < shape<2>(sA_to_r); dot_idx++) {
        // load row/col from smem to rmem
        Tensor sA_col = sA_to_r(_, _, dot_idx);
        CUTE_UNROLL
        for (int i = 0; i < size(tmp_A); i++) {
            tmp_A(i) = sA_col(i);
        }
        Tensor sB_row = sB_to_r(_, _, dot_idx);
        CUTE_UNROLL
        for (int i = 0; i < size(tmp_B); i++) {
            tmp_B(i) = sB_row(i);
        }

        // outer product
        CUTE_UNROLL
        for (int i = 0; i < shape<0>(thread_results); i++) {
            CUTE_UNROLL
            for (int j = 0; j < shape<1>(thread_results); j++) {
                thread_results(i, j) += tmp_A(i) * tmp_B(j);
            }
        }
    }
    __syncthreads();
}
```
TODO: above two code snippets should be shown side by side in the article.
A few things to take note of:
- tensors are indexed via `tensor(i)` rather than `tensor[i]` as you may be used to seeing.
- loop bounds are determined by the shape of tensors. For example, the outer loop `dot_idx` ranges from `0..shape<2>(gA)`. In this case `shape(gA) = (BM, BK, k)` where `k = K / BK` or the number of blocktiles along a row of gA. Then we select the correct tile by indexing into the last mode of `gA_to_r`. Simon's kernel achieves this keeping points to the start of the right tile, and advancing the pointers throughout the loop.
- indexing is trivial since everything is handled by tensors and layouts we defined earlier. In the entire CuTe snippet you will not see a single indexing calculation inside the parenthesis, unlike the reference kernel. Let's zoom in on one loop:
```cpp
Tensor gA_tile = gA_to_r(_, _, tile_idx);
CUTE_UNROLL
for (int i = 0; i < size(gA_tile); i++) {
    sA_to_w(i) = gA_tile(i);
}
```
Notice rather than complex indexing expressions, we've defined two tensors `sA_to_w` and `gA_tile` which represent the source and destination of our copy and they are of equal size. So all that's left to do is iterate over the size of the tensor and issue each copy instruction. Above, I wrote it explicitly but in CuTe there is actually a macro for this:
```
Tensor gA_tile = gA_to_r(_, _, tile_idx);
copy(gA_tile, sA_to_w);
```
which would expand to the code above, and even do things like vectorize loads if possible (more on this later).

We see the same pattern again when loading smem -> rmem
```cpp
Tensor sA_col = sA_to_r(_, _, dot_idx);
CUTE_UNROLL
for (int i = 0; i < size(tmp_A); i++) {
    tmp_A(i) = sA_col(i);
}
```
For comparison here is what it looks like in the reference kernel:
```cpp
for (uint i = 0; i < TM; ++i) {
    regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
}
```
The indexing expression has two parts
- `(threadRow * TM + i) * BK`: this selects the correct row from `sA` which increments with `i` since we are moving down a col of sA. In CuTe this is handled in the `local_tile` when tiling sA.
- `+ dotIdx`:  this selects the col we are copying. `sA_to_r` is of shape `(TM, 1, BK)`, so indexing into the last mode will select one thread col.

And finally, the outer product loop is basically the same but we get some nicer indexing ergenomics from CuTe again.

Now we arrive at the last snippet: storing the results back to gmem.
Simon's kernel:
```cpp
// write out the results
for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
        C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
            alpha * threadResults[resIdxM * TN + resIdxN] +
            beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
}
```
Cute:
```cpp
// write results back to gmem
CUTE_UNROLL
for (int i = 0; i < shape<0>(thread_results); i++) {
    CUTE_UNROLL
    for (int j = 0; j < shape<1>(thread_results); j++) {
        gC_to_w(i, j) = alpha * thread_results(i, j) + beta * gC_to_w(i, j);
    }
}
```
Again, much nicer indexing courtesy of the up front work we did defining layouts. Instead of indexing directly into C, we index into the chunk of C that this thread should handle according to our layout `gC_to_w`. Recalling how `gC_to_w` was defined we can see how it is ultimately equivalent to the indexing expression from Simon's kernel.
```cpp
Tensor gC = local_tile(mC, C_tiler, make_coord(blockIdx.y, blockIdx.x)); // (BM, BN)
Tensor gC_to_w = local_tile(gC, shape(C_thread_layout), make_coord(thread_row_C, thread_col_C)); // (TM, TN)
```
First we created `gC` which is the (BM, BN) blocktile from `C` that this block needs to handle. Within Simon's kernel this manifests as:
`C += cRow * BM * N + cCol * BN;`

Then within `gC` we further tile into the threadtiles, to finally extract this thread's portion of the blocktile. In Simon's kernel that manifests as:
```
(<red>threadRow * TM</red> <green> + resIdxM </green>) * N + <red>threadCol * TN</red> <green> + resIdxN </green>
```
<red> these two multiplies </red> (color code in final article). Finally we loop over the threadtile with nice 2d indexing, which in Simon's kernel is handled by the <green> these adds </green>.

## Conclusion (for now)
I'm releasing these blogposts as a multipart series to hold myself accountable, rather then trying to write one huge post and kicking the can down the road. Next post I'll be diving in to some of the more optimized kernels in Simon's article. Implementing Simon's kernel 6 (the vectorized loads kernel), we will dig into some of the library internals to see how the `copy` macro automatically handles vectorization when possible. Implementing kernel 7 (warp tiling) we will see more features of cute including advanced copying and gain a deeper understanding of the framework by attempting to write the exact same access/computation patterns as Simon's kernel with CuTe idioms.
