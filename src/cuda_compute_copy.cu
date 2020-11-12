#include <stdint.h>
#include <cuda_runtime.h>

#define UNROLL 8
#define WARP_SIZE 32

__global__ static void simple_copy(void *dst_v, void *src_v, size_t size)
{
    ulong2 *dst = (ulong2 *)dst_v;
    ulong2 *src = (ulong2 *)src_v;

    int nwarps =  gridDim.x * blockDim.x / WARP_SIZE;
    int warp = (blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE;
    int wid = threadIdx.x % WARP_SIZE;

    size /= sizeof(*src);

    ulong2 *s = src + warp * WARP_SIZE * UNROLL;
    ulong2 *d = dst + warp * WARP_SIZE * UNROLL;

    while (s < src + size) {
        // It's faster to do a bunch of reads, followed by a bunch of writes,
        // instead of going one by one.
        ulong2 data[UNROLL];

        #pragma unroll
        for (int u=0; u<UNROLL; u++) {
            data[u] = s[u * WARP_SIZE + wid];
        }

        #pragma unroll
        for (int u=0; u<UNROLL; u++) {
            d[u * WARP_SIZE + wid] = data[u];
        }

        s += nwarps * WARP_SIZE * UNROLL;
        d += nwarps * WARP_SIZE * UNROLL;
    }
}

extern "C" void ComputeCopy(cudaStream_t stream, void *dst_v, void *src_v, size_t size)
{
    // Decent numbers for A100, haven't tried anything else.
    simple_copy<<<14, 512, 0, stream>>>(dst_v, src_v, size);
}
