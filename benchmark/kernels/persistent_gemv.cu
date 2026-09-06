// Does a persistent "GEMV engine" beat one launch per GEMV on this box?
//
// The decode step runs 465 dependent FP8 GEMVs (4.94 ms, ~68% of memory roofline). Each is
// a separate kernel, so the GPU fully drains between them: the trace shows ~1.06 ms/step of
// inter-kernel gaps on the main stream. This benchmark isolates exactly that effect. The
// GEMV body is IDENTICAL in both variants; only the dispatch differs:
//
//   baseline    N kernel launches, one per GEMV (what the engine does today, under a graph)
//   persistent  ONE resident kernel that walks the same chain, separating stages with a
//               software grid barrier (the megakernel pattern)
//
// Shapes are the real ones: GDN in_proj_qkvz [16384,2560] and out_proj [2560,6144], x36
// layers, M=4 rows (a k=3 verify step). Weights are e4m3 with a per-output-channel scale,
// as in the FP8 dense path.
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)

constexpr int M_ROWS   = 4;     // verify rows per step
constexpr int BLOCK    = 256;
constexpr int ROWS_PER_BLK = 4; // output channels per block

struct Desc {
  const __nv_fp8_e4m3* w;   // [N, K] row-major
  const float* wscale;      // [N]
  const __nv_bfloat16* x;   // [M, K]
  __nv_bfloat16* y;         // [M, N]
  int N, K;
};

// One GEMV stage: y[m, n] = sum_k w[n,k]*x[m,k] * wscale[n].  Grid-stride over output rows so
// the same body works for a fixed persistent grid and for a per-stage sized grid.
// Vectorized body: 16-byte loads (16 e4m3 weights per thread per step), which is what a
// production GEMV does. Keeping the body realistic matters -- a slow body would inflate each
// stage and understate the fixed inter-stage drain the persistent variant removes.
__device__ __forceinline__ void gemv_body(const Desc& d, int block_id, int num_blocks) {
  const int lane = threadIdx.x;
  const int KV = d.K >> 4;                      // 16 e4m3 per vector load
  for (int row0 = block_id * ROWS_PER_BLK; row0 < d.N; row0 += num_blocks * ROWS_PER_BLK) {
    float acc[ROWS_PER_BLK][M_ROWS];
#pragma unroll
    for (int r = 0; r < ROWS_PER_BLK; ++r)
#pragma unroll
      for (int m = 0; m < M_ROWS; ++m) acc[r][m] = 0.f;

    for (int kv = lane; kv < KV; kv += BLOCK) {
      float xv[M_ROWS][16];
#pragma unroll
      for (int m = 0; m < M_ROWS; ++m) {
        const float4* xp = (const float4*)(d.x + (size_t)m * d.K + (size_t)kv * 16);
        float4 a = xp[0], b = xp[1];             // 16 bf16 = 32 B
        const __nv_bfloat16* ah = (const __nv_bfloat16*)&a;
        const __nv_bfloat16* bh = (const __nv_bfloat16*)&b;
#pragma unroll
        for (int i = 0; i < 8; ++i) { xv[m][i] = __bfloat162float(ah[i]); xv[m][8 + i] = __bfloat162float(bh[i]); }
      }
#pragma unroll
      for (int r = 0; r < ROWS_PER_BLK; ++r) {
        int row = row0 + r;
        if (row < d.N) {
          uint4 wq = *(const uint4*)(d.w + (size_t)row * d.K + (size_t)kv * 16);
          const __nv_fp8_e4m3* wp = (const __nv_fp8_e4m3*)&wq;
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            float wv = float(wp[i]);
#pragma unroll
            for (int m = 0; m < M_ROWS; ++m) acc[r][m] += wv * xv[m][i];
          }
        }
      }
    }
    __shared__ float red[ROWS_PER_BLK][M_ROWS][BLOCK / 32];
    int warp = lane >> 5, tl = lane & 31;
#pragma unroll
    for (int r = 0; r < ROWS_PER_BLK; ++r)
#pragma unroll
      for (int m = 0; m < M_ROWS; ++m) {
        float v = acc[r][m];
#pragma unroll
        for (int off = 16; off; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tl == 0) red[r][m][warp] = v;
      }
    __syncthreads();
    if (lane < ROWS_PER_BLK * M_ROWS) {
      int r = lane / M_ROWS, m = lane % M_ROWS, row = row0 + r;
      if (row < d.N) {
        float v = 0.f;
        for (int w = 0; w < BLOCK / 32; ++w) v += red[r][m][w];
        d.y[m * d.N + row] = __float2bfloat16(v * d.wscale[row]);
      }
    }
    __syncthreads();
  }
}

__global__ void gemv_single(Desc d) { gemv_body(d, blockIdx.x, gridDim.x); }

// Software grid barrier: every resident block arrives, the last one flips the generation.
// Valid only when the whole grid is co-resident (we size the grid from the occupancy API).
__device__ __forceinline__ void grid_barrier(unsigned int* counter, volatile unsigned int* gen,
                                             unsigned int num_blocks) {
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned int g = *gen;                 // generation before arriving
    __threadfence();                       // this stage's stores visible device-wide first
    if (atomicAdd(counter, 1u) == num_blocks - 1) {
      atomicExch(counter, 0u);
      __threadfence();
      atomicAdd((unsigned int*)gen, 1u);   // release every waiter
    } else {
      while (*gen == g) __nanosleep(32);
    }
  }
  __syncthreads();
}

// Fairest comparison for the persistent idea: the hardware-assisted cooperative-groups
// barrier rather than a hand-rolled one.
__global__ void gemv_coop(const Desc* __restrict__ chain, int n_stages) {
  cg::grid_group grid = cg::this_grid();
  for (int s = 0; s < n_stages; ++s) {
    gemv_body(chain[s], blockIdx.x, gridDim.x);
    if (s + 1 < n_stages) grid.sync();
  }
}

__global__ void gemv_persistent(const Desc* __restrict__ chain, int n_stages,
                                unsigned int* counter, volatile unsigned int* gen, unsigned int nblocks) {
  for (int s = 0; s < n_stages; ++s) {
    gemv_body(chain[s], blockIdx.x, gridDim.x);
    if (s + 1 < n_stages) grid_barrier(counter, gen, nblocks);
  }
}

int main(int argc, char** argv) {
  int layers = (argc > 1) ? atoi(argv[1]) : 36;
  int iters  = (argc > 2) ? atoi(argv[2]) : 50;
  const int H = 2560, QKVZ = 16384, OUTK = 6144;

  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, 0));

  std::vector<Desc> host_chain;
  std::vector<void*> allocs;
  auto mk = [&](int N, int K) {
    Desc d{}; d.N = N; d.K = K;
    void *w, *s, *x, *y;
    CK(cudaMalloc(&w, (size_t)N * K));                       // e4m3 = 1 byte
    CK(cudaMalloc(&s, (size_t)N * sizeof(float)));
    CK(cudaMalloc(&x, (size_t)M_ROWS * K * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&y, (size_t)M_ROWS * N * sizeof(__nv_bfloat16)));
    CK(cudaMemset(w, 0x3c, (size_t)N * K));                  // ~1.0 in e4m3
    CK(cudaMemset(s, 0, (size_t)N * sizeof(float)));
    CK(cudaMemset(x, 0, (size_t)M_ROWS * K * sizeof(__nv_bfloat16)));
    allocs.insert(allocs.end(), {w, s, x, y});
    d.w = (const __nv_fp8_e4m3*)w; d.wscale = (const float*)s;
    d.x = (const __nv_bfloat16*)x; d.y = (__nv_bfloat16*)y;
    return d;
  };
  int mode = (argc > 3) ? atoi(argv[3]) : 0;
  if (mode == 0) {
    for (int l = 0; l < layers; ++l) {               // GDN dense pair per layer (big stages)
      host_chain.push_back(mk(QKVZ, H));
      host_chain.push_back(mk(H, OUTK));
    }
  } else {
    // The production mix is dominated by SMALL dependent GEMVs (465 launches/step for
    // ~6 GB, i.e. ~13 MB and ~10.6 us each) -- the regime where the fixed inter-kernel
    // drain is the largest share of a stage, so it is the best case for a persistent grid.
    for (int l = 0; l < layers; ++l) {
      host_chain.push_back(mk(H, H));                // HC mix        6.6 MB
      host_chain.push_back(mk(1280, H));             // shared gate_up 3.3 MB
      host_chain.push_back(mk(H, 640));              // shared down    1.6 MB
      host_chain.push_back(mk(H, H));                // HC combine mix 6.6 MB
    }
  }
  int n_stages = host_chain.size();
  size_t bytes = 0;
  for (auto& d : host_chain) bytes += (size_t)d.N * d.K;

  Desc* dev_chain; CK(cudaMalloc(&dev_chain, n_stages * sizeof(Desc)));
  CK(cudaMemcpy(dev_chain, host_chain.data(), n_stages * sizeof(Desc), cudaMemcpyHostToDevice));
  unsigned int* sync; CK(cudaMalloc(&sync, 2 * sizeof(unsigned int)));
  CK(cudaMemset(sync, 0, 2 * sizeof(unsigned int)));

  int blocks_per_sm = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, gemv_persistent, BLOCK, 0));
  unsigned int nblocks = blocks_per_sm * prop.multiProcessorCount;
  printf("GPU %s: %d SMs, %d blocks/SM resident -> persistent grid %u blocks\n",
         prop.name, prop.multiProcessorCount, blocks_per_sm, nblocks);
  printf("chain: %d stages, %.2f GB of weights, M=%d\n", n_stages, bytes / 1e9, M_ROWS);

  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto time_it = [&](const char* name, auto&& fn) {
    for (int i = 0; i < 5; ++i) fn();                 // warm up
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) fn();
    CK(cudaEventRecord(e1));
    CK(cudaDeviceSynchronize());
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    ms /= iters;
    printf("  %-28s %7.3f ms/chain  %6.2f us/stage  %6.0f GB/s\n",
           name, ms, ms * 1000.0 / n_stages, bytes / (ms * 1e-3) / 1e9);
    return ms;
  };

  printf("\neager launches:\n");
  float t_base = time_it("baseline (1 launch/GEMV)", [&] {
    for (int s = 0; s < n_stages; ++s) {
      int grid = (host_chain[s].N + ROWS_PER_BLK - 1) / ROWS_PER_BLK;
      gemv_single<<<grid, BLOCK>>>(host_chain[s]);
    }
  });
  float t_pers = time_it("persistent (1 launch)", [&] {
    gemv_persistent<<<nblocks, BLOCK>>>(dev_chain, n_stages, sync, sync + 1, nblocks);
  });

  // Same comparison under a CUDA graph: production replays a captured graph, so this is the
  // number that matters -- it already removes per-launch CPU cost from the baseline.
  cudaStream_t st; CK(cudaStreamCreate(&st));
  cudaGraph_t g_base, g_pers; cudaGraphExec_t x_base, x_pers;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  for (int s = 0; s < n_stages; ++s) {
    int grid = (host_chain[s].N + ROWS_PER_BLK - 1) / ROWS_PER_BLK;
    gemv_single<<<grid, BLOCK, 0, st>>>(host_chain[s]);
  }
  CK(cudaStreamEndCapture(st, &g_base));
  CK(cudaGraphInstantiate(&x_base, g_base, nullptr, nullptr, 0));
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  gemv_persistent<<<nblocks, BLOCK, 0, st>>>(dev_chain, n_stages, sync, sync + 1, nblocks);
  CK(cudaStreamEndCapture(st, &g_pers));
  CK(cudaGraphInstantiate(&x_pers, g_pers, nullptr, nullptr, 0));

  printf("\nunder CUDA graph replay:\n");
  float g_b = time_it("baseline graph", [&] { cudaGraphLaunch(x_base, st); });
  float g_p = time_it("persistent graph", [&] { cudaGraphLaunch(x_pers, st); });

  // cooperative launch (requires a co-resident grid, same sizing rule)
  int coop_bps = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&coop_bps, gemv_coop, BLOCK, 0));
  unsigned int coop_blocks = coop_bps * prop.multiProcessorCount;
  void* args[] = {&dev_chain, &n_stages};
  float t_coop = time_it("persistent (coop grid.sync)", [&] {
    cudaLaunchCooperativeKernel((void*)gemv_coop, dim3(coop_blocks), dim3(BLOCK), args, 0, 0);
  });
  printf("  (cooperative grid: %u blocks)\n", coop_blocks);

  printf("\nspeedup vs baseline: sw-barrier %.2fx, coop %.2fx, graph %.2fx\n",
         t_base / t_pers, t_base / t_coop, g_b / g_p);
  for (void* p : allocs) cudaFree(p);
  return 0;
}
