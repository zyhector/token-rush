/*
 * Marlin, ported for Token Rush: bf16 activations, asymmetric int4 (w = q * s + m
 * with bf16 s and m per group of 128), fp32 cross-block reduction, no L2 cache
 * hints (the createpolicy / cp.async.L2::cache_hint pair is an illegal instruction
 * on sm_120), M <= 16 only. The tiling, the pipeline and the weight layout are
 * Marlin's; see tokenrush/marlin.py for the packing.
 *
 * Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
 * Modifications Copyright (C) 2026 Token Rush
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace marlin_bf16 {

constexpr int ceildiv(int a, int b) { return (a + b - 1) / b; }

template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) { return elems[i]; }
};

using I4 = Vec<int, 4>;
using scalar2_t = __nv_bfloat162;
using FragA = Vec<scalar2_t, 4>;
using FragB = Vec<scalar2_t, 2>;
using FragC = Vec<float, 4>;
using FragS = Vec<scalar2_t, 1>;   // quantization scales (and minimums, same layout)

__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr, bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "{\n"
    "   .reg .pred p;\n"
    "   setp.ne.b32 p, %0, 0;\n"
    "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
    "}\n" :: "r"((int) pred), "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
}

// Plain async copy for the weights: Marlin's evict-first L2 hint is not executable on sm_120.
__device__ inline void cp_async4_stream(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async_fence() { asm volatile("cp.async.commit_group;\n" ::); }

template <int n>
__device__ inline void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(n)); }

// m16n8k16 tensor core mma with bf16 inputs and fp32 accumulation.
__device__ inline void mma(const FragA& a_frag, const FragB& frag_b, FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  float* c = reinterpret_cast<float*>(&frag_c);
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
    :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),  "r"(b[0]),  "r"(b[1]),
       "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3])
  );
}

__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem)
  );
}

template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
}

// Four exact bf16 codes 0..15 from nibbles 0, 4 (lo) and 16, 20 (hi) of q: OR the
// nibble into the mantissa of 128.0 (0x4300) and subtract 128.
__device__ inline FragB dequant(int q) {
  const int MASK = 0x000f000f;
  const int EX = 0x43004300;
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  const int SUB = 0x43004300;
  FragB frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<scalar2_t*>(&lo), *reinterpret_cast<const scalar2_t*>(&SUB));
  frag_b[1] = __hsub2(*reinterpret_cast<scalar2_t*>(&hi), *reinterpret_cast<const scalar2_t*>(&SUB));
  return frag_b;
}

// w = q * s + m, one bf16 rounding (what dequantize_int4 in fp32-then-bf16 produces).
__device__ inline void scale(FragB& frag_b, FragS& frag_s, FragS& frag_m, int i) {
  scalar2_t s = __bfloat162bfloat162(reinterpret_cast<__nv_bfloat16*>(&frag_s)[i]);
  scalar2_t m = __bfloat162bfloat162(reinterpret_cast<__nv_bfloat16*>(&frag_m)[i]);
  frag_b[0] = __hfma2(frag_b[0], s, m);
  frag_b[1] = __hfma2(frag_b[1], s, m);
}

__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do
      asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
    while (state != count);
  }
  __syncthreads();
}

__device__ inline void barrier_release(int* lock, bool reset = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      lock[0] = 0;
      return;
    }
    int val = 1;
    asm volatile ("fence.acq_rel.gpu;\n");
    asm volatile ("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val));
  }
}

template <
  const int threads,
  const int thread_m_blocks,
  const int thread_n_blocks,
  const int thread_k_blocks,
  const int stages,
  const int group_blocks,
  const bool partial          // no cross-block reduction: block slice_idx writes fp32 slot slice_idx of C_part
>
__global__ void Marlin(
  const int4* __restrict__ A,     // bf16 input [m, k]
  const int4* __restrict__ B,     // int4 weights [k/16, n*16/8] in Marlin layout
        int4* __restrict__ C,     // bf16 output [m, n]
  const int4* __restrict__ s,     // bf16 scales [k/groupsize, n] (permuted)
  const int4* __restrict__ mn,    // bf16 minimums, same layout as s
        int4* __restrict__ C_tmp, // fp32 partial sums, [16 * thread_m_blocks, n] in fragment order
  int prob_m, int prob_n, int prob_k,
  int* locks,
  int n_slots                     // partial: slots [slice_count, n_slots) of a column are zero-filled
) {
  static_assert(group_blocks > 0, "grouped quantization only");
  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  int iters = ceildiv(k_tiles * n_tiles, gridDim.x);
  iters = (group_blocks / thread_k_blocks) * ceildiv(iters, (group_blocks / thread_k_blocks));

  int slice_row = (iters * blockIdx.x) % k_tiles;
  int slice_col_par = (iters * blockIdx.x) / k_tiles;
  int slice_col = slice_col_par;
  int slice_iters;
  int slice_count = 0;
  int slice_idx;

  auto init_slice = [&] () {
    slice_iters = iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
    if (slice_iters < 0 || slice_col_par >= n_tiles)
      slice_iters = 0;
    if (slice_iters == 0)
      return;
    if (slice_row + slice_iters > k_tiles)
      slice_iters = k_tiles - slice_row;
    slice_count = 1;
    slice_idx = 0;
    int col_first = iters * ceildiv(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = ceildiv(k_tiles - col_off, iters);
      if (col_off > 0)
        slice_count++;
      int delta_first = iters * blockIdx.x - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
        slice_idx = slice_count - 1;
      else {
        slice_idx = slice_count - 1 - delta_first / iters;
        if (col_off > 0)
          slice_idx--;
      }
    }
  };
  init_slice();

  int a_gl_stride = prob_k / 8;
  constexpr int a_sh_stride = 16 * thread_k_blocks / 8;
  constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 8;
  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
  constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);
  constexpr int a_sh_rd_delta_o = 2 * ((threads / 32) / (thread_n_blocks / 4));
  constexpr int a_sh_rd_delta_i = a_sh_stride * 16;
  constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks);
  constexpr int a_sh_wr_iters = ceildiv(a_sh_stage, a_sh_wr_delta);

  int b_gl_stride = 16 * prob_n / 32;
  constexpr int b_sh_stride = 32 * thread_n_blocks / 4;
  int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
  int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride);
  constexpr int b_sh_wr_delta = threads;
  constexpr int b_sh_rd_delta = threads;
  constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

  int s_gl_stride = prob_n / 8;
  constexpr int s_sh_stride = 16 * thread_n_blocks / 8;
  constexpr int s_sh_stage = s_sh_stride;
  int s_gl_rd_delta = s_gl_stride;

  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
  a_gl_rd += a_gl_rd_delta_o * slice_row;
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
  int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % 16) + (threadIdx.x % 32) / 16;
  a_sh_rd += 2 * ((threadIdx.x / 32) / (thread_n_blocks / 4));

  int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
  b_gl_rd += b_sh_stride * slice_col;
  b_gl_rd += b_gl_rd_delta_o * slice_row;
  int b_sh_wr = threadIdx.x;
  int b_sh_rd = threadIdx.x;

  int s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) + s_sh_stride * slice_col + threadIdx.x;
  int s_sh_wr = threadIdx.x;
  int s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) / 4;

  bool a_sh_wr_pred[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;
  bool s_sh_wr_pred = threadIdx.x < s_sh_stride;

  auto transform_a = [&] (int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row;
  };
  int a_sh_wr_trans[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
  int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) {
    #pragma unroll
    for (int j = 0; j < thread_m_blocks; j++)
      a_sh_rd_trans[i][j] = transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);
  }

  const int4* B_ptr[b_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++)
    B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;

  extern __shared__ int4 sh[];
  int4* sh_a = sh;
  int4* sh_b = sh_a + (stages * a_sh_stage);
  int4* sh_s = sh_b + (stages * b_sh_stage);
  int4* sh_m = sh_s + (stages * s_sh_stage);
  FragA frag_a[2][thread_m_blocks];
  I4 frag_b_quant[2];
  FragC frag_c[thread_m_blocks][4][2];
  FragS frag_s[2][4];
  FragS frag_m[2][4];

  auto zero_accums = [&] () {
    #pragma unroll
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
      reinterpret_cast<float*>(frag_c)[i] = 0;
  };

  auto fetch_to_shared = [&] (int pipe, int a_off, bool pred = true) {
    if (pred) {
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;
      #pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++) {
        cp_async4_pred(
          &sh_a_stage[a_sh_wr_trans[i]],
          &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
          a_sh_wr_pred[i]
        );
      }
      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
      #pragma unroll
      for (int i = 0; i < b_sh_wr_iters; i++) {
        cp_async4_stream(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr], B_ptr[i]);
        B_ptr[i] += b_gl_rd_delta_o;
      }
      if (pipe % (group_blocks / thread_k_blocks) == 0) {
        int4* sh_s_stage = sh_s + s_sh_stage * pipe;
        int4* sh_m_stage = sh_m + s_sh_stage * pipe;
        if (s_sh_wr_pred) {
          cp_async4_stream(&sh_s_stage[s_sh_wr], &s[s_gl_rd]);
          cp_async4_stream(&sh_m_stage[s_sh_wr], &mn[s_gl_rd]);
        }
        s_gl_rd += s_gl_rd_delta;
      }
    }
    cp_async_fence();
  };

  auto wait_for_stage = [&] () {
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  auto fetch_to_registers = [&] (int k, int pipe) {
    int grp = (group_blocks / thread_k_blocks) * (pipe / (group_blocks / thread_k_blocks));
    reinterpret_cast<int4*>(&frag_s[k % 2])[0] = (sh_s + s_sh_stage * grp)[s_sh_rd];
    reinterpret_cast<int4*>(&frag_m[k % 2])[0] = (sh_m + s_sh_stage * grp)[s_sh_rd];
    int4* sh_a_stage = sh_a + a_sh_stage * pipe;
    #pragma unroll
    for (int i = 0; i < thread_m_blocks; i++)
      ldsm4(frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;
    frag_b_quant[k % 2] = *reinterpret_cast<I4*>(&sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd]);
  };

  auto matmul = [&] (int k) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      int b_quant = frag_b_quant[k % 2][j];
      int b_quant_shift = b_quant >> 8;
      FragB frag_b0 = dequant(b_quant);
      scale(frag_b0, frag_s[k % 2][j], frag_m[k % 2][j], 0);
      FragB frag_b1 = dequant(b_quant_shift);
      scale(frag_b1, frag_s[k % 2][j], frag_m[k % 2][j], 1);
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        mma(frag_a[k % 2][i], frag_b0, frag_c[i][j][0]);
        mma(frag_a[k % 2][i], frag_b1, frag_c[i][j][1]);
      }
    }
  };

  auto thread_block_reduce = [&] () {
    constexpr int red_off = threads / b_sh_stride / 2;
    if (red_off >= 1) {
      int red_idx = threadIdx.x / b_sh_stride;
      constexpr int red_sh_stride = b_sh_stride * 4 * 2;
      constexpr int red_sh_delta = b_sh_stride;
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
      #pragma unroll
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
        #pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (i <= red_idx && red_idx < 2 * i) {
            #pragma unroll
            for (int j = 0; j < 4 * 2; j++) {
              int red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                float* c_rd = reinterpret_cast<float*>(&sh[red_sh_delta * j + red_sh_rd]);
                float* c_wr = reinterpret_cast<float*>(&sh[red_sh_wr]);
                #pragma unroll
                for (int k = 0; k < 4; k++)
                  reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + j][k] += c_rd[k] + c_wr[k];
              }
              sh[red_sh_wr] = reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j];
            }
          }
          __syncthreads();
        }
        if (red_idx == 0) {
          #pragma unroll
          for (int i = 0; i < 4 * 2; i++) {
            float* c_rd = reinterpret_cast<float*>(&sh[red_sh_delta * i + red_sh_rd]);
            #pragma unroll
            for (int j = 0; j < 4; j++)
              reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + i][j] += c_rd[j];
          }
        }
        __syncthreads();
      }
    }
  };

  // Cross-block reduction of a column slice through an fp32 buffer in fragment
  // order (Marlin reduces through the bf16 output; at K = 17408 that costs bits).
  auto global_reduce = [&] (bool first = false, bool last = false) {
    constexpr int tb_m = thread_m_blocks * 16;
    constexpr int tb_n = thread_n_blocks * 16;
    constexpr int c_size = tb_m * tb_n * sizeof(float) / 16;
    constexpr int active_threads = 32 * thread_n_blocks / 4;
    constexpr int num_floats = thread_m_blocks * 4 * 2 * 4;
    constexpr int th_size = num_floats * sizeof(float) / 16;
    if (threadIdx.x >= active_threads)
      return;
    int c_cur_offset = c_size * slice_col;
    if (!first) {
      float* frag_c_ptr = reinterpret_cast<float*>(&frag_c);
      #pragma unroll
      for (int k = 0; k < th_size; k++) {
        int4 v = C_tmp[c_cur_offset + active_threads * k + threadIdx.x];
        float* vf = reinterpret_cast<float*>(&v);
        #pragma unroll
        for (int f = 0; f < 4; f++)
          frag_c_ptr[k * 4 + f] += vf[f];
      }
    }
    if (!last) {
      int4* frag_c_ptr = reinterpret_cast<int4*>(&frag_c);
      #pragma unroll
      for (int k = 0; k < th_size; k++)
        C_tmp[c_cur_offset + active_threads * k + threadIdx.x] = frag_c_ptr[k];
    }
  };

  auto write_result = [&] () {
    int c_gl_stride = prob_n / 8;
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
    constexpr int c_sh_rd_delta = c_sh_stride * (threads / (2 * thread_n_blocks));

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * slice_col;
    int c_sh_wr = (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
    c_sh_wr += 32 * (threadIdx.x / 32);
    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));

    int c_gl_wr_end = c_gl_stride * prob_m;

    auto write = [&] (int idx, float c0, float c1) {
      ((scalar2_t*) sh)[idx] = __floats2bfloat162_rn(c0, c1);
    };
    if (threadIdx.x / 32 < thread_n_blocks / 4) {
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          int wr = c_sh_wr + 8 * j;
          write(wr + (4 * c_sh_stride) * 0 + 0, frag_c[i][j][0][0], frag_c[i][j][0][1]);
          write(wr + (4 * c_sh_stride) * 8 + 0, frag_c[i][j][0][2], frag_c[i][j][0][3]);
          write(wr + (4 * c_sh_stride) * 0 + 4, frag_c[i][j][1][0], frag_c[i][j][1][1]);
          write(wr + (4 * c_sh_stride) * 8 + 4, frag_c[i][j][1][2], frag_c[i][j][1][3]);
        }
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ceildiv(16 * thread_m_blocks, threads / (2 * thread_n_blocks)); i++) {
      if (c_gl_wr < c_gl_wr_end) {
        C[c_gl_wr] = sh[c_sh_rd];
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
  };

  // Partial mode: this block's fp32 result [prob_m, prob_n] into slot `slot` of C_tmp
  // (row-major, like C), or zeros when `zero`; the consumer sums the slots.
  auto write_partial = [&] (int slot, bool zero) {
    int c_gl_stride = prob_n / 4;
    constexpr int TN4 = 4 * thread_n_blocks;
    constexpr int c_sh_stride = TN4 + 1;
    int c_gl_wr_delta = c_gl_stride * (threads / TN4);
    constexpr int c_sh_rd_delta = c_sh_stride * (threads / TN4);
    int c_gl_wr = c_gl_stride * (threadIdx.x / TN4) + (threadIdx.x % TN4);
    c_gl_wr += TN4 * slice_col;
    int c_sh_wr = (2 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
    c_sh_wr += 32 * (threadIdx.x / 32);
    int c_sh_rd = c_sh_stride * (threadIdx.x / TN4) + (threadIdx.x % TN4);
    int c_gl_wr_end = c_gl_stride * prob_m;
    int4* Cp = C_tmp + (size_t) slot * prob_m * (prob_n / 4);
    if (!zero) {
      auto write = [&] (int idx, float c0, float c1) {
        ((float2*) sh)[idx] = make_float2(c0, c1);
      };
      if (threadIdx.x / 32 < thread_n_blocks / 4) {
        #pragma unroll
        for (int i = 0; i < thread_m_blocks; i++) {
          #pragma unroll
          for (int j = 0; j < 4; j++) {
            int wr = c_sh_wr + 8 * j;
            write(wr + (2 * c_sh_stride) * 0 + 0, frag_c[i][j][0][0], frag_c[i][j][0][1]);
            write(wr + (2 * c_sh_stride) * 8 + 0, frag_c[i][j][0][2], frag_c[i][j][0][3]);
            write(wr + (2 * c_sh_stride) * 0 + 4, frag_c[i][j][1][0], frag_c[i][j][1][1]);
            write(wr + (2 * c_sh_stride) * 8 + 4, frag_c[i][j][1][2], frag_c[i][j][1][3]);
          }
          c_sh_wr += 16 * (2 * c_sh_stride);
        }
      }
      __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < ceildiv(16 * thread_m_blocks, threads / TN4); i++) {
      if (c_gl_wr < c_gl_wr_end) {
        Cp[c_gl_wr] = zero ? make_int4(0, 0, 0, 0) : sh[c_sh_rd];
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
  };

  auto start_pipes = [&] () {
    #pragma unroll
    for (int i = 0; i < stages - 1; i++)
      fetch_to_shared(i, i, i < slice_iters);
    zero_accums();
    wait_for_stage();
    fetch_to_registers(0, 0);
    a_gl_rd += a_gl_rd_delta_o * (stages - 1);
  };
  start_pipes();

  while (slice_iters) {
    #pragma unroll
    for (int pipe = 0; pipe < stages;) {
      #pragma unroll
      for (int k = 0; k < b_sh_wr_iters; k++) {
        fetch_to_registers(k + 1, pipe % stages);
        if (k == b_sh_wr_iters - 2) {
          fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages);
          pipe++;
          wait_for_stage();
        }
        matmul(k);
      }
      slice_iters--;
      if (slice_iters == 0)
        break;
    }
    a_gl_rd += a_gl_rd_delta_o * stages;

    if (slice_iters == 0) {
      cp_async_wait<0>();
      bool last = slice_idx == slice_count - 1;
      thread_block_reduce();
      if (partial) {
        write_partial(slice_idx, false);
        if (last)
          for (int slot = slice_count; slot < n_slots; slot++)
            write_partial(slot, true);
      } else {
        if (slice_count > 1) {
          barrier_acquire(&locks[slice_col], slice_idx);
          global_reduce(slice_idx == 0, last);
          barrier_release(&locks[slice_col], last);
        }
        if (last)
          write_result();
      }
      slice_row = 0;
      slice_col_par++;
      slice_col++;
      init_slice();
      if (slice_iters) {
        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
        #pragma unroll
        for (int i = 0; i < b_sh_wr_iters; i++)
          B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
        if (slice_col == 0) {
          #pragma unroll
          for (int i = 0; i < b_sh_wr_iters; i++)
            B_ptr[i] -= b_gl_stride;
        }
        s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
        start_pipes();
      }
    }
  }
}

const int THREADS = 256;
const int STAGES = 4;

template <int TM, int TN, int TK, int GB>
constexpr int smem_bytes() {
  return STAGES * ((16 * TK / 8) * (16 * TM) + (32 * TN / 4) * TK + 2 * (16 * TN / 8)) * 16;
}

#define CALL_IF(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, GROUP_BLOCKS, PARTIAL) \
  else if ( \
    thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS && thread_k_blocks == THREAD_K_BLOCKS && \
    group_blocks == GROUP_BLOCKS && partial == PARTIAL \
  ) { \
    constexpr int SMEM = smem_bytes<THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, GROUP_BLOCKS>(); \
    static bool configured = false; \
    if (!configured) { \
      cudaFuncSetAttribute( \
        Marlin<THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS, PARTIAL>, \
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM); \
      configured = true; \
    } \
    Marlin< \
      THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS, PARTIAL \
    ><<<blocks, THREADS, SMEM, stream>>>( \
      A_ptr, B_ptr, C_ptr, s_ptr, m_ptr, red_ptr, prob_m, prob_n, prob_k, locks, n_slots \
    ); \
  }

// Partial mode: how many fp32 slots a launch writes (the longest column slice + 1).
int partial_slots(int prob_n, int prob_k, int thread_k, int thread_n, int sms, int dev) {
  if (sms == -1)
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (thread_k == -1 || thread_n == -1) {
    thread_k = 128;
    thread_n = 128;
  }
  int k_tiles = prob_k / thread_k;
  int n_tiles = prob_n / thread_n;
  int gb = 8 / (thread_k / 16);        // group_blocks / thread_k_blocks
  int iters = ceildiv(k_tiles * n_tiles, sms);
  iters = gb * ceildiv(iters, gb);
  return ceildiv(k_tiles, iters) + 1;
}

const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;

int marlin_bf16_cuda(
  const void* A, const void* B, void* C, const void* s, const void* mn, void* red,
  int prob_m, int prob_n, int prob_k,
  void* workspace, int groupsize, int dev, cudaStream_t stream,
  int thread_k, int thread_n, int sms, bool partial, int n_slots
) {
  if (sms == -1)
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (thread_k == -1 || thread_n == -1) {
    thread_k = 128;
    thread_n = 128;
  }
  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;
  int group_blocks = groupsize / 16;
  int blocks = sms;

  if (prob_m > 16 || prob_n % thread_n != 0 || prob_k % thread_k != 0 || prob_k % groupsize != 0)
    return ERR_PROB_SHAPE;
  if (prob_m == 0 || prob_n == 0 || prob_k == 0)
    return 0;

  const int4* A_ptr = (const int4*) A;
  const int4* B_ptr = (const int4*) B;
  int4* C_ptr = (int4*) C;
  const int4* s_ptr = (const int4*) s;
  const int4* m_ptr = (const int4*) mn;
  int4* red_ptr = (int4*) red;
  int* locks = (int*) workspace;
  int thread_m_blocks = 1;

  int ret = 0;
  if (false) {}
  CALL_IF(1, 8, 8, 8, false)
  CALL_IF(1, 16, 4, 8, false)
  CALL_IF(1, 8, 8, 8, true)
  CALL_IF(1, 16, 4, 8, true)
  else
    ret = ERR_KERN_SHAPE;
  return ret;
}

}  // namespace marlin_bf16
