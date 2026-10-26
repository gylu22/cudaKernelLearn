#include <cuda_runtime.h>

#include "sgemm/launchers.h"

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define OFFSET 1
#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))
// block 32x32 大小
__global__ void sgemm_naive_f32_kernel(float *a, 
          float *b, float *c, int M,int N, int K){
  
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx < N && ty <M){
      float sum =0; 
      #pragma unroll
      for (int i=0; i<K; i++){
        sum += a[ty * K + i] * b[i * N + tx];
      }
      c[ty * N + tx] = sum;
    }
}

// shared memory + Block Tile 
template <const int BM = 32, const int BN = 32, const int BK = 32>
__global__ void sgemm_sliced_k_f32_kernel(float *a, float *b, float *c, int M,
                                          int N, int K) {

  float __shared__ tileA[BM][BK],tileB[BK][BN];
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx; 

  float *a_ptr = a + by * blockDim.y * K;
  float *b_ptr = b + bx * blockDim.x; 

  // 外部循环 （K + BK-1）/ BK次 
  float sum = 0.f;
  for (int k=0; k<K; k+= BK){
    // 将一个tile的数据导入到shared memory 
    if (row < M && k + tx < K ){
      tileA[ty][tx] = a_ptr[ty * K + k + tx];
    }
    else {
      tileA[ty][tx] = 0.f;
    }
    if (col < N && k + ty < K){
      tileB[ty][tx] = b_ptr[(k + ty) * N + tx];
    }
    else {
      tileB[ty][tx] = 0.f;
    }
    __syncthreads();

    #pragma unroll
    for (int i =0;i<BK;i++){
      sum += tileA[ty][i] * tileB[i][tx];
    }
    __syncthreads();
  }
  if (row < M && col < N) {
    c[row * N + col] = sum;
  }

}

// block size 16 x 16 Tile Size: 128 x 128 
// 单个线程计算 128 / 16 x 128  / 16 = 8x8 大小  
// shared memory 大小 128 x 8

template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8>
__global__ void sgemm_t_8x8_sliced_k_f32x4_kernel(float *a, float *b, float *c,
                                                  int M, int N, int K) {

    float __shared__ tileA[BM][BK], tileB[BK][BN];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float *a_ptr = a + blockIdx.y * BM * K;
    float *b_ptr = b + blockIdx.x * BN;
    float *c_ptr = c + blockIdx.y * BM * N + blockIdx.x * BN;
    int tid = ty * blockDim.x + tx; // 0-127

    float temp[TM][TN] = {0.f};
    // 重新映射线程
    int ta_y = tid / (BK / 4); // 0-127
    int ta_x = (tid % (BK / 4)) * 4 ; // 0,4     
    int tb_y = tid / (BN / 4); // 0-7
    int tb_x = tid % (BN / 4) * 4; // 0,4,8...  

    // 外层循环（M+BK -1）/ BK
    for (int k=0; k<K; k+=BK){
  
      FLOAT4(tileA[ta_y][ta_x]) = FLOAT4(a_ptr[ta_y * K + ta_x + k]);
      FLOAT4(tileB[tb_y][tb_x]) = FLOAT4(b_ptr[(k + tb_y) * N + tb_x]);
      // float4 x_vec = FLOAT4(a_ptr[ta_y * K + ta_x + k]);
      // tileA[ta_y][ta_x] = x_vec.x;
      // tileA[ta_y][ta_x+1] = x_vec.y;
      // tileA[ta_y][ta_x+2] = x_vec.z;
      // tileA[ta_y][ta_x+3] = x_vec.w;
      
      // float4 b_vec = FLOAT4(b_ptr[(k + tb_y) * N + tb_x]);
      // tileB[tb_y][tb_x] = b_vec.x;
      // tileB[tb_y][tb_x+1] = b_vec.y;
      // tileB[tb_y][tb_x+2] = b_vec.z;
      // tileB[tb_y][tb_x+3] = b_vec.w;
      
      // 第一次同步，保证第一次循环计算之前所有元素都读取到shared memory
      __syncthreads();
      // 计算每个线程的BK x BK的区域
      // 内层循环BK次
      #pragma unroll
      for (int i = 0; i < BK; i++) {
        float tileA_BK[TM];
        float tileB_BK[TN];
        #pragma unroll
        for (int j = 0; j < TM; j++) {
          tileA_BK[j] = tileA[ty * TM + j][i];
        }
        #pragma unroll
        for (int j = 0; j < TN; j++) {
          tileB_BK[j] = tileB[i][tx * TN + j];
        }
        #pragma unroll
        for (int a_i = 0; a_i < TM; a_i++) {
          for (int b_j = 0; b_j < TN; b_j++) {
            temp[a_i][b_j] = __fmaf_rn(tileA_BK[a_i], tileB_BK[b_j],temp[a_i][b_j]);
          }
        }
      }
      // 第二次同步，保证下一次更新shared memory之前已经计算完成
      __syncthreads();
      }
      #pragma unroll
      for (int i=0;i<TM;i++){
        for (int j=0;j<TN;j++){
          c_ptr[(ty * TM + i) * N + tx * TN + j] = temp[i][j];
        }
      }
}


template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8>
__global__ void sgemm_t_8x8_sliced_k_bcf_f32x4_kernel(float *a, float *b, float *c,
                                                  int M, int N, int K) {

    float __shared__ tileA[BK][BM], tileB[BK][BN];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float *a_ptr = a + blockIdx.y * BM * K;
    float *b_ptr = b + blockIdx.x * BN;
    float *c_ptr = c + blockIdx.y * BM * N + blockIdx.x * BN;
    int tid = ty * blockDim.x + tx; // 0-127

    float temp[TM][TN] = {0.f};
    // 重新映射线程
    int ta_y = tid / (BK / 4); // 0-127
    int ta_x = (tid % (BK / 4)) * 4 ; // 0,4     
    int tb_y = tid / (BN / 4); // 0-7
    int tb_x = tid % (BN / 4) * 4; // 0,4,8...  

    // 外层循环（M+BK -1）/ BK
    for (int k=0; k<K; k+=BK){
  
      float4 x_vec = FLOAT4(a_ptr[ta_y * K + ta_x + k]);
      tileA[ta_x][ta_y] = x_vec.x;
      tileA[ta_x+1][ta_y] = x_vec.y;
      tileA[ta_x+2][ta_y] = x_vec.z;
      tileA[ta_x+3][ta_y] = x_vec.w;

      FLOAT4(tileB[tb_y][tb_x]) = FLOAT4(b_ptr[(k + tb_y) * N + tb_x]);
      
      // 第一次同步，保证第一次循环计算之前所有元素都读取到shared memory
      __syncthreads();
      // 计算每个线程的BK x BK的区域
      // 内层循环BK次
      #pragma unroll
      for (int i = 0; i < BK; i++) {
        float tileA_BK[TM];
        float tileB_BK[TN];
        #pragma unroll
        for (int j = 0; j < TM; j++) {
          // tileA_BK[j] = tileA[ty * TM + j][i];
          tileA_BK[j] = tileA[i][ty * TM + j];
        }
        // FLOAT4(tileA_BK[0]) = FLOAT4(tileA[i][ty * TM ]);
        // FLOAT4(tileA_BK[4]) = FLOAT4(tileA[i][ty * TM + 4]);

        #pragma unroll
        for (int j = 0; j < TN; j++) {
          tileB_BK[j] = tileB[i][tx * TN + j];
        }
        // FLOAT4(tileB_BK[0]) = FLOAT4(tileB[i][tx * TN]);
        // FLOAT4(tileB_BK[4]) = FLOAT4(tileB[i][tx * TN + 4]);

        #pragma unroll
        for (int a_i = 0; a_i < TM; a_i++) {
          for (int b_j = 0; b_j < TN; b_j++) {
            temp[a_i][b_j] = __fmaf_rn(tileA_BK[a_i], tileB_BK[b_j],temp[a_i][b_j]);
          }
        }
      }
      // 第二次同步，保证下一次更新shared memory之前已经计算完成
      __syncthreads();
      }
      #pragma unroll
      for (int i=0;i<TM;i++){
        for (int j=0;j<TN;j++){
          c_ptr[(ty * TM + i) * N + tx * TN + j] = temp[i][j];
        }
      }
}




template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8>
__global__ void sgemm_t_8x8_sliced_k_bcf_db_f32x4_kernel(float *a, float *b, float *c,
                                                  int M, int N, int K) {

    float __shared__ tileA[2][BK][BM], tileB[2][BK][BN];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float *a_ptr = a + blockIdx.y * BM * K;
    float *b_ptr = b + blockIdx.x * BN;
    float *c_ptr = c + blockIdx.y * BM * N + blockIdx.x * BN;
    int tid = ty * blockDim.x + tx; // 0-127

    float temp[TM][TN] = {0.f};
    // 重新映射线程
    int ta_y = tid / (BK / 4); // 0-127
    int ta_x = (tid % (BK / 4)) * 4 ; // 0,4     
    int tb_y = tid / (BN / 4); // 0-7
    int tb_x = tid % (BN / 4) * 4; // 0,4,8...  
    // 提前读取第一次循环的数据
    float4 x_vec = FLOAT4(a_ptr[ta_y * K + ta_x]);
    tileA[0][ta_x][ta_y] = x_vec.x;
    tileA[0][ta_x+1][ta_y] = x_vec.y;
    tileA[0][ta_x+2][ta_y] = x_vec.z;
    tileA[0][ta_x+3][ta_y] = x_vec.w;

    FLOAT4(tileB[0][tb_y][tb_x]) = FLOAT4(b_ptr[tb_y * N + tb_x]);
    __syncthreads();
    int read_stage_idx = 0;                                               
    // 外层循环（M+BK -1）/ BK
    for (int k=0; k<K; k+=BK){
      int load_stage_idx = read_stage_idx ^ 1;
      if (k+BK < K){
      float4 x_vec = FLOAT4(a_ptr[ta_y * K + ta_x + k + BK]);
      tileA[load_stage_idx][ta_x][ta_y] = x_vec.x;
      tileA[load_stage_idx][ta_x+1][ta_y] = x_vec.y;
      tileA[load_stage_idx][ta_x+2][ta_y] = x_vec.z;
      tileA[load_stage_idx][ta_x+3][ta_y] = x_vec.w;
      FLOAT4(tileB[load_stage_idx][tb_y][tb_x]) = FLOAT4(b_ptr[(k +BK+tb_y) * N + tb_x]);
      }
      // 计算每个线程的BK x BK的区域
      // 内层循环BK次
      #pragma unroll
      for (int i = 0; i < BK; i++) {
        float tileA_BK[TM];
        float tileB_BK[TN];
        #pragma unroll
        for (int j = 0; j < TM; j++) {
          // tileA_BK[j] = tileA[ty * TM + j][i];
          tileA_BK[j] = tileA[read_stage_idx][i][ty * TM + j];
        }
        #pragma unroll
        for (int j = 0; j < TN; j++) {
          tileB_BK[j] = tileB[read_stage_idx][i][tx * TN + j];
        }
        #pragma unroll
        for (int a_i = 0; a_i < TM; a_i++) {
          for (int b_j = 0; b_j < TN; b_j++) {
            // temp[a_i][b_j] += tileA_BK[a_i] * tileB_BK[b_j];
            // PTX 内联函数 
            temp[a_i][b_j] = __fmaf_rn(tileA_BK[a_i], tileB_BK[b_j],temp[a_i][b_j]);
          }
        }
      }
      // 第二次同步，保证下一次更新shared memory之前已经计算完成
      __syncthreads();
      read_stage_idx ^=1;

      }
      #pragma unroll
      for (int i = 0; i < TM; i++) {
        // 计算当前线程在 C 矩阵当前行的基础偏移量
        int row_offset = (ty * TM + i) * N + tx * TN;
        
        // 构造前 4 个元素的 float4 向量 (j = 0, 1, 2, 3)
        float4 vec0 = make_float4(temp[i][0], temp[i][1], temp[i][2], temp[i][3]);
        // 构造后 4 个元素的 float4 向量 (j = 4, 5, 6, 7)
        float4 vec1 = make_float4(temp[i][4], temp[i][5], temp[i][6], temp[i][7]);
        
        // 直接触发 128-bit 向量化存储 (ST.128)
        FLOAT4(c_ptr[row_offset])     = vec0;
        FLOAT4(c_ptr[row_offset + 4]) = vec1;
      }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmResolveBankExtraCol(int M, int N, int K, float alpha,
                                         float *A, float *B, float beta,
                                         float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  const int extraCols = 5;
  __shared__ float Bs[BK * (BN + extraCols)];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    // transpose A while loading it
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    tmp = reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 0] = tmp.x;
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 1] = tmp.y;
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 2] = tmp.z;
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 3] = tmp.w;
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * (BN + extraCols) + threadCol * TN + i];
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

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      // load C vector into registers
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      // perform GEMM update in reg
      tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
      // write back
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
  }
}


// PyTorch validation and dispatch live in csrc/bindings/sgemm_api.cpp.

namespace sgemm {
namespace {

float* mutable_ptr(const float* ptr) {
  return const_cast<float*>(ptr);
}

}  // namespace

cudaError_t launch_simt_fp32_00_naive(
    const float* a, const float* b, float* c,
    int m, int n, int k, cudaStream_t stream) {
  constexpr int kBlockM = 32;
  constexpr int kBlockN = 32;
  const dim3 block(kBlockN, kBlockM);
  const dim3 grid(CEIL_DIV(n, kBlockN), CEIL_DIV(m, kBlockM));
  sgemm_naive_f32_kernel<<<grid, block, 0, stream>>>(
      mutable_ptr(a), mutable_ptr(b), c, m, n, k);
  return cudaPeekAtLastError();
}

cudaError_t launch_simt_fp32_01_smem_tiled(
    const float* a, const float* b, float* c,
    int m, int n, int k, cudaStream_t stream) {
  constexpr int kBlockM = 32;
  constexpr int kBlockN = 32;
  constexpr int kBlockK = 32;
  const dim3 block(kBlockN, kBlockM);
  const dim3 grid(CEIL_DIV(n, kBlockN), CEIL_DIV(m, kBlockM));
  sgemm_sliced_k_f32_kernel<kBlockM, kBlockN, kBlockK>
      <<<grid, block, 0, stream>>>(mutable_ptr(a), mutable_ptr(b), c, m, n, k);
  return cudaPeekAtLastError();
}

cudaError_t launch_simt_fp32_02_thread_tile_vec4(
    const float* a, const float* b, float* c,
    int m, int n, int k, cudaStream_t stream) {
  constexpr int kBlockM = 128;
  constexpr int kBlockN = 128;
  constexpr int kBlockK = 8;
  constexpr int kThreadM = 8;
  constexpr int kThreadN = 8;
  const dim3 block(kBlockN / kThreadN, kBlockM / kThreadM);
  const dim3 grid(n / kBlockN, m / kBlockM);
  sgemm_t_8x8_sliced_k_f32x4_kernel<
      kBlockM, kBlockN, kBlockK, kThreadM, kThreadN>
      <<<grid, block, 0, stream>>>(mutable_ptr(a), mutable_ptr(b), c, m, n, k);
  return cudaPeekAtLastError();
}

cudaError_t launch_simt_fp32_03_transposed_smem_vec4(
    const float* a, const float* b, float* c,
    int m, int n, int k, cudaStream_t stream) {
  constexpr int kBlockM = 128;
  constexpr int kBlockN = 128;
  constexpr int kBlockK = 8;
  constexpr int kThreadM = 8;
  constexpr int kThreadN = 8;
  const dim3 block(kBlockN / kThreadN, kBlockM / kThreadM);
  const dim3 grid(n / kBlockN, m / kBlockM);
  sgemm_t_8x8_sliced_k_bcf_f32x4_kernel<
      kBlockM, kBlockN, kBlockK, kThreadM, kThreadN>
      <<<grid, block, 0, stream>>>(mutable_ptr(a), mutable_ptr(b), c, m, n, k);
  return cudaPeekAtLastError();
}

cudaError_t launch_simt_fp32_04_double_buffer(
    const float* a, const float* b, float* c,
    int m, int n, int k, cudaStream_t stream) {
  constexpr int kBlockM = 128;
  constexpr int kBlockN = 128;
  constexpr int kBlockK = 8;
  constexpr int kThreadM = 8;
  constexpr int kThreadN = 8;
  const dim3 block(kBlockN / kThreadN, kBlockM / kThreadM);
  const dim3 grid(n / kBlockN, m / kBlockM);
  sgemm_t_8x8_sliced_k_bcf_db_f32x4_kernel<
      kBlockM, kBlockN, kBlockK, kThreadM, kThreadN>
      <<<grid, block, 0, stream>>>(mutable_ptr(a), mutable_ptr(b), c, m, n, k);
  return cudaPeekAtLastError();
}

cudaError_t launch_simt_fp32_05_padded_smem(
    const float* a, const float* b, float* c,
    int m, int n, int k, cudaStream_t stream) {
  constexpr int kBlockM = 128;
  constexpr int kBlockN = 128;
  constexpr int kBlockK = 8;
  constexpr int kThreadM = 8;
  constexpr int kThreadN = 8;
  constexpr float kAlpha = 1.0f;
  constexpr float kBeta = 0.0f;
  const dim3 block((kBlockM * kBlockN) / (kThreadM * kThreadN));
  const dim3 grid(n / kBlockN, m / kBlockM);
  sgemmResolveBankExtraCol<
      kBlockM, kBlockN, kBlockK, kThreadM, kThreadN>
      <<<grid, block, 0, stream>>>(m, n, k, kAlpha, mutable_ptr(a),
                                   mutable_ptr(b), kBeta, c);
  return cudaPeekAtLastError();
}

}  // namespace sgemm
