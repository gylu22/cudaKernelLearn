// Archived pre-integration experiment. Not part of the default build.
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

using namespace nvcuda;

#define WARP_SIZE 32 
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define PAD 8
HOST_DEVICE_INLINE
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }


////////////////////////////////////  一个不好的版本 /////////////////////////////////////////
// __device__ __forceinline__ int _swizzle(int row, int col){

//     int swizzle_row = (row / 4 ) % 8; // 0,1，7
//     int swizzle_col = col ^ swizzle_row;
//     return swizzle_col;
// }


// template <const int BM = 128, const int BN = 128, const int BK = 8,
//           const int TM = 8, const int TN = 8>
// __global__ void sgemm_t_8x8_sliced_k_swizzle_f32x4_kernel(float *a, float *b, float *c,
//                                                   int M, int N, int K) {
//     // 转置读取B到shared memory
//     float __shared__ tileA[BM][BK], tileB[BN][BK];
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;

//     float *a_ptr = a + blockIdx.y * BM * K;
//     float *b_ptr = b + blockIdx.x * BN;
//     float *c_ptr = c + blockIdx.y * BM * N + blockIdx.x * BN;
//     int tid = ty * blockDim.x + tx; 

//     float temp[TM][TN] = {0.f};
//     // shared memory 行和列索引计算，对B转置读取
//     // smem_y 行索引 id
//     // smem_x 列索引 id 
//     int smem_y_a = tid / (BK / 4); // 0-127
//     int smem_x_a = (tid % (BK / 4)) * 4 ; // 0,4 
//     // B                                                 
//     int smem_y_b = tid / (BN / 4); // 0,1...,7 
//     int smem_x_b = tid % (BN / 4) * 4; // 0,4,8,..,124                                            
//     // 外层循环（M+BK -1）/ BK
//     for (int k=0; k<K; k+=BK){
//       // 读取tile A 
//       float4 a_vec = FLOAT4(a_ptr[smem_y_a * K + smem_x_a + k]);
//       tileA[smem_y_a][_swizzle(smem_y_a,smem_x_a)] = a_vec.x;
//       tileA[smem_y_a][_swizzle(smem_y_a,smem_x_a+1)] = a_vec.y;
//       tileA[smem_y_a][_swizzle(smem_y_a,smem_x_a+2)] = a_vec.z;
//       tileA[smem_y_a][_swizzle(smem_y_a,smem_x_a+3)] = a_vec.w;

//       // 读取tile B 
//       float4 b_vec = FLOAT4(b_ptr[(k + smem_y_b) * N + smem_x_b]);
//       tileB[smem_x_b][_swizzle(smem_x_b,smem_y_b)] =b_vec.x;
//       tileB[smem_x_b+1][_swizzle(smem_x_b+1,smem_y_b)] =b_vec.y;
//       tileB[smem_x_b+2][_swizzle(smem_x_b+2,smem_y_b)] =b_vec.z;
//       tileB[smem_x_b+3][_swizzle(smem_x_b+3,smem_y_b)] =b_vec.w;
      
//       __syncthreads();
//       // 计算每个线程的BK x BK的区域
//       // 内层循环BK次
//       #pragma unroll
//       for (int i = 0; i < BK; i++) {
//         float tileA_BK[TM];
//         float tileB_BK[TN];
//         #pragma unroll
//         for (int j = 0; j < TM; j++) {
//           // tileA_BK[j] = tileA[i][ty * TM + j];
//           tileA_BK[j] = tileA[ty * TM + j][_swizzle(ty * TM + j,i)];
//         }
//         #pragma unroll
//         for (int j = 0; j < TN; j++) {
//           tileB_BK[j] = tileB[tx * TN + j][_swizzle(tx * TN + j,i)];
//         }
//         #pragma unroll
//         for (int a_i = 0; a_i < TM; a_i++) {
//           for (int b_j = 0; b_j < TN; b_j++) {
//             temp[a_i][b_j] = __fmaf_rn(tileA_BK[a_i], tileB_BK[b_j],temp[a_i][b_j]);
//           }
//         }
//       }
//       // 第二次同步，保证下一次更新shared memory之前已经计算完成
//       __syncthreads();
//       }
//       #pragma unroll
//       for (int i=0;i<TM;i++){
//         for (int j=0;j<TN;j++){
//           c_ptr[(ty * TM + i) * N + tx * TN + j] = temp[i][j];
//         }
//       }
// }




// #define SWIZZLE_FLOAT4(row, col) ((((row) ^ ((col) >> 2)) << 2) + ((col) & 3))

// template <const int BM = 128, const int BN = 128, const int BK = 8,
//           const int TM = 8, const int TN = 8>
// __global__ void sgemm_t_8x8_sliced_k_swizzle_f32x4_kernel(float *a, float *b, float *c,
//                                                   int M, int N, int K) {

//     float __shared__ tileA[BK][BM], tileB[BK][BN];
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;

//     float *a_ptr = a + blockIdx.y * BM * K;
//     float *b_ptr = b + blockIdx.x * BN;
//     float *c_ptr = c + blockIdx.y * BM * N + blockIdx.x * BN;
//     int tid = ty * blockDim.x + tx; // 这里应该是 0-255 (16x16)

//     float temp[TM][TN] = {0.f};
    
//     // 重新映射线程
//     int ta_y = tid / (BK / 4);     // 0-127
//     int ta_x = (tid % (BK / 4)) * 4 ; // 0,4     
//     int tb_y = tid / (BN / 4);     // 0-7
//     int tb_x = (tid % (BN / 4)) * 4;  // 0,4,8...  

//     // 外层循环
//     for (int k=0; k<K; k+=BK){
//       float4 x_vec = FLOAT4(a_ptr[ta_y * K + ta_x + k]);
      
//       // 【改动 1: tileA 写入】
//       // 注意：每一行的 row 都在递增，必须传入正确的行号 (ta_x + offset) 参与异或
//       tileA[ta_x][SWIZZLE_FLOAT4(ta_x, ta_y)]     = x_vec.x;
//       tileA[ta_x+1][SWIZZLE_FLOAT4(ta_x+1, ta_y)] = x_vec.y;
//       tileA[ta_x+2][SWIZZLE_FLOAT4(ta_x+2, ta_y)] = x_vec.z;
//       tileA[ta_x+3][SWIZZLE_FLOAT4(ta_x+3, ta_y)] = x_vec.w;

//       // 【改动 2: tileB 写入】
//       // FLOAT4 指针强转是安全的，因为 tb_x 是 4 的倍数，Swizzle 后依然严格保证 16 字节对齐
//       FLOAT4(tileB[tb_y][SWIZZLE_FLOAT4(tb_y, tb_x)]) = FLOAT4(b_ptr[(k + tb_y) * N + tb_x]);
      
//       __syncthreads();

//       #pragma unroll
//       for (int i = 0; i < BK; i++) {
//         float tileA_BK[TM];
//         float tileB_BK[TN];
        
//         #pragma unroll
//         for (int j = 0; j < TM; j++) {
//           // 【改动 3: tileA 读取】
//           tileA_BK[j] = tileA[i][SWIZZLE_FLOAT4(i, ty * TM + j)];
//         }
//         #pragma unroll
//         for (int j = 0; j < TN; j++) {
//           // 【改动 4: tileB 读取】
//           tileB_BK[j] = tileB[i][SWIZZLE_FLOAT4(i, tx * TN + j)];
//         }
        
//         #pragma unroll
//         for (int a_i = 0; a_i < TM; a_i++) {
//           for (int b_j = 0; b_j < TN; b_j++) {
//             temp[a_i][b_j] = __fmaf_rn(tileA_BK[a_i], tileB_BK[b_j], temp[a_i][b_j]);
//           }
//         }
//       }
//       __syncthreads();
//     }
    
//     // 写回 Global Memory...
//     #pragma unroll
//     for (int i=0;i<TM;i++){
//       for (int j=0;j<TN;j++){
//         c_ptr[(ty * TM + i) * N + tx * TN + j] = temp[i][j];
//       }
//     }
// }

// template <const int BM = 128, const int BN = 128, const int BK = 8,
//           const int TM = 8, const int TN = 8>
// __global__ void sgemm_t_8x8_sliced_k_swizzle_f32x4_kernel(float *a, float *b, float *c,
//                                                   int M, int N, int K) {

//     float __shared__ tileA[BK][BM], tileB[BK][BN];
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;

//     float *a_ptr = a + blockIdx.y * BM * K;
//     float *b_ptr = b + blockIdx.x * BN;
//     float *c_ptr = c + blockIdx.y * BM * N + blockIdx.x * BN;
//     int tid = ty * blockDim.x + tx; // 0-127

//     float temp[TM][TN] = {0.f};
//     // 重新映射线程
//     int ta_y = tid / (BK / 4); // 0-127
//     int ta_x = (tid % (BK / 4)) * 4 ; // 0,4     
//     int tb_y = tid / (BN / 4); // 0-7
//     int tb_x = tid % (BN / 4) * 4; // 0,4,8...  

//     // 外层循环（M+BK -1）/ BK
//     for (int k=0; k<K; k+=BK){
  
//       float4 x_vec = FLOAT4(a_ptr[ta_y * K + ta_x + k]);
//       tileA[ta_x][ta_y] = x_vec.x;
//       tileA[ta_x+1][ta_y] = x_vec.y;
//       tileA[ta_x+2][ta_y] = x_vec.z;
//       tileA[ta_x+3][ta_y] = x_vec.w;

//       FLOAT4(tileB[tb_y][tb_x]) = FLOAT4(b_ptr[(k + tb_y) * N + tb_x]);
      
//       // 第一次同步，保证第一次循环计算之前所有元素都读取到shared memory
//       __syncthreads();
//       // 计算每个线程的BK x BK的区域
//       // 内层循环BK次
//       #pragma unroll
//       for (int i = 0; i < BK; i++) {
//         float tileA_BK[TM];
//         float tileB_BK[TN];

//         FLOAT4(tileA_BK[0]) = FLOAT4(tileA[i][ty * TM ]);
//         FLOAT4(tileA_BK[4]) = FLOAT4(tileA[i][ty * TM + 4]);
//         FLOAT4(tileB_BK[0]) = FLOAT4(tileB[i][tx * TN]);
//         FLOAT4(tileB_BK[4]) = FLOAT4(tileB[i][tx * TN + 4]);

//         #pragma unroll
//         for (int a_i = 0; a_i < TM; a_i++) {
//           for (int b_j = 0; b_j < TN; b_j++) {
//             temp[a_i][b_j] = __fmaf_rn(tileA_BK[a_i], tileB_BK[b_j],temp[a_i][b_j]);
//           }
//         }
//       }
//       // 第二次同步，保证下一次更新shared memory之前已经计算完成
//       __syncthreads();
//       }
//       #pragma unroll
//       for (int i=0;i<TM;i++){
//         for (int j=0;j<TN;j++){
//           c_ptr[(ty * TM + i) * N + tx * TN + j] = temp[i][j];
//         }
//       }
// }

// 核心 Swizzle 宏：以 16 Bytes (4 floats) 为 Chunk，针对宽度为 128 (512 Bytes) 的行
#define SWIZZLE_FLOAT4(row, col) ((((row) ^ ((col) >> 2)) << 2) + ((col) & 3))
#define SWIZZLE_B(col) ((col) ^ ((((col) >> 5) & 3) << 2))
// 为了代码简洁，假设你外部定义了类似这样的宏
// #define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8>
__global__ void sgemm_t_8x8_sliced_k_swizzle_f32x4_kernel(float *a, float *b, float *c,
                                                  int M, int N, int K) {

    float __shared__ tileA[BK][BM], tileB[BK][BN];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float *a_ptr = a + blockIdx.y * BM * K;
    float *b_ptr = b + blockIdx.x * BN;
    float *c_ptr = c + blockIdx.y * BM * N + blockIdx.x * BN;
    int tid = ty * blockDim.x + tx; // 这里应该是 0-255 (block size 16x16)

    float temp[TM][TN] = {0.f};
    
    // 重新映射线程 (Global to Shared)
    int ta_y = tid / (BK / 4);     // 0-127
    int ta_x = (tid % (BK / 4)) * 4 ; // 0, 4     
    int tb_y = tid / (BN / 4);     // 0-7
    int tb_x = (tid % (BN / 4)) * 4;  // 0, 4, 8...  

    // 外层循环
    for (int k = 0; k < K; k += BK) {
  
      float4 x_vec = FLOAT4(a_ptr[ta_y * K + ta_x + k]);
      
      // 【修改点 1：TileA 写入 Swizzle】
      // 注意：ta_x 才是行号 (BK维度)，ta_y 是列号 (BM维度)
      tileA[ta_x][SWIZZLE_FLOAT4(ta_x, ta_y)]     = x_vec.x;
      tileA[ta_x+1][SWIZZLE_FLOAT4(ta_x+1, ta_y)] = x_vec.y;
      tileA[ta_x+2][SWIZZLE_FLOAT4(ta_x+2, ta_y)] = x_vec.z;
      tileA[ta_x+3][SWIZZLE_FLOAT4(ta_x+3, ta_y)] = x_vec.w;

      // 【修改点 2：TileB 写入 Swizzle】
      // FLOAT4 对齐安全，因为 tb_x 是 4 的倍数，Swizzle 后依然是 4 的倍数
      FLOAT4(tileB[tb_y][SWIZZLE_B(tb_x)]) = FLOAT4(b_ptr[(k + tb_y) * N + tb_x]);
      
      // 第一次同步
      __syncthreads();
      
      // 内层循环 BK 次
      #pragma unroll
      for (int i = 0; i < BK; i++) {
        float tileA_BK[TM];
        float tileB_BK[TN];
        
        // 【修改点 3：TileA & TileB 向量化读取 Swizzle】
        // 行号都是 i，列号分别是 ty * TM (及 +4) 和 tx * TN (及 +4)
        FLOAT4(tileA_BK[0]) = FLOAT4(tileA[i][SWIZZLE_FLOAT4(i, ty * TM)]);
        FLOAT4(tileA_BK[4]) = FLOAT4(tileA[i][SWIZZLE_FLOAT4(i, ty * TM + 4)]);
        
        FLOAT4(tileB_BK[0]) = FLOAT4(tileB[i][SWIZZLE_B(tx * TN)]);
        FLOAT4(tileB_BK[4]) = FLOAT4(tileB[i][SWIZZLE_B(tx * TN + 4)]);

        #pragma unroll
        for (int a_i = 0; a_i < TM; a_i++) {
          for (int b_j = 0; b_j < TN; b_j++) {
            temp[a_i][b_j] = __fmaf_rn(tileA_BK[a_i], tileB_BK[b_j], temp[a_i][b_j]);
          }
        }
      }
      // 第二次同步
      __syncthreads();
    }
    
    // 将寄存器 temp 写回 Global Memory
    #pragma unroll
    for (int i = 0; i < TM; i++){
      for (int j = 0; j < TN; j++){
        c_ptr[(ty * TM + i) * N + tx * TN + j] = temp[i][j];
      }
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8>
__global__ void sgemm_t_8x8_sliced_k_bcf_db_swizzle_f32x4_kernel(float *a, float *b, float *c,
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

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }




void sgemm_t_8x8_sliced_k_swizzle_f32x4(torch::Tensor a, torch::Tensor b,
                              torch::Tensor c) {
CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
const int M = a.size(0);
const int K = a.size(1);
const int N = b.size(1);
CHECK_TORCH_TENSOR_SHAPE(a, M, K)
CHECK_TORCH_TENSOR_SHAPE(b, K, N)
CHECK_TORCH_TENSOR_SHAPE(c, M, N)
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;

dim3 block(BN / TN, BM / TM);
dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

sgemm_t_8x8_sliced_k_swizzle_f32x4_kernel<BM, BN, BK, TM, TN>
    <<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),
                      reinterpret_cast<float *>(b.data_ptr()),
                      reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

