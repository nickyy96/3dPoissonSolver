#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <mpi.h>
#include <fstream>

// Include HIP headers
#include <hip/hip_runtime.h>

// GPU error checking macro
#define GPU_CHECK(error)                                                                         \
  if (error != cudaSuccess)                                                                      \
  {                                                                                              \
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(error), __FILE__, __LINE__); \
    exit(EXIT_FAILURE);                                                                          \
  }

#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaStream_t hipStream_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError

__global__ void update_back_boundary_kernel(
    double *u_boundary, const double *u_boundary_old,
    const double *u_interior, const double *u_ghost,
    const double *rhs, double h_sq,
    int nx, int ny, int nz, int iter,
    int west, int east, int back, int front, int south, int north,
    double *d_block_max_diffs, int init_block_idx)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  double local_diff = 0.0;

  if (i < ny && j < nz)
  {
    int boundary_idx = i * nz + j;
    int global_idx = 0 * ny * nz + i * nz + j;

    // red black
    if ((i + j) % 2 == (iter % 2))
    {
      // indices of 6 neighbors

      // ghost of west is left, (i, j - 1)
      double u_left = (j == 0) ? u_ghost[west + i * nx] : u_boundary_old[back + boundary_idx - 1];
      // ghost of east is right, (i, j + 1)
      double u_right = (j == nz - 1) ? u_ghost[east + i * nx] : u_boundary_old[back + boundary_idx + 1];
      // ghost of north is up, (i - 1, j)
      double u_up = (i == 0) ? u_ghost[north + j] : u_boundary_old[back + boundary_idx - nz];
      // ghost of south is down, (i + 1, j)
      double u_down = (i == ny - 1) ? u_ghost[south + j] : u_boundary_old[back + boundary_idx + nz];
      // constant
      double u_back = u_ghost[back + boundary_idx];
      // either boundary on north at (i + 1, j), south at (i + 1, j), or interior
      // also need to handle case where we are on j edge, similar logic but with other faces
      double u_front;
      if (i == 0)
      {
        u_front = u_boundary_old[north + j + nz];
      }
      else if (i == ny - 1)
      {
        u_front = u_boundary_old[south + j + nz];
      }
      else if (j == 0)
      {
        u_front = u_boundary_old[west + i * nx + 1];
      }
      else if (j == nz - 1)
      {
        u_front = u_boundary_old[east + i * nx + 1];
      }
      else
      {
        // (i - 1) * (ny - 2) * (nz - 2) + (j - 1) * (nz - 2) + (k - 1)
        u_front = u_interior[(i - 1) * (nz - 2) + (j - 1)];
      }

      // need to compute global rhs
      // on back plane this is actually the same
      double rhs_val = rhs[global_idx];

      // double val = ((u_left + u_right) + (u_down + u_up) + (u_back + u_front) - rhs_val * h_sq) / 6.0;
      double val = ((u_left + u_right) - rhs_val * h_sq) / 6.0;
      local_diff = fabs(val - u_boundary_old[boundary_idx]);
      u_boundary[boundary_idx] = val;
    }
  }

  __shared__ double sdata[625];
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  sdata[tid] = local_diff;
  __syncthreads();

  int n = blockDim.x * blockDim.y;
  for (int s = n / 2; s > 0; s >>= 1)
  {
    if (tid < s)
    {
      double val_other = sdata[tid + s];
      if (val_other > sdata[tid])
      {
        sdata[tid] = val_other;
      }
    }
    __syncthreads();
  }

  if (tid == 0)
  {
    int block_idx = init_block_idx + blockIdx.y * gridDim.x + blockIdx.x;
    d_block_max_diffs[block_idx] = sdata[0];
  }
}

__global__ void update_front_boundary_kernel(
    double *u_boundary, const double *u_boundary_old,
    const double *u_interior, const double *u_ghost,
    const double *rhs, double h_sq,
    int nx, int ny, int nz, int iter,
    int west, int east, int back, int front, int south, int north,
    double *d_block_max_diffs, int init_block_idx)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  double local_diff = 0.0;

  if (i < ny && j < nz)
  {
    int boundary_idx = front + i * nz + j;
    // last i element is front plane
    // holding x at nx - 1
    int global_idx = (nx - 1) * ny * nz + i * nz + j;

    // red black
    if ((i + j) % 2 == (iter % 2))
    {
      // indices of 6 neighbors

      // ghost of west is left, but at far right of xy, (i, j - 1)
      double u_left = (j == 0) ? u_ghost[west + i * nx + (nx - 1)] : u_boundary_old[boundary_idx - 1];
      // ghost of east is right, but at far right of xy, (i, j + 1)
      double u_right = (j == nz - 1) ? u_ghost[east + i * nx + (nx - 1)] : u_boundary_old[boundary_idx + 1];
      // ghost of north is up, but at bottom of xz plane, (i - 1, j)
      double u_up = (i == 0) ? u_ghost[north + (nx - 1) * nz + j] : u_boundary_old[boundary_idx - nz];
      // ghost of south is down, but at bottom of xz plane, (i + 1, j)
      double u_down = (i == ny - 1) ? u_ghost[south + (nx - 1) * nz + j] : u_boundary_old[boundary_idx + nz];
      // constant
      double u_front = u_ghost[boundary_idx];
      // either boundary on north at (i + 1, j), south at (i + 1, j), or interior
      // also need to handle case where we are on j edge, similar logic but with other faces
      double u_back;
      if (i == 0)
      {
        u_back = u_boundary_old[north + (nx - 2) * nz + j];
      }
      else if (i == ny - 1)
      {
        u_back = u_boundary_old[south + (nx - 2) * nx + j];
      }
      else if (j == 0)
      {
        u_back = u_boundary_old[west + i * nx + nx - 2];
      }
      else if (j == nz - 1)
      {
        u_back = u_boundary_old[east + i * nx + nx - 2];
      }
      else
      {
        // 0 -> because we are at the back plane
        // nx - 3 -> because we are at the front plane
        // (i - 1) * (ny - 2) * (nz - 2) + (j - 1) * (nz - 2) + (k - 1)
        u_back = u_interior[(nx - 3) * (ny - 2) * (nz - 2) + (i - 1) * (nz - 2) + (j - 1)];
      }

      // need to compute global rhs
      // on back plane this is actually the same
      double rhs_val = rhs[global_idx];

      double val = ((u_left + u_right) + (u_down + u_up) + (u_back + u_front) - rhs_val * h_sq) / 6.0;
      local_diff = fabs(val - u_boundary_old[boundary_idx]);
      u_boundary[boundary_idx] = val;
    }
  }

  __shared__ double sdata[625];
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  sdata[tid] = local_diff;
  __syncthreads();

  int n = blockDim.x * blockDim.y;
  for (int s = n / 2; s > 0; s >>= 1)
  {
    if (tid < s)
    {
      double val_other = sdata[tid + s];
      if (val_other > sdata[tid])
      {
        sdata[tid] = val_other;
      }
    }
    __syncthreads();
  }

  if (tid == 0)
  {
    int block_idx = init_block_idx + blockIdx.y * gridDim.x + blockIdx.x;
    d_block_max_diffs[block_idx] = sdata[0];
  }
}

__global__ void update_west_boundary_kernel(
    double *u_boundary, const double *u_boundary_old,
    const double *u_interior, const double *u_ghost,
    const double *rhs, double h_sq,
    int nx, int ny, int nz, int iter,
    int west, int east, int back, int front, int south, int north,
    bool ignore_x_lower, bool ignore_x_upper,
    double *d_block_max_diffs, int init_block_idx)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  double local_diff = 0.0;

  // we do not want to process if we are on edge of dirichlet boundary
  if (i < ny && j < nx && (j != 0 || !ignore_x_lower) && (j != nx - 1 || !ignore_x_upper))
  {
    int boundary_idx = west + i * nx + j;
    // holding z at 0
    int global_idx = j * ny * nz + i * nz + 0;

    // red black
    if ((i + j) % 2 == (iter % 2))
    {
      // indices of 6 neighbors

      // ghost of back is left, (i, j - 1)
      double u_left = (j == 0) ? u_ghost[back + i * nz] : u_boundary_old[boundary_idx - 1];
      // ghost of front is right, (i, j + 1)
      double u_right = (j == nz - 1) ? u_ghost[front + i * nz] : u_boundary_old[boundary_idx + 1];
      // ghost of north is up, (i - 1, j)
      double u_up = (i == 0) ? u_ghost[north + j * nz] : u_boundary_old[boundary_idx - nx];
      // ghost of south is down, (i + 1, j)
      double u_down = (i == ny - 1) ? u_ghost[south + j * nz] : u_boundary_old[boundary_idx + nx];
      // constant
      double u_back = u_ghost[boundary_idx];
      // either boundary on north at (i, j + 1), south at (i, j + 1), or interior
      // also need to handle case where we are on j edge, similar logic but with other faces
      double u_front;
      if (i == 0)
      {
        u_front = u_boundary_old[north + j * nz + 1];
      }
      else if (i == ny - 1)
      {
        u_front = u_boundary_old[south + j * nz + 1];
      }
      else if (j == 0)
      {
        u_front = u_boundary_old[back + i * nz + 1];
      }
      else if (j == nz - 1)
      {
        u_front = u_boundary_old[front + i * nz + 1];
      }
      else
      {
        // (i - 1) * (ny - 2) * (nz - 2) + (j - 1) * (nz - 2) + (k - 1)
        u_front = u_interior[(j - 1) * (ny - 2) * (nz - 2) + (i - 1) * (nz - 2)];
      }

      // need to compute global rhs
      // on back plane this is actually the same
      double rhs_val = rhs[global_idx];

      double val = ((u_left + u_right) + (u_down + u_up) + (u_back + u_front) - rhs_val * h_sq) / 6.0;
      local_diff = fabs(val - u_boundary_old[boundary_idx]);
      u_boundary[boundary_idx] = val;
    }
  }

  __shared__ double sdata[625];
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  sdata[tid] = local_diff;
  __syncthreads();

  int n = blockDim.x * blockDim.y;
  for (int s = n / 2; s > 0; s >>= 1)
  {
    if (tid < s)
    {
      double val_other = sdata[tid + s];
      if (val_other > sdata[tid])
      {
        sdata[tid] = val_other;
      }
    }
    __syncthreads();
  }

  if (tid == 0)
  {
    int block_idx = init_block_idx + blockIdx.y * gridDim.x + blockIdx.x;
    d_block_max_diffs[block_idx] = sdata[0];
  }
}

__global__ void update_east_boundary_kernel(
    double *u_boundary, const double *u_boundary_old,
    const double *u_interior, const double *u_ghost,
    const double *rhs, double h_sq,
    int nx, int ny, int nz, int iter,
    int west, int east, int back, int front, int south, int north,
    bool ignore_x_lower, bool ignore_x_upper,
    double *d_block_max_diffs, int init_block_idx)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  double local_diff = 0.0;

  // we do not want to process if we are on edge of dirichlet boundary
  if (i < ny && j < nx && (j != 0 || !ignore_x_lower) && (j != nx - 1 || !ignore_x_upper))
  {
    int boundary_idx = east + i * nx + j;
    // holding z at nz - 1
    int global_idx = j * ny * nz + i * nz + (nz - 1);

    // red black
    if ((i + j) % 2 == (iter % 2))
    {
      // indices of 6 neighbors

      // ghost of back is left, but at far right of yz, (i, j - 1)
      double u_left = (j == 0) ? u_ghost[back + i * nz + (nz - 1)] : u_boundary_old[boundary_idx - 1];
      // ghost of front is right, but at far right of xy, (i, j + 1)
      double u_right = (j == nz - 1) ? u_ghost[front + i * nz + (nz - 1)] : u_boundary_old[boundary_idx + 1];
      // ghost of north is up, but at far right of xz, (i - 1, j)
      double u_up = (i == 0) ? u_ghost[north + j * nz + (nz - 1)] : u_boundary_old[boundary_idx - nx];
      // ghost of south is down, but at far right of xz, (i + 1, j)
      double u_down = (i == ny - 1) ? u_ghost[south + j * nz + (nz - 1)] : u_boundary_old[boundary_idx + nx];
      // constant
      double u_front = u_ghost[boundary_idx];
      // either boundary on north at (i, j + 1), south at (i, j + 1), or interior
      // also need to handle case where we are on j edge, similar logic but with other faces
      double u_back;
      if (i == 0)
      {
        u_back = u_boundary_old[north + j * nz + (nz - 2)];
      }
      else if (i == ny - 1)
      {
        u_back = u_boundary_old[south + j * nz + (nz - 2)];
      }
      else if (j == 0)
      {
        u_back = u_boundary_old[back + i * nz + (nz - 2)];
      }
      else if (j == nz - 1)
      {
        u_back = u_boundary_old[front + i * nz + (nz - 2)];
      }
      else
      {
        // (i - 1) * (ny - 2) * (nz - 2) + (j - 1) * (nz - 2) + (k - 1)
        u_back = u_interior[(j - 1) * (ny - 2) * (nz - 2) + (i - 1) * (nz - 2) + (nz - 3)];
      }

      // need to compute global rhs
      // on back plane this is actually the same
      double rhs_val = rhs[global_idx];

      double val = ((u_left + u_right) + (u_down + u_up) + (u_back + u_front) - rhs_val * h_sq) / 6.0;
      local_diff = fabs(val - u_boundary_old[boundary_idx]);
      u_boundary[boundary_idx] = val;
    }
  }

  __shared__ double sdata[625];
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  sdata[tid] = local_diff;
  __syncthreads();

  int n = blockDim.x * blockDim.y;
  for (int s = n / 2; s > 0; s >>= 1)
  {
    if (tid < s)
    {
      double val_other = sdata[tid + s];
      if (val_other > sdata[tid])
      {
        sdata[tid] = val_other;
      }
    }
    __syncthreads();
  }

  if (tid == 0)
  {
    int block_idx = init_block_idx + blockIdx.y * gridDim.x + blockIdx.x;
    d_block_max_diffs[block_idx] = sdata[0];
  }
}

__global__ void update_north_boundary_kernel(
    double *u_boundary, const double *u_boundary_old,
    const double *u_interior, const double *u_ghost,
    const double *rhs, double h_sq,
    int nx, int ny, int nz, int iter,
    int west, int east, int back, int front, int south, int north,
    bool ignore_x_lower, bool ignore_x_upper,
    double *d_block_max_diffs, int init_block_idx)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  double local_diff = 0.0;

  // we do not want to process if we are on edge of dirichlet boundary
  if (i < ny && j < nx && (i != 0 || !ignore_x_lower) && (i != nx - 1 || !ignore_x_upper))
  {
    int boundary_idx = north + i * nz + j;
    // holding y at 0
    int global_idx = i * ny * nz + j;

    // red black
    if ((i + j) % 2 == (iter % 2))
    {
      // indices of 6 neighbors

      // ghost of west is left, but at far right of xy, (i, j - 1)
      double u_left = (j == 0) ? u_ghost[west + i * ny + (ny - 1)] : u_boundary_old[boundary_idx - 1];
      // ghost of front is right, (i, j + 1)
      double u_right = (j == nz - 1) ? u_ghost[east + i * ny + (ny - 1)] : u_boundary_old[boundary_idx + 1];
      // ghost of back is up, (i - 1, j)
      double u_up = (i == 0) ? u_ghost[back + j] : u_boundary_old[boundary_idx - nz];
      // ghost of south is down, (i + 1, j)
      double u_down = (i == nx - 1) ? u_ghost[front + j] : u_boundary_old[boundary_idx + nz];
      // constant
      double u_front = u_ghost[boundary_idx];
      // either boundary on back at (i, j + 1), front at (i, j + 1), or interior
      // also need to handle case where we are on j edge, similar logic but with other faces
      double u_back;
      if (i == 0)
      {
        u_back = u_boundary_old[back + j + nz];
      }
      else if (i == nx - 1)
      {
        u_back = u_boundary_old[front + j + nz];
      }
      else if (j == 0)
      {
        u_back = u_boundary_old[west + i * ny + (ny - 2)];
      }
      else if (j == ny - 1)
      {
        u_back = u_boundary_old[east + i * ny + (ny - 2)];
      }
      else
      {
        // (i - 1) * (ny - 2) * (nz - 2) + (j - 1) * (nz - 2) + (k - 1)
        u_back = u_interior[(i - 1) * (ny - 2) * (nz - 2) + (j - 1)];
      }

      // need to compute global rhs
      // on back plane this is actually the same
      double rhs_val = rhs[global_idx];

      double val = ((u_left + u_right) + (u_down + u_up) + (u_back + u_front) - rhs_val * h_sq) / 6.0;
      local_diff = fabs(val - u_boundary_old[boundary_idx]);
      u_boundary[boundary_idx] = val;
    }
  }

  __shared__ double sdata[625];
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  sdata[tid] = local_diff;
  __syncthreads();

  int n = blockDim.x * blockDim.y;
  for (int s = n / 2; s > 0; s >>= 1)
  {
    if (tid < s)
    {
      double val_other = sdata[tid + s];
      if (val_other > sdata[tid])
      {
        sdata[tid] = val_other;
      }
    }
    __syncthreads();
  }

  if (tid == 0)
  {
    int block_idx = init_block_idx + blockIdx.y * gridDim.x + blockIdx.x;
    d_block_max_diffs[block_idx] = sdata[0];
  }
}

__global__ void update_south_boundary_kernel(
    double *u_boundary, const double *u_boundary_old,
    const double *u_interior, const double *u_ghost,
    const double *rhs, double h_sq,
    int nx, int ny, int nz, int iter,
    int west, int east, int back, int front, int south, int north,
    bool ignore_x_lower, bool ignore_x_upper,
    double *d_block_max_diffs, int init_block_idx)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  double local_diff = 0.0;

  // we do not want to process if we are on edge of dirichlet boundary
  if (i < ny && j < nx && (i != 0 || !ignore_x_lower) && (i != nx - 1 || !ignore_x_upper))
  {
    int boundary_idx = south + i * nz + j;
    // holding y at ny - 1
    int global_idx = i * ny * nz + (ny - 1) * nz + j;

    // red black
    if ((i + j) % 2 == (iter % 2))
    {
      // indices of 6 neighbors

      // ghost of west is left, (i, j - 1)
      double u_left = (j == 0) ? u_ghost[west + i * ny] : u_boundary_old[boundary_idx - 1];
      // ghost of front is right, (i, j + 1)
      double u_right = (j == nz - 1) ? u_ghost[east + i * ny] : u_boundary_old[boundary_idx + 1];
      // ghost of back is up, but at bottom of yz plane, (i - 1, j)
      double u_up = (i == 0) ? u_ghost[back + j + (ny - 1) * nz] : u_boundary_old[boundary_idx - nz];
      // ghost of south is down, (i + 1, j)
      double u_down = (i == nx - 1) ? u_ghost[front + j + (ny - 1) * nz] : u_boundary_old[boundary_idx + nz];
      // constant
      double u_back = u_ghost[boundary_idx];
      // either boundary on back at (i, j + 1), front at (i, j + 1), or interior
      // also need to handle case where we are on j edge, similar logic but with other faces
      double u_front;
      if (i == 0)
      {
        u_front = u_boundary_old[back + j + (ny - 2) * nz];
      }
      else if (i == nx - 1)
      {
        u_front = u_boundary_old[front + j + (ny - 2) * nz];
      }
      else if (j == 0)
      {
        u_front = u_boundary_old[west + i * ny + 1];
      }
      else if (j == ny - 1)
      {
        u_front = u_boundary_old[east + i * ny + 1];
      }
      else
      {
        // (i - 1) * (ny - 2) * (nz - 2) + (j - 1) * (nz - 2) + (k - 1)
        u_front = u_interior[(i - 1) * (ny - 2) * (nz - 2) + (ny - 3) * (nz - 2) + (j - 1)];
      }

      // need to compute global rhs
      // on back plane this is actually the same
      double rhs_val = rhs[global_idx];

      double val = ((u_left + u_right) + (u_down + u_up) + (u_back + u_front) - rhs_val * h_sq) / 6.0;
      local_diff = fabs(val - u_boundary_old[boundary_idx]);
      u_boundary[boundary_idx] = val;
    }
  }

  __shared__ double sdata[625];
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  sdata[tid] = local_diff;
  __syncthreads();

  int n = blockDim.x * blockDim.y;
  for (int s = n / 2; s > 0; s >>= 1)
  {
    if (tid < s)
    {
      double val_other = sdata[tid + s];
      if (val_other > sdata[tid])
      {
        sdata[tid] = val_other;
      }
    }
    __syncthreads();
  }

  if (tid == 0)
  {
    int block_idx = init_block_idx + blockIdx.y * gridDim.x + blockIdx.x;
    d_block_max_diffs[block_idx] = sdata[0];
  }
}

// Kernel to perform one iteration on interior
__global__ void update_interior_kernel(
    double *u_interior, const double *u_interior_old, const double *u_boundary,
    const double *rhs, double h_sq,
    int nx, int ny, int nz, int iter,
    int back, int front, int north, int south, int west, int east,
    double *d_block_max_diffs)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  double local_diff = 0.0;

  if (i < nx - 2 &&
      j < ny - 2 &&
      k < nz - 2)
  {
    // index in the 1D array
    int interior_idx = i * ((ny - 2) * (nz - 2)) + j * (nz - 2) + k;
    int global_idx = (i + 1) * (ny * nz) + (j + 1) * nz + (k + 1);

    // red black
    if ((i + j + k) % 2 == (iter % 2))
    {
      // figure out where we need to pull from
      bool on_left = (k == 0);
      bool on_right = (k == nz - 3);
      bool on_up = (j == 0);
      bool on_down = (j == ny - 3);
      bool on_back = (i == 0);
      bool on_front = (i == nx - 3);

      // indices of 6 neighbors
      double u_left = on_left ? u_boundary[west + (j + 1) * nx + (i + 1)] : u_interior_old[interior_idx - 1];
      double u_right = on_right ? u_boundary[east + (j + 1) * nx + (i + 1)] : u_interior_old[interior_idx + 1];

      double u_up = on_up ? u_boundary[north + (i + 1) * nz + (k + 1)] : u_interior_old[interior_idx - (nz - 2)];
      double u_down = on_down ? u_boundary[south + (i + 1) * nz + (k + 1)] : u_interior_old[interior_idx + (nz - 2)];

      double u_back = on_back ? u_boundary[back + (j + 1) * nz + (k + 1)] : u_interior_old[interior_idx - ((ny - 2) * (nz - 2))];
      double u_front = on_front ? u_boundary[front + (j + 1) * nz + (k + 1)] : u_interior_old[interior_idx + ((ny - 2) * (nz - 2))];

      double rhs_val = rhs[global_idx];

      double val = ((u_left + u_right) + (u_down + u_up) + (u_back + u_front) - rhs_val * h_sq) / 6.0;
      local_diff = fabs(val - u_interior_old[interior_idx]);
      u_interior[interior_idx] = val;
    }
  }

  __shared__ double sdata[512];
  int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
  sdata[tid] = local_diff;
  __syncthreads();

  int n = blockDim.x * blockDim.y * blockDim.z;
  for (int s = n / 2; s > 0; s >>= 1)
  {
    if (tid < s)
    {
      double val_other = sdata[tid + s];
      if (val_other > sdata[tid])
      {
        sdata[tid] = val_other;
      }
    }
    __syncthreads();
  }

  if (tid == 0)
  {
    int block_idx = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
    d_block_max_diffs[block_idx] = sdata[0];
  }
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int my_rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // init gpu device
  int ndevices = 0;
  hipGetDeviceCount(&ndevices);

  int my_device = my_rank % ndevices;
  hipSetDevice(my_device);

  // 3d constants
  const int N = 100;
  const double start = 0.0;
  const double end = 1.0;
  const double h = (end - start) / (N - 1);
  const double h_squared = h * h;
  const double tol = 1e-6;
  const int max_iter = 20000;
  const double pi = acos(-1.0);
  const int n = 2, m = 2, l = 2;

  // Create a 3D Cartesian communicator
  int dims[3] = {0, 0, 0};
  MPI_Dims_create(size, 3, dims);
  int periods[3] = {0, 1, 1};
  MPI_Comm my_cart_dim;
  MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &my_cart_dim);

  int coords[3];
  MPI_Cart_coords(my_cart_dim, my_rank, 3, coords);

  // Determine neighbors in the Cartesian topology
  int north, south, east, west, front, back;
  // x-dir, y-dir, z-dir
  // in the x direction, our design suggests that its boundaries are the yz planes at the front and back
  // y direction -> north and south
  // z direction -> east and west
  MPI_Cart_shift(my_cart_dim, 0, 1, &back, &front);
  MPI_Cart_shift(my_cart_dim, 1, 1, &north, &south);
  MPI_Cart_shift(my_cart_dim, 2, 1, &west, &east);
  
  // Local grid sizes
  int local_Nx = N / dims[0];
  int local_Ny = N / dims[1];
  int local_Nz = N / dims[2];

  // Add ghost layers
  int local_Nx_with_ghosts = local_Nx + 2;
  int local_Ny_with_ghosts = local_Ny + 2;
  int local_Nz_with_ghosts = local_Nz + 2;

  // Starting integers
  int x_start_idx = coords[0] * local_Nx;
  int y_start_idx = coords[1] * local_Ny;
  int z_start_idx = coords[2] * local_Nz;

  // Function to map 2D indices to 1D index
  auto idx = [=](int i, int j, int k)
  {
    return i * (local_Ny * local_Nz) + j * local_Nz + k;
  };

  // total size of problem
  int total_size = local_Nx * local_Ny * local_Nz;
  // interior of problem -> no boundaries
  int interior_size = (local_Nx - 2) * (local_Ny - 2) * (local_Nz - 2);

  // 2d sizes for face exchanges
  int xy_size = local_Nx * local_Ny;
  int xz_size = local_Nx * local_Nz;
  int yz_size = local_Ny * local_Nz;
  int boundary_size = 2 * xy_size + 2 * xz_size + 2 * yz_size;

  double *u = new double[total_size];
  double *rhs = new double[total_size];
  double *exact = new double[total_size];

  // interior arrays
  double *u_interior = new double[interior_size];
  double *u_interior_old = new double[interior_size];

  // boundary arrays
  double *u_boundary = new double[boundary_size];
  double *u_boundary_old = new double[boundary_size];
  double *u_ghost = new double[boundary_size];
  // double *u_back = new double[yz_size];
  // double *u_front = new double[yz_size];
  // double *u_north = new double[xz_size];
  // double *u_south = new double[xz_size];
  // double *u_west = new double[xy_size];
  // double *u_east = new double[xy_size];
  int u_back = 0;
  int u_front = yz_size;

  int u_north = yz_size * 2;
  int u_south = yz_size * 2 + xz_size;

  int u_west = yz_size * 2 + xz_size * 2;
  int u_east = yz_size * 2 + xz_size * 2 + xy_size;

  // Residuals and errors
  std::vector<double> avg_residual_per_iter;
  std::vector<double> avg_error_per_iter;

  // Initialize arrays
  for (int i = 0; i < total_size; i++)
  {
    rhs[i] = 0.0;
    exact[i] = 0.0;
  }

  // Compute rhs and exact solution
  for (int i = 0; i < local_Nx; ++i)
  {
    int global_i = x_start_idx + i;
    double x = start + global_i * h;
    for (int j = 0; j < local_Ny; ++j)
    {
      int global_j = y_start_idx + j;
      double y = start + global_j * h;
      for (int k = 0; k < local_Nz; ++k)
      {
        int global_k = z_start_idx + k;

        double z = start + global_k * h;
        double val = sin(n * pi * x) * cos(m * pi * y) * sin(l * pi * z);
        exact[idx(i, j, k)] = val;

        double lap = -(n * n + m * m + l * l) * pi * pi * val;
        rhs[idx(i, j, k)] = lap;
      }
    }
  }

  if (coords[0] == dims[0] - 1)
  {
    // need to do this so we populate boundary data at front
    for (int i = 0; i < local_Ny; ++i)
    {
      for (int j = 0; j < local_Nz; ++j)
      {
        u_boundary[u_front + i * local_Nz + j] = exact[idx(local_Nx - 1, i, j)];
      }
    }
  }

  if (coords[0] == 0)
  {
    for (int i = 0; i < local_Ny; ++i)
    {
      for (int j = 0; j < local_Nz; ++j)
      {
        u_boundary[u_back + i * local_Nz + j] = exact[idx(0, i, j)];
      }
    }
  }

  // device arrays
  double *d_u_interior, *d_u_interior_old, *d_u_boundary, *d_u_boundary_old, *d_u_ghost;
  double *d_rhs, *d_block_max_diffs, *d_block_max_diffs_boundary;
  // allocate and copy
  GPU_CHECK(cudaMalloc(&d_u_interior, interior_size * sizeof(double)));
  GPU_CHECK(cudaMalloc(&d_u_interior_old, interior_size * sizeof(double)));
  GPU_CHECK(cudaMalloc(&d_u_boundary, boundary_size * sizeof(double)));
  GPU_CHECK(cudaMalloc(&d_u_boundary_old, boundary_size * sizeof(double)));
  GPU_CHECK(cudaMalloc(&d_u_ghost, boundary_size * sizeof(double)));
  GPU_CHECK(cudaMalloc(&d_rhs, total_size * sizeof(double)));

  GPU_CHECK(cudaMemcpy(d_u_interior, u_interior, interior_size * sizeof(double), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_u_interior_old, u_interior, interior_size * sizeof(double), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_u_boundary, u_boundary, boundary_size * sizeof(double), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_u_boundary_old, u_boundary, boundary_size * sizeof(double), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_u_ghost, u_ghost, boundary_size * sizeof(double), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_rhs, rhs, total_size * sizeof(double), cudaMemcpyHostToDevice));

  // Set up kernel launch parameters
  dim3 interior_block_size(8, 8, 8);
  dim3 interior_grid_size(((local_Nx - 2) + interior_block_size.x - 1) / interior_block_size.x,
                          ((local_Ny - 2) + interior_block_size.y - 1) / interior_block_size.y,
                          ((local_Nz - 2) + interior_block_size.z - 1) / interior_block_size.z);

  dim3 boundary_block_size(25, 25);
  dim3 xy_grid_size((local_Nx + boundary_block_size.x - 1) / boundary_block_size.x,
                    (local_Ny + boundary_block_size.y - 1) / boundary_block_size.y);
  dim3 xz_grid_size((local_Nx + boundary_block_size.x - 1) / boundary_block_size.x,
                    (local_Nz + boundary_block_size.y - 1) / boundary_block_size.y);
  dim3 yz_grid_size((local_Ny + boundary_block_size.x - 1) / boundary_block_size.x,
                    (local_Nz + boundary_block_size.y - 1) / boundary_block_size.y);

  int num_interior_blocks = interior_grid_size.x * interior_grid_size.y * interior_grid_size.z;
  int num_xy_blocks = xy_grid_size.x * xy_grid_size.y;
  int num_xz_blocks = xz_grid_size.x * xz_grid_size.y;
  int num_yz_blocks = yz_grid_size.x * yz_grid_size.y;

  int num_back_idx = 0;
  int num_front_idx = num_yz_blocks;
  int num_west_idx = num_yz_blocks * 2;
  int num_east_idx = num_yz_blocks * 2 + num_xy_blocks;
  int num_north_idx = num_yz_blocks * 2 + num_xy_blocks * 2;
  int num_south_idx = num_yz_blocks * 2 + num_xy_blocks * 2 + num_xz_blocks;
  int num_boundary_blocks = 2 * num_xy_blocks + 2 * num_xz_blocks + 2 * num_yz_blocks;

  double *block_max_diffs = new double[num_interior_blocks];
  double *block_max_diffs_boundary = new double[num_boundary_blocks];

  GPU_CHECK(cudaMalloc(&d_block_max_diffs, num_interior_blocks * sizeof(double)));
  GPU_CHECK(cudaMalloc(&d_block_max_diffs_boundary, num_boundary_blocks * sizeof(double)));

  cudaStream_t stream_interior, stream_back, stream_front, stream_east, stream_west, stream_north, stream_south;
  cudaStreamCreate(&stream_interior);
  cudaStreamCreate(&stream_back);
  cudaStreamCreate(&stream_front);
  cudaStreamCreate(&stream_east);
  cudaStreamCreate(&stream_west);
  cudaStreamCreate(&stream_north);
  cudaStreamCreate(&stream_south);

  // MPI datatypes for face exchanges
  MPI_Datatype yz_plane_type;
  MPI_Type_contiguous(yz_size, MPI_DOUBLE, &yz_plane_type);
  MPI_Type_commit(&yz_plane_type);

  MPI_Datatype xz_plane_type;
  MPI_Type_contiguous(xz_size, MPI_DOUBLE, &xz_plane_type);
  MPI_Type_commit(&xz_plane_type);

  MPI_Datatype xy_plane_type;
  MPI_Type_contiguous(xy_size, MPI_DOUBLE, &xy_plane_type);
  MPI_Type_commit(&xy_plane_type);

  // Performance Metrics (exported to CSV)
  int num_internal_points = N * N * (N - 2);
  int flops_per_point = 13;                 // 7 adds, 3 subtractions, 2 multiplication, 1 division
  int bytes_per_point = 8 * sizeof(double); // reads and writes from red black (averaged at 7.5 and rounded up)

  // Iterative update
  double diff = std::numeric_limits<double>::infinity();
  int iter = 0;

  double start_time = MPI_Wtime();

  while (diff > tol && iter < max_iter)
  {
    diff = 0.0;

    // point swap
    double *temp = d_u_interior_old;
    d_u_interior_old = d_u_interior;
    d_u_interior = temp;

    temp = d_u_boundary_old;
    d_u_boundary_old = d_u_boundary;
    d_u_boundary = temp;

    // red black update loop
    for (int i = 0; i < 2; ++i)
    {
      // This should simply update its portion of interior and return
      update_interior_kernel<<<interior_grid_size, interior_block_size, 0, stream_interior>>>(
          d_u_interior, d_u_interior_old, d_u_boundary_old,
          d_rhs, h_squared,
          local_Nx, local_Ny, local_Nz, iter,
          u_back, u_front, u_north, u_south, u_west, u_east,
          d_block_max_diffs);

      GPU_CHECK(cudaMemcpyAsync(block_max_diffs, d_block_max_diffs, num_interior_blocks * sizeof(double), cudaMemcpyDeviceToHost, stream_interior));

      // residual and error
      double residual = 0.0;
      double error = 0.0;

      MPI_Request reqs[12];
      int req_count = 0;

      // X - direction(non - periodic) : back / front
      if (back != MPI_PROC_NULL)
      {
        MPI_Isend(&u_boundary[u_back], 1, yz_plane_type, back, 0, my_cart_dim, &reqs[req_count++]);
        MPI_Irecv(&u_ghost[u_back], 1, yz_plane_type, back, 0, my_cart_dim, &reqs[req_count++]);
      }
      if (front != MPI_PROC_NULL)
      {
        MPI_Isend(&u_boundary[u_front], 1, yz_plane_type, front, 0, my_cart_dim, &reqs[req_count++]);
        MPI_Irecv(&u_ghost[u_front], 1, yz_plane_type, front, 0, my_cart_dim, &reqs[req_count++]);
      }

      // Y-direction (periodic): north/south
      MPI_Isend(&u_boundary[u_north], 1, xz_plane_type, north, 0, my_cart_dim, &reqs[req_count++]);
      MPI_Irecv(&u_ghost[u_north], 1, xz_plane_type, north, 0, my_cart_dim, &reqs[req_count++]);

      MPI_Isend(&u_boundary[u_south], 1, xz_plane_type, south, 0, my_cart_dim, &reqs[req_count++]);
      MPI_Irecv(&u_ghost[u_south], 1, xz_plane_type, south, 0, my_cart_dim, &reqs[req_count++]);

      // Z-direction (periodic): front/back
      MPI_Isend(&u_boundary[u_west], 1, xy_plane_type, west, 0, my_cart_dim, &reqs[req_count++]);
      MPI_Irecv(&u_ghost[u_west], 1, xy_plane_type, west, 0, my_cart_dim, &reqs[req_count++]);

      MPI_Isend(&u_boundary[u_east], 1, xy_plane_type, east, 0, my_cart_dim, &reqs[req_count++]);
      MPI_Irecv(&u_ghost[u_east], 1, xy_plane_type, east, 0, my_cart_dim, &reqs[req_count++]);

      // Wait for communication to complete
      MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

      // Update solution
      // skip first and last point because of dirchlet bounds for x
      bool ignore_x_lower = coords[0] == 0;
      bool ignore_x_upper = coords[0] == dims[0] - 1;

      GPU_CHECK(cudaMemcpy(d_u_ghost, u_ghost, boundary_size * sizeof(double), cudaMemcpyHostToDevice));

      if (!ignore_x_lower)
      {
        update_back_boundary_kernel<<<yz_grid_size, boundary_block_size, 0, stream_back>>>(
            d_u_boundary, d_u_boundary_old,
            d_u_interior_old, d_u_ghost,
            d_rhs, h_squared,
            local_Nx, local_Ny, local_Nz, iter,
            u_west, u_east, u_back, u_front, u_south, u_north,
            d_block_max_diffs_boundary, num_back_idx);
      }

      if (!ignore_x_upper)
      {
        update_front_boundary_kernel<<<yz_grid_size, boundary_block_size, 0, stream_front>>>(
            d_u_boundary, d_u_boundary_old,
            d_u_interior_old, d_u_ghost,
            d_rhs, h_squared,
            local_Nx, local_Ny, local_Nz, iter,
            u_west, u_east, u_back, u_front, u_south, u_north,
            d_block_max_diffs_boundary, num_front_idx);
      }

      update_west_boundary_kernel<<<xy_grid_size, boundary_block_size, 0, stream_west>>>(
          d_u_boundary, d_u_boundary_old,
          d_u_interior_old, d_u_ghost,
          d_rhs, h_squared,
          local_Nx, local_Ny, local_Nz, iter,
          u_west, u_east, u_back, u_front, u_south, u_north,
          ignore_x_lower, ignore_x_upper,
          d_block_max_diffs_boundary, num_west_idx);

      update_east_boundary_kernel<<<xy_grid_size, boundary_block_size, 0, stream_east>>>(
          d_u_boundary, d_u_boundary_old,
          d_u_interior_old, d_u_ghost,
          d_rhs, h_squared,
          local_Nx, local_Ny, local_Nz, iter,
          u_west, u_east, u_back, u_front, u_south, u_north,
          ignore_x_lower, ignore_x_upper,
          d_block_max_diffs_boundary, num_east_idx);

      update_north_boundary_kernel<<<xz_grid_size, boundary_block_size, 0, stream_north>>>(
          d_u_boundary, d_u_boundary_old,
          d_u_interior_old, d_u_ghost,
          d_rhs, h_squared,
          local_Nx, local_Ny, local_Nz, iter,
          u_west, u_east, u_back, u_front, u_south, u_north,
          ignore_x_lower, ignore_x_upper,
          d_block_max_diffs_boundary, num_north_idx);

      update_south_boundary_kernel<<<xz_grid_size, boundary_block_size, 0, stream_south>>>(
          d_u_boundary, d_u_boundary_old,
          d_u_interior_old, d_u_ghost,
          d_rhs, h_squared,
          local_Nx, local_Ny, local_Nz, iter,
          u_west, u_east, u_back, u_front, u_south, u_north,
          ignore_x_lower, ignore_x_upper,
          d_block_max_diffs_boundary, num_south_idx);

      cudaDeviceSynchronize();

      // copy data (we only need the new boundary data for the next halo)
      GPU_CHECK(cudaMemcpyAsync(u_boundary, d_u_boundary, boundary_size * sizeof(double), cudaMemcpyDeviceToHost));
      GPU_CHECK(cudaMemcpyAsync(block_max_diffs_boundary, d_block_max_diffs_boundary, num_boundary_blocks * sizeof(double), cudaMemcpyDeviceToHost));

      cudaDeviceSynchronize();

      // // reduce
      for (int i = 0; i < num_interior_blocks; i++)
      {
        if (block_max_diffs[i] > diff)
          diff = block_max_diffs[i];
      }

      // reduce boundary
      int init_i = 0;
      if (ignore_x_lower)
        init_i = num_front_idx;

      if (ignore_x_upper)
      {
        for (int i = init_i; i < num_front_idx; i++)
        {
          if (block_max_diffs_boundary[i] > diff)
            diff = block_max_diffs_boundary[i];
        }

        for (int i = num_west_idx; i < num_boundary_blocks; i++)
        {
          if (block_max_diffs_boundary[i] > diff)
            diff = block_max_diffs_boundary[i];
        }
      }
      else
      {
        for (int i = init_i; i < num_boundary_blocks; i++)
        {
          if (block_max_diffs_boundary[i] > diff)
            diff = block_max_diffs_boundary[i];
        }
      }

      // Compute global maximum difference
      double global_diff;
      MPI_Allreduce(&diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, my_cart_dim);
      diff = global_diff;

      // print iter to sdout
      ++iter;
    }
  }

  double end_time = MPI_Wtime();
  double computation_time = end_time - start_time;

  // copy interior back
  cudaMemcpy(u_interior, d_u_interior, interior_size * sizeof(double), cudaMemcpyDeviceToHost);
  // aggregate interior and boundary
  for (int i = 0; i < local_Nx; i++)
  {
    for (int j = 0; j < local_Ny; j++)
    {
      for (int k = 0; k < local_Nz; k++)
      {
        int idx_full = i * (local_Ny * local_Nz) + j * local_Nz + k;

        if (i == 0)
        {
          // Back boundary (YZ plane)
          u[idx_full] = u_boundary[u_back + j * local_Nz + k];
        }
        else if (i == local_Nx - 1)
        {
          // Front boundary (YZ plane)
          u[idx_full] = u_boundary[u_front + j * local_Nz + k];
        }
        else if (j == 0)
        {
          // North boundary (XZ plane)
          u[idx_full] = u_boundary[u_north + i * local_Nz + k];
        }
        else if (j == local_Ny - 1)
        {
          // South boundary (XZ plane)
          u[idx_full] = u_boundary[u_south + i * local_Nz + k];
        }
        else if (k == 0)
        {
          // West boundary (XY plane)
          u[idx_full] = u_boundary[u_west + j * local_Nx + i];
        }
        else if (k == local_Nz - 1)
        {
          // East boundary (XY plane)
          u[idx_full] = u_boundary[u_east + j * local_Nx + i];
        }
        else
        {
          // Interior
          int idx_interior = (i - 1) * ((local_Ny - 2) * (local_Nz - 2)) + (j - 1) * (local_Nz - 2) + (k - 1);
          u[idx_full] = u_interior[idx_interior];
        }
      }
    }
  }

  // Compute global error norm
  double local_sum = 0.0;
  for (int i = 0; i < local_Nx; ++i)
  {
    for (int j = 0; j < local_Ny; ++j)
    {
      for (int k = 0; k < local_Nz; ++k)
      {
        double d = u[idx(i, j, k)] - exact[idx(i, j, k)];
        local_sum += d * d;
      }
    }
  }

  double global_sum;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  double global_time;
  MPI_Allreduce(&computation_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  // Sum up total FLOPs and bytes across all processes
  long total_flops = long(iter) * long(num_internal_points) * long(flops_per_point);
  long total_bytes = long(iter) * long(num_internal_points) * long(bytes_per_point);

  if (my_rank == 0)
  {
    double loss = std::sqrt(global_sum / num_internal_points);
    std::cout << "Computation took " << global_time << " seconds.\n";
    std::cout << "Converged in " << iter << " iterations.\n";
    std::cout << "L2 norm of the error: " << loss << "\n";

    // Compute operational intensity and achieved performance
    double operational_intensity = static_cast<double>(total_flops) / total_bytes;
    double achieved_performance = total_flops / global_time;

    // Write residual and error data to CSV
    std::ofstream csv_file_residuals;
    csv_file_residuals.open("residual_error.csv");
    if (csv_file_residuals.is_open())
    {
      csv_file_residuals << "Iteration,Avg Residual,Avg Error\n"; // CSV header
      for (size_t i = 0; i < avg_residual_per_iter.size(); ++i)
      {
        csv_file_residuals << (i + 1) << ","
                           << avg_residual_per_iter[i] << ","
                           << avg_error_per_iter[i] << "\n";
      }
      csv_file_residuals.close();
    }

    // Write performance data to CSV
    std::ofstream csv_file;
    csv_file.open("performance_data.csv", std::ios::app);
    if (csv_file.is_open())
    {
      csv_file << "MPI 3D np=" << size << " -O2" << ","
               << operational_intensity << ","
               << achieved_performance / 1e12 << "," // Convert to TFLOPs/s
               << global_time << ","
               << global_time / iter << ","
               << loss << ","
               << iter << ","
               << total_flops << ","
               << total_bytes << ","
               << "darkslategrey" << "\n";
      csv_file.close();
    }
  }

  // Free datatypes and allocated memory
  MPI_Type_free(&yz_plane_type);
  MPI_Type_free(&xz_plane_type);
  MPI_Type_free(&xy_plane_type);

  delete[] u;
  delete[] rhs;
  delete[] exact;
  delete[] u_interior;
  delete[] u_boundary;
  delete[] u_ghost;
  delete[] block_max_diffs;
  delete[] block_max_diffs_boundary;

  cudaFree(d_u_interior);
  cudaFree(d_u_interior_old);
  cudaFree(d_u_boundary);
  cudaFree(d_u_boundary_old);
  cudaFree(d_u_ghost);
  cudaFree(d_rhs);
  cudaFree(d_block_max_diffs);
  cudaFree(d_block_max_diffs_boundary);

  cudaStreamDestroy(stream_interior);
  cudaStreamDestroy(stream_back);
  cudaStreamDestroy(stream_front);
  cudaStreamDestroy(stream_east);
  cudaStreamDestroy(stream_west);
  cudaStreamDestroy(stream_north);
  cudaStreamDestroy(stream_south);

  MPI_Finalize();
  return 0;
}
