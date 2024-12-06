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

// Kernel to perform one iteration on boundary
__global__ void update_boundary_kernel(
    double *u_new, const double *u_old, const double *rhs,
    double h_sq, int lower_bound_inc, int upper_bound_dec,
    int n_global_x, int n_global_y, int n_global_z, int iter,
    double *d_block_max_diffs)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  int num_boundaries = ((i == 1 || i == n_global_x - 2) + (j == 1 || j == n_global_y - 2) + (k == 1 || k == n_global_z - 2);
  if (num_boundaries == 0 || lower_bound_inc || )
    return;

  double local_diff = 0.0;
  if (in_boundary)
  {
    int idx = i * (n_global_y * n_global_z) + j * n_global_z + k;

    if ((i + j + k) % 2 == (iter % 2))
    {
      double u_left = u_old[idx - 1];
      double u_right = u_old[idx + 1];
      double u_down = u_old[idx - n_global_z];
      double u_up = u_old[idx + n_global_z];
      double u_back = u_old[idx - (n_global_x * n_global_y)];
      double u_front = u_old[idx + (n_global_x * n_global_y)];
      double rhs_val = rhs[idx];

      double val = ((u_left + u_right) + (u_down + u_up) + (u_back + u_front) - rhs_val * h_sq) / 6.0;
      local_diff = fabs(val - u_old[idx]);
      u_new[idx] = val;
    }
    else
    {
      u_new[idx] = u_old[idx];
    }
  }

  // Reduction for max diff
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
    int block_idx = blockIdx.z * (gridDim.y * gridDim.x) + blockIdx.y * gridDim.x + blockIdx.x;
    d_block_max_diffs[block_idx] = sdata[0];
  }
}

// Kernel to perform one iteration on interior
__global__ void update_interior_kernel(
    double *u_new, const double *u_old, const double *rhs,
    double h_sq,
    int n_global_x, int n_global_y, int n_global_z, int iter,
    double *d_block_max_diffs)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  double local_diff = 0.0;

  if (i > 1 && i < n_global_x - 2 &&
      j > 1 && j < n_global_y - 2 &&
      k > 1 && k < n_global_z - 2)
  {
    // index in the 1D array
    int idx = i * (n_global_y * n_global_z) + j * n_global_z + k;

    // red black
    if ((i + j + k) % 2 != (iter % 2))
    {
      u_new[idx] = u_old[idx];
    }
    else
    {
      // indices of 6 neighbors
      int idx_left = idx - 1;  // (i, j, k-1)
      int idx_right = idx + 1; // (i, j, k+1)

      int idx_down = idx - n_global_x; // (i, j-1, k)
      int idx_up = idx + n_global_x;   // (i, j+1, k)

      int idx_back = idx - (n_global_x * n_global_y);  // (i-1, j, k)
      int idx_front = idx + (n_global_x * n_global_y); // (i+1, j, k)

      double u_left = u_old[idx_left];
      double u_right = u_old[idx_right];

      double u_down = u_old[idx_down];
      double u_up = u_old[idx_up];

      double u_back = u_old[idx_back];
      double u_front = u_old[idx_front];

      double rhs_val = rhs[idx];

      double val = ((u_left + u_right) + (u_down + u_up) + (u_back + u_front) - rhs_val * h_sq) / 6.0;
      local_diff = fabs(val - u_old[idx]);
      u_new[idx] = val;
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

  printf("ndevices = %d\n", ndevices);

  int my_device = my_rank % ndevices;
  hipSetDevice(my_device);

  // 3d constants
  const int N = 1000;
  const double start = 0.0;
  const double end = 1.0;
  const double h = (end - start) / (N - 1);
  const double h_squared = h * h;
  const double tol = 1e-6;
  const int max_iter = 20000;
  const double pi = acos(-1.0);
  const int n = 1, m = 1, l = 1;

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
  MPI_Cart_shift(my_cart_dim, 0, 1, &west, &east);
  MPI_Cart_shift(my_cart_dim, 1, 1, &north, &south);
  MPI_Cart_shift(my_cart_dim, 2, 1, &front, &back);

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
    return i * (local_Ny_with_ghosts * local_Nz_with_ghosts) + j * local_Nz_with_ghosts + k;
  };

  // Initialization stage
  int init_flops = 0;
  int init_bytes = 0;

  int total_size = local_Nx_with_ghosts * local_Ny_with_ghosts * local_Nz_with_ghosts;
  double *u = new double[total_size];
  double *u_old = new double[total_size];
  double *rhs = new double[total_size];
  double *exact = new double[total_size];

  // Start init clock
  double init_start_time = MPI_Wtime();

  // Residuals and errors
  std::vector<double> avg_residual_per_iter;
  std::vector<double> avg_error_per_iter;

  // Initialize arrays
  for (int i = 0; i < total_size; i++)
  {
    u[i] = 0.0;
    u_old[i] = 0.0;
    rhs[i] = 0.0;
    exact[i] = 0.0;
  }

  // Compute rhs and exact solution
  for (int i = 1; i <= local_Nx; ++i)
  {
    int global_i = x_start_idx + i - 1;
    double x = start + global_i * h;
    for (int j = 1; j <= local_Ny; ++j)
    {
      int global_j = y_start_idx + j - 1;
      double y = start + global_j * h;
      for (int k = 1; k <= local_Nz; ++k)
      {
        int global_k = z_start_idx + k - 1;

        double z = start + global_k * h;
        double val = sin(n * pi * x) * cos(m * pi * y) * sin(l * pi * z);
        exact[idx(i, j, k)] = val;

        double lap = -(n * n + m * m + l * l) * pi * pi * val;
        rhs[idx(i, j, k)] = lap;

        init_flops += 8;
        init_bytes += 2;
      }
    }
  }

  // Apply Dirichlet boundary conditions at x = 0 and x = N-1
  if (coords[0] == 0)
  {
    // This rank holds the x=0 boundary locally at i=1
    for (int j = 1; j <= local_Ny; ++j)
    {
      for (int k = 1; k <= local_Nz; ++k)
      {
        u[idx(0, j, k)] = 0;
        init_bytes += 1;
      }
    }
  }

  if (coords[0] == dims[0] - 1)
  {
    // This rank holds the x=N-1 boundary locally at i=local_Nx
    for (int j = 1; j <= local_Ny; ++j)
    {
      for (int k = 1; k <= local_Nz; ++k)
      {
        u[idx(local_Nx + 1, j, k)] = 1;
        init_bytes += 1;
      }
    }
  }

  // device arrays
  double *d_u, *d_u_old, *d_rhs, *d_block_max_diffs;
  // allocate and copy
  GPU_CHECK(cudaMalloc(&d_u, total_size * sizeof(double)));
  GPU_CHECK(cudaMalloc(&d_u_old, total_size * sizeof(double)));
  GPU_CHECK(cudaMalloc(&d_rhs, total_size * sizeof(double)));

  GPU_CHECK(cudaMemcpy(d_u, u, total_size * sizeof(double), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_u_old, u_old, total_size * sizeof(double), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_rhs, rhs, total_size * sizeof(double), cudaMemcpyHostToDevice));

  // End init time
  double init_end_time = MPI_Wtime();

  // Set up kernel launch parameters
  dim3 block_size(10, 10, 10);
  dim3 grid_size((local_Nx + block_size.x - 1) / block_size.x,
                 (local_Ny + block_size.y - 1) / block_size.y,
                 (local_Nz + block_size.z - 1) / block_size.z);
  int num_blocks = grid_size.x * grid_size.y * grid_size.z;
  GPU_CHECK(cudaMalloc(&d_block_max_diffs, num_blocks * sizeof(double)));

  // MPI datatypes for face exchanges
  MPI_Datatype yz_plane_type;
  MPI_Type_contiguous(local_Ny * local_Nz, MPI_DOUBLE, &yz_plane_type);
  MPI_Type_commit(&yz_plane_type);

  MPI_Datatype xz_plane_type;
  MPI_Type_vector(local_Nx, local_Nz, local_Ny_with_ghosts * local_Nz_with_ghosts, MPI_DOUBLE, &xz_plane_type);
  MPI_Type_commit(&xz_plane_type);

  MPI_Datatype xy_plane_type;
  MPI_Type_vector(local_Nx * local_Ny, 1, local_Nz_with_ghosts, MPI_DOUBLE, &xy_plane_type);
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

    for (int i = 0; i < total_size; i++)
    {
      u_old[i] = u[i];
    }

    // residual and error
    double residual = 0.0;
    double error = 0.0;

    MPI_Request reqs[12];
    int req_count = 0;

    // X - direction(non - periodic) : west / east
    if (west != MPI_PROC_NULL)
    {
      MPI_Isend(&u_old[idx(1, 1, 1)], 1, yz_plane_type, west, 0, my_cart_dim, &reqs[req_count++]);
      MPI_Irecv(&u_old[idx(0, 1, 1)], 1, yz_plane_type, west, 0, my_cart_dim, &reqs[req_count++]);
    }
    if (east != MPI_PROC_NULL)
    {
      MPI_Isend(&u_old[idx(local_Nx, 1, 1)], 1, yz_plane_type, east, 0, my_cart_dim, &reqs[req_count++]);
      MPI_Irecv(&u_old[idx(local_Nx + 1, 1, 1)], 1, yz_plane_type, east, 0, my_cart_dim, &reqs[req_count++]);
    }

    // Y-direction (periodic): north/south
    MPI_Isend(&u_old[idx(1, 1, 1)], 1, xz_plane_type, north, 0, my_cart_dim, &reqs[req_count++]);
    MPI_Irecv(&u_old[idx(1, 0, 1)], 1, xz_plane_type, north, 0, my_cart_dim, &reqs[req_count++]);

    MPI_Isend(&u_old[idx(1, local_Ny, 1)], 1, xz_plane_type, south, 0, my_cart_dim, &reqs[req_count++]);
    MPI_Irecv(&u_old[idx(1, local_Ny + 1, 1)], 1, xz_plane_type, south, 0, my_cart_dim, &reqs[req_count++]);

    // Z-direction (periodic): front/back
    MPI_Isend(&u_old[idx(1, 1, 1)], 1, xy_plane_type, front, 0, my_cart_dim, &reqs[req_count++]);
    MPI_Irecv(&u_old[idx(1, 1, 0)], 1, xy_plane_type, front, 0, my_cart_dim, &reqs[req_count++]);

    MPI_Isend(&u_old[idx(1, 1, local_Nz)], 1, xy_plane_type, back, 0, my_cart_dim, &reqs[req_count++]);
    MPI_Irecv(&u_old[idx(1, 1, local_Nz + 1)], 1, xy_plane_type, back, 0, my_cart_dim, &reqs[req_count++]);

    // Wait for communication to complete
    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

    GPU_CHECK(cudaMemcpy(d_u_old, u_old, total_size * sizeof(double), cudaMemcpyHostToDevice));

    // Update solution
    // skip first and last point because of dirchlet bounds for x
    int lower_bound_inc = 0;
    if (coords[0] == 0)
      lower_bound_inc = 1;

    int upper_bound_dec = 0;
    if (coords[0] == dims[0] - 1)
      upper_bound_dec = 1;

    update_kernel<<<grid_size, block_size>>>(
        d_u, d_u_old, d_rhs, h_squared, lower_bound_inc, upper_bound_dec,
        local_Nx_with_ghosts, local_Ny_with_ghosts, local_Nz_with_ghosts, iter,
        d_block_max_diffs);

    cudaDeviceSynchronize();

    double *block_max_diffs = new double[num_blocks];
    cudaMemcpy(block_max_diffs, d_block_max_diffs, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);

    // reduce
    for (int i = 0; i < num_blocks; i++)
    {
      if (block_max_diffs[i] > diff)
        // printf("diff: %f\n", block_max_diffs[i]);
        diff = block_max_diffs[i];
    }
    delete[] block_max_diffs;

    // copy data
    cudaMemcpy(u, d_u, total_size * sizeof(double), cudaMemcpyDeviceToHost);

    if (my_rank == 0)
    {
      // Normalize by the number of points
      // avg_residual_per_iter.push_back(residual / num_internal_points);
      // avg_error_per_iter.push_back(std::sqrt(error / num_internal_points));
    }

    // Compute global maximum difference
    double global_diff;
    MPI_Allreduce(&diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, my_cart_dim);
    diff = global_diff;

    ++iter;
  }

  double end_time = MPI_Wtime();
  double computation_time = end_time - start_time;

  // Compute global error norm
  double local_sum = 0.0;
  for (int i = 1; i <= local_Nx; ++i)
  {
    for (int j = 1; j <= local_Ny; ++j)
    {
      for (int k = 1; k <= local_Nz; ++k)
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
  double total_flops = iter * num_internal_points * flops_per_point;
  double total_bytes = iter * num_internal_points * bytes_per_point;

  double global_init_time;
  double end_init_time = init_end_time - init_start_time;

  MPI_Allreduce(&end_init_time, &global_init_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  if (my_rank == 0)
  {
    double loss = std::sqrt(global_sum / num_internal_points);
    std::cout << "Computation took " << global_time << " seconds.\n";
    std::cout << "Converged in " << iter << " iterations.\n";
    std::cout << "L2 norm of the error: " << loss << "\n";

    // Compute operational intensity and achieved performance for initialization
    double operational_intensity_init = init_flops / init_bytes;
    double achieved_performance_init = init_flops / global_init_time;

    // Compute operational intensity and achieved performance
    double operational_intensity = total_flops / total_bytes;
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

    // Write init data to CSV
    std::ofstream csv_file_init;
    csv_file_init.open("init_data.csv", std::ios::app);
    if (csv_file_init.is_open())
    {
      csv_file_init << "MPI 3D np=" << size << " -O2" << ","
                    << operational_intensity_init << ","
                    << achieved_performance_init / 1e12 << "," // Convert to TFLOPs/s
                    << global_init_time << ","
                    << "darkslategrey" << "\n";
      csv_file_init.close();
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
  delete[] u_old;
  delete[] rhs;
  delete[] exact;

  cudaFree(d_u);
  cudaFree(d_u_old);
  cudaFree(d_rhs);
  cudaFree(d_block_max_diffs);

  MPI_Finalize();
  return 0;
}
