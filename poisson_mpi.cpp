#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <mpi.h>
#include <fstream>

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int my_rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // 3d constants
  const int N = 1000;
  const double start = 0.0;
  const double end = 1.0;
  const double h = (end - start) / (N - 1);
  const double tol = 1e-6;
  const int max_iter = 10000;
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

  // time initialization
  auto start_init = std::chrono::high_resolution_clock::now();

  // Initialization stage
  int init_flops = 0;
  int init_bytes = 0;

  int total_size = local_Nx_with_ghosts * local_Ny_with_ghosts * local_Nz_with_ghosts;
  double *u = new double[total_size];
  double *u_old = new double[total_size];
  double *rhs = new double[total_size];
  double *exact = new double[total_size];

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
        double val = sin(n * pi * x) * cos(m * pi * y) * sin(k * pi * z);
        exact[idx(i, j, k)] = val;

        double lap = (n * n + m * m + l * l) * pi * pi * val;
        rhs[idx(i, j, k)] = lap;
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
        u[idx(1, j, k)] = exact[idx(1, j, k)];
        init_flops += 1;
        init_bytes += 16;
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
        u[idx(local_Nx, j, k)] = exact[idx(local_Nx, j, k)];
        init_flops += 1;
        init_bytes += 16;
      }
    }
  }

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
  // int local_internal_points = (local_Nx - 2 + 1) * (local_Ny - 2 + 1);
  // int flops_per_point = 11;                 // 3 multiplies, 3 adds, 4 divide, 1 subtract
  // int bytes_per_point = 6 * sizeof(double); // 5 reads, 1 write

  double local_flops = 0.0;
  double local_bytes = 0.0;

  // Iterative update
  double diff = std::numeric_limits<double>::infinity();
  int iter = 0;

  double start_time = MPI_Wtime();

  while (diff > tol && iter < max_iter)
  {
    std::cout << "Diff: " << diff << std::endl;
    diff = 0.0;
    for (int i = 0; i < total_size; i++)
    {
      u_old[i] = u[i];
    }

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

    // Update solution
    for (int i = 1; i <= local_Nx; ++i)
    {
      for (int j = 1; j <= local_Ny; ++j)
      {
        for (int k = 1; k <= local_Nz; ++k)
        {
          double x_update = u_old[idx(i - 1, j, k)] + u_old[idx(i + 1, j, k)] / (h * h);
          double y_update = u_old[idx(i, j - 1, k)] + u_old[idx(i, j + 1, k)] / (h * h);
          double z_update = u_old[idx(i, j, k - 1)] + u_old[idx(i, j, k + 1)] / (h * h);

          double val = (x_update + y_update + z_update - rhs[idx(i, j, k)]) / (6.0 / (h * h));

          double d = std::fabs(val - u_old[idx(i, j, k)]);
          if (d > diff)
            diff = d;
          u[idx(i, j, k)] = val;
        }
      }
    }

    // // Update performance counters
    // int total_points = local_Nx * local_Ny; // Includes interior + boundary
    // local_flops += total_points * flops_per_point;
    // local_bytes += total_points * bytes_per_point;

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
  double total_flops = 0.0;
  double total_bytes = 0.0;
  MPI_Allreduce(&local_flops, &total_flops, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_bytes, &total_bytes, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if (my_rank == 0)
  {
    double loss = std::sqrt(global_sum / (N * N * N));
    std::cout << "Computation took " << global_time << " seconds.\n";
    std::cout << "Converged in " << iter << " iterations.\n";
    std::cout << "L2 norm of the error: " << loss << "\n";

    // Compute operational intensity and achieved performance
    double operational_intensity = total_flops / total_bytes;     // FLOPs per byte
    double achieved_performance = total_flops / computation_time; // FLOPs per second

    // Write performance data to CSV
    // std::ofstream csv_file;
    // csv_file.open("performance_data.csv", std::ios::app);
    // csv_file << "MPI 2D np=" << size << " -O3 ffast-math ftree-vectorize march=native"
    //          << "," << operational_intensity << ","
    //          << achieved_performance / 1e9 << "," // Convert to GFLOPs/s
    //          << computation_time << ","
    //          << iter << ","
    //          << total_flops << ","
    //          << total_bytes << ","
    //          << "darkslategrey" << "\n";
    // csv_file.close();
  }

  // Free datatypes and allocated memory
  MPI_Type_free(&yz_plane_type);
  MPI_Type_free(&xz_plane_type);
  MPI_Type_free(&xy_plane_type);

  delete[] u;
  delete[] u_old;
  delete[] rhs;
  delete[] exact;

  MPI_Finalize();
  return 0;
}
