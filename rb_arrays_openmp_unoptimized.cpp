#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>


const double pi = acos(-1.0);

int periodic_index(int index, int N) {
    if (index == -1) {
        // if index is -1 wrap arround and take last index 
        return N - 1;
    } else if (index == N) {
        // if index is last, wrap arround and take first index 
        return 0; 
    } else {
        // do nothinng 
        return index; // Return the index if within bounds
    }
}

int main() {
    // Grid and parameters
    printf("Running Poisson solver with red black.\n");

    // Grid size - remember its cubed because 3d 
    const int N = 2;   
    // Grid spacing                   
    const double h = 1.0 / (N - 1); 
     // Maximum iterations        
    const int max_iter = 10000;            
    // Tolerance for convergence
    const double tol = 1e-6;               
     // Parameters for the exact solution
    const int n = 2, m = 2, l = 2;         

    // Initialize 3D arrays
    // Solution
    double*** u = new double**[N];
    double*** u_old = new double**[N];
    double*** rhs = new double**[N];
    double*** exact = new double**[N];
    

    for (int i = 0; i < N; ++i) {
        u[i] = new double*[N];
        u_old[i] = new double*[N];
        rhs[i] = new double*[N];
        exact[i] = new double*[N];
        for (int j = 0; j < N; ++j) {
            u[i][j] = new double[N]();
            u_old[i][j] = new double[N]();
            rhs[i][j] = new double[N]();
            exact[i][j] = new double[N]();
        }
    }
    // time initialization
    auto start_init = std::chrono::high_resolution_clock::now();
           

    // Initialization stage
    int init_flops = 0;
    int init_bytes = 0;


    // Compute rhs and exact solution
    #pragma omp parallel for collapse(3) reduction(+:init_flops, init_bytes)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                double x = i * h;
                double y = j * h;
                double z = k * h;
                exact[i][j][k] = sin(n * pi * x) * cos(m * pi * y) * sin(l * pi * z);
                rhs[i][j][k] = -(n * n + m * m + l * l) * pi * pi * exact[i][j][k];

                init_flops += 10; 
                init_bytes += 3 * sizeof(double); 
            }
        }
    }

    // Apply Dirichlet boundary conditions at 0 and N-1
    #pragma omp parallel for reduction(+:init_bytes) collapse(2)
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < N; ++k) {
            double y = j * h;
            double z = k * h;
            u[0][j][k] = exact[0][j][k];
            u[N - 1][j][k] = exact[N - 1][j][k]; 
            u_old[N - 1][j][k] = exact[N - 1][j][k]; 
            init_bytes += 2 * sizeof(double); 
        }
    }


    std::cout << "u array before iteration:\n";
for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < N; ++k) {
            std::cout << "u[" << i << "][" << j << "][" << k << "] = " << u[i][j][k] << "\n";
        }
    }
}
std::cout << std::endl;



    // end intilization timing 
    auto end_init = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> init_time = end_init - start_init;

    // Compute operational intensity and performance for initialization
    double operational_intensity_init = static_cast<double>(init_flops) / init_bytes;
    double achieved_performance_init = init_flops / init_time.count(); 

    // memory ussage for roofline 
    // because skip first and last dim in X
    int num_points_internal = N * N * (N-2);

    int flops_per_point = 7;  
    int bytes_per_point = 6 * sizeof(double);
    // total flops and bytes
    double total_flops_solver = 0.0;
    double total_bytes_solver = 0.0;
    


    // solving stage 
    int iter = 0;
    // average update between iterations to know when to stop
    double max_update = std::numeric_limits<double>::infinity();
    // timer 
    auto start_solver = std::chrono::high_resolution_clock::now();

    double h_squared_div_6 = 6.0 / (h * h);
    double h_squared = h * h;


    // Residuals and errors
    std::vector<double> avg_residual_per_iter;
    std::vector<double> avg_error_per_iter;
    
    
    while (max_update > tol && iter < max_iter) {
    max_update = 0.0;

    // Swap pointers for current and previous solutions
    double*** temp = u_old;
    u_old = u;
    u = temp;
        
    double residual = 0.0;
    double error = 0.0;

    // Update interior points: first red points, then black points
    #pragma omp parallel for reduction(+:residual, error) reduction(max:max_update) collapse(3)
    for (int color = 0; color < 2; ++color) { // 0 = red, 1 = black
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 0; j < N; ++j) { 
                for (int k = 0; k < N; ++k) { 
                    if (((i + j + k) % 2) == (color)) { // Red-Black check
                        // Compute periodic indices
                        int j_prev = periodic_index(j - 1, N);
                        int j_next = periodic_index(j + 1, N); 
                        int k_prev = periodic_index(k - 1, N); 
                        int k_next = periodic_index(k + 1, N); 
                        
                        // Update solution
                        double new_value = (
                            (u_old[i - 1][j][k] + u_old[i + 1][j][k]) +
                            (u_old[i][j_prev][k] + u_old[i][j_next][k]) +
                            (u_old[i][j][k_prev] + u_old[i][j][k_next]) -
                             (h_squared * rhs[i][j][k])) / 6.0;
                        
                        // Compute update magnitude
                        double update = std::abs(new_value - u[i][j][k]);
                        if (update > max_update) {
                            max_update = update;
                        }

                        // Update the solution
                        u[i][j][k] = new_value;

                        // Calculate residual and error
                        double res = u[i][j][k] - exact[i][j][k];
                        residual += res; 
                        error += res * res;
                    }
                }
            }
        }
    }

    // Normalize by the number of internal points
    avg_residual_per_iter.push_back(residual / num_points_internal);
    avg_error_per_iter.push_back(sqrt(error / num_points_internal));

    // Update performance counters
    total_flops_solver += num_points_internal * flops_per_point;
    total_bytes_solver += num_points_internal * bytes_per_point;
    ++iter;
}


    // end timer 
    auto end_solver = std::chrono::high_resolution_clock::now();
    // take time on solver 
    std::chrono::duration<double> solver_time = end_solver - start_solver;
    // get average time per iteration
    double avg_iteration_time = solver_time.count() / iter;

    // compute performance for solver 
    // FLOPs per byte
    double operational_intensity_solver = total_flops_solver / total_bytes_solver;    
     // FLOPs per second
    double achieved_performance_solver = total_flops_solver / solver_time.count();


    double last_rmse = avg_error_per_iter.back();

        double independent_rmse = 0.0;
    double total_error = 0.0;

    #pragma omp parallel for reduction(+:total_error) collapse(3)
    for (int i = 1; i < N - 1; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                double error = u[i][j][k] - exact[i][j][k];
                total_error += error * error;
            }
        }
    }

    // Normalize by the number of internal points and compute RMSE
    independent_rmse = sqrt(total_error / num_points_internal);
    printf("Independent RMSE computed separately: %.6f\n", independent_rmse);

  




    // Output results
    printf("Initialization time: %.2f seconds.\n", init_time.count());
    printf("Initialization FLOPs: %d, Bytes: %d\n", init_flops, init_bytes);
    printf("Operational Intensity (Initialization): %.2f\n", operational_intensity_init);
    printf("Achieved Performance (Initialization): %.2f (FLOPs)\n", achieved_performance_init);
    printf("\n");
    printf("Solver converged in %d iterations.\n", iter);
    printf("Final average update: %.6f\n", max_update);
    printf("Final RMSE %.6f\n", last_rmse);
   // printf("Total RMSE %.6f\n", total);
    printf("Average iteration time: %.2f seconds.\n", avg_iteration_time);
    printf("Total FLOPs for solver + error checks: %.2f\n", total_flops_solver);
    printf("Total Bytes for solver + error checks: %.2f\n", total_bytes_solver);
    printf("Operational Intensity for solver + error check: %.2f\n", operational_intensity_solver);
    printf("Achieved Performance for solver + error check: %.2f (TFLOPs) \n ", achieved_performance_solver);
    printf("Total solver time: %.2f seconds.\n", solver_time.count());
    printf("\n");
    printf("Total time (including initialization): %.2f seconds.\n", (init_time + solver_time).count());

    // Write residuals and errors to a CSV file
std::ofstream csv_file("residual_error_data_rb.csv");
if (csv_file.is_open()) {
    csv_file << "Iteration,Avg Residual,Avg Error\n"; // CSV header
    for (size_t i = 0; i < avg_residual_per_iter.size(); ++i) {
        csv_file << (i + 1) << "," 
                 << avg_residual_per_iter[i] << "," 
                 << avg_error_per_iter[i] << "\n";
    }
    csv_file.close();
    printf("Residual and error data written to residual_error_data.csv.\n");
} else {
    printf("Error: Unable to open file for writing.\n");
}
// free memory 
for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
        delete[] u[i][j];
        delete[] u_old[i][j];
        delete[] rhs[i][j];
        delete[] exact[i][j];
    }
    delete[] u[i];
   delete[] u_old[i];
    delete[] rhs[i];
    delete[] exact[i];
}
delete[] u;
delete[] u_old;
delete[] rhs;
delete[] exact;


    return 0;
}
