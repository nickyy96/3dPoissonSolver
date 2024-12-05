#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <limits>
#include <stdio.h>
#include <stdlib.h>

const double pi = 3.14159;

int periodic_index(int index, int N) {
    if (index == -1) {
        // if index is -1 wrap arround and take last index 
        return N - 2;
    } else if (index == N) {
        // if index is last, wrap arround and take first index 
        return 1; 
    } else {
        // do nothinng 
        return index; // Return the index if within bounds
    }
}

int main() {
    // Grid and parameters
    printf("Running Poisson solver with no optimizations.\n");
    // Grid size - remember its cubed because 3d 
    const int N = 100;   
    // Grid spacing                   
    const double h = 1.0 / (N - 1); 
     // Maximum iterations        
    const int max_iter = 10000;            
    // Tolerance for convergence
    const double tol = 1e-6;               
     // Parameters for the exact solution
    const int n = 1, m = 1, l = 1;         

    // Initialize 3D arrays
    // Solution
    std::vector<std::vector<std::vector<double>>> u(N, std::vector<std::vector<double>>(N, std::vector<double>(N, 0.0))); 
    // Previous solution
    std::vector<std::vector<std::vector<double>>> u_old = u;    
    // RHS                                                        
    std::vector<std::vector<std::vector<double>>> rhs(N, std::vector<std::vector<double>>(N, std::vector<double>(N, 0.0)));
    // Exact solution
    std::vector<std::vector<std::vector<double>>> exact = u;                                                           

    // time initialization
    auto start_init = std::chrono::high_resolution_clock::now();

       

    // Initialization stage
    int init_flops = 0;
    int init_bytes = 0;


    // Compute rhs and exact solution
    for (int i = 0; i < N; ++i) {
        double x = i * h;
        for (int j = 0; j < N; ++j) {
            double y = j * h;
            for (int k = 0; k < N; ++k) {
                double z = k * h;
                exact[i][j][k] = sin(n * pi * x) * cos(m * pi * y) * sin(k * pi * z);
                rhs[i][j][k] = -(n * n + m * m + l * l) * pi * pi * exact[i][j][k];
                      
                init_flops += 8; 
                init_bytes += 16; 
            }
        }
    }

    // Apply Dirichlet boundary conditions at 0 and N-1
    for (int j = 0; j < N; ++j) {
        double y = j * h;
        for (int k = 0; k < N; ++k) {
            double z = k * h;
            u[N - 1][j][k] = exact[N - 1][j][k]; 
            init_flops += 1;
            init_bytes += 16;
        }
    }



    // end intilization timing 
    auto end_init = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> init_time = end_init - start_init;

    // Compute operational intensity and performance for initialization
    double operational_intensity_init = static_cast<double>(init_flops) / init_bytes;
    double achieved_performance_init = init_flops / init_time.count(); 

    // memory ussage for roofline 
    // because skip first and last dim in X
    int num_points_internal = N * N * (N-2);
    // 5 adds, 1 subtract, 4 divides for iterations, 8 flops from residual and error updates 
    int flops_per_point = 18;  
     // 8 reads, 1 write for iterations, 9 for residual and error updates
    int bytes_per_point = 18 * sizeof(double);
    // total flops and bytes
    double total_flops_solver = 0.0;
    double total_bytes_solver = 0.0;
    


    // solving stage 
    int iter = 0;
    // average update between iterations to know when to stop
    double avg_update = std::numeric_limits<double>::infinity();
    // timer 
    auto start_solver = std::chrono::high_resolution_clock::now();

    double h_squared_div_6 = 6.0 / (h * h);
    double h_squared = h * h;


    // Residuals and errors
    std::vector<double> avg_residual_per_iter;
    std::vector<double> avg_error_per_iter;
    
    
    while (avg_update > tol && iter < max_iter) {
        avg_update = 0.0;
        
        // Copy current solution to previous
        u_old = u; 

        // residual and error
        double residual = 0.0;
        double error = 0.0;

        // Update interior points
        // skip first and last point because of dirchlet bounds 
        for (int i = 1; i < N - 1; ++i) {
            // go through all j and k points because periodic in x and y 
            for (int j = 0; j < N; ++j) { 
                for (int k = 0; k < N; ++k) { 
                    // call weap around function because periodic 
                    int j_prev = periodic_index(j - 1, N);
                    int j_next = periodic_index(j + 1, N); 
                    int k_prev = periodic_index(k - 1, N); 
                    int k_next = periodic_index(k + 1, N); 
                    
                    // update 
                    double new_value = (
                        (u_old[i - 1][j][k] + u_old[i + 1][j][k])/(h_squared) +
                        (u_old[i][j_prev][k] + u_old[i][j_next][k])/(h_squared) +
                        (u_old[i][j][k_prev] + u_old[i][j][k_next])/(h_squared) -
                        rhs[i][j][k]) / h_squared_div_6;
                    
                    // sum average update 
                    avg_update += std::abs(new_value - u[i][j][k]);

                    // update the solution
                    u[i][j][k] = new_value; 

                    residual += (u[i][j][k] - exact[i][j][k]); 
                    error += (u[i][j][k] - exact[i][j][k]) * (u[i][j][k] - exact[i][j][k]);
                }
            }
        }
        // Normalize by the number of points
        avg_update /= num_points_internal; 
        avg_residual_per_iter.push_back(residual / num_points_internal);
        avg_error_per_iter.push_back(sqrt(error / num_points_internal));

        // update performance 
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
    // Output results
    printf("Initialization time: %.2f seconds.\n", init_time.count());
    printf("Initialization FLOPs: %d, Bytes: %d\n", init_flops, init_bytes);
    printf("Operational Intensity (Initialization): %.2f\n", operational_intensity_init);
    printf("Achieved Performance (Initialization): %.2f (FLOPs)\n", achieved_performance_init);
    printf("\n");
    printf("Solver converged in %d iterations.\n", iter);
    printf("Final average update: %.6f\n", avg_update);
    printf("Final RMSE %.6f\n", last_rmse);
    printf("Average iteration time: %.2f seconds.\n", avg_iteration_time);
    printf("Total FLOPs for solver + error checks: %.2f\n", total_flops_solver);
    printf("Total Bytes for solver + error checks: %.2f\n", total_bytes_solver);
    printf("Operational Intensity for solver + error check: %.2f\n", operational_intensity_solver);
    printf("Achieved Performance for solver + error check: %.2f (FLOPs) \n ", achieved_performance_solver);
    printf("Total solver time: %.2f seconds.\n", solver_time.count());
    printf("\n");
    printf("Total time (including initialization): %.2f seconds.\n", (init_time + solver_time).count());

    printf("Iteration   Avg Residual       Avg Error\n");
    for (size_t i = 0; i < avg_residual_per_iter.size(); ++i) {
        printf("%10lu   %14.6e   %14.6e\n", i + 1, avg_residual_per_iter[i], avg_error_per_iter[i]);
    }




    return 0;
}
