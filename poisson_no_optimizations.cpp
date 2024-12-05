#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <limits>
#include <stdio.h>
#include <stdlib.h>

const double pi = acos(-1.0);

int periodic_index(int index, int N) {
    if (index == -1) {
        return N - 2; // Wrap to the last interior index
    } else if (index == N) {
        return 1; // Wrap to the first interior index
    } else {
        return index; // Return the index if within bounds
    }
}

int main() {
    // Grid and parameters
    printf("Running Poisson solver with no optimizations.\n");
    const int N = 100;                      // Grid size
    const double h = 1.0 / (N - 1);         // Grid spacing
    const int max_iter = 10000;             // Maximum iterations
    const double tol = 1e-6;                // Tolerance for convergence
    const int n = 1, m = 1, l = 1;          // Parameters for the exact solution

    // Initialize 3D arrays
    std::vector<std::vector<std::vector<double>>> u(N, std::vector<std::vector<double>>(N, std::vector<double>(N, 0.0))); // Solution
    std::vector<std::vector<std::vector<double>>> u_old = u;                                                             // Previous solution
    std::vector<std::vector<std::vector<double>>> rhs(N, std::vector<std::vector<double>>(N, std::vector<double>(N, 0.0))); // Source term
    std::vector<std::vector<std::vector<double>>> exact = u;                                                             // Exact solution

    // Start timing initialization
    auto start_init = std::chrono::high_resolution_clock::now();

    // Compute rhs and exact solution
    for (int i = 0; i < N; ++i) {
        double x = i * h;
        for (int j = 0; j < N; ++j) {
            double y = j * h;
            for (int k = 0; k < N; ++k) {
                double z = k * h;
                exact[i][j][k] = sin(n * pi * x) * cos(m * pi * y) * sin(k * pi * z);
                rhs[i][j][k] = -(n * n + m * m + l * l) * pi * pi * exact[i][j][k];
            }
        }
    }

    // Apply Dirichlet boundary conditions
    for (int j = 0; j < N; ++j) {
        double y = j * h;
        for (int k = 0; k < N; ++k) {
            double z = k * h;
            u[0][j][k] = 0.0; // Dirichlet at x = 0
            u[N - 1][j][k] = exact[N - 1][j][k]; // Dirichlet at x = 1
        }
    }

    // End timing initialization
    auto end_init = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> init_time = end_init - start_init;

    // Iterative solver
    int iter = 0;
    double avg_update = std::numeric_limits<double>::infinity(); // Max update in the solution
    auto start_solver = std::chrono::high_resolution_clock::now();
    
    while (avg_update > tol && iter < max_iter) {
        avg_update = 0.0;
        u_old = u; // Copy current solution to previous

        // Update interior points
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 0; j < N; ++j) { // Periodic in y
                for (int k = 0; k < N; ++k) { // Periodic in z
                    int j_prev = periodic_index(j - 1, N); // Wrap for y-1
                    int j_next = periodic_index(j + 1, N); // Wrap for y+1
                    int k_prev = periodic_index(k - 1, N); // Wrap for z-1
                    int k_next = periodic_index(k + 1, N); // Wrap for z+1

                    double new_value = (
                        (u_old[i - 1][j][k] + u_old[i + 1][j][k])/(h * h) +
                        (u_old[i][j_prev][k] + u_old[i][j_next][k])/(h * h) +
                        (u_old[i][j][k_prev] + u_old[i][j][k_next])/(h * h) -
                        rhs[i][j][k]) / (6.0 / (h * h));

                    avg_update += std::abs(new_value - u[i][j][k]);
                    u[i][j][k] = new_value; // Update the solution
                }
            }
        }
        avg_update /= (N * N * N); // Normalize by the number of points
        ++iter;
    }

    auto end_solver = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solver_time = end_solver - start_solver;
    double avg_iteration_time = solver_time.count() / iter;

    // Compute RMSE
    auto start_rmse = std::chrono::high_resolution_clock::now();
    double rmse = 0.0;
    for (int i = 1; i < N-1; ++i) {
        for (int j = 1; j < N-1; ++j) {
            for (int k = 1; k < N-1; ++k) {
                double error = u[i][j][k] - exact[i][j][k];
                rmse += error * error;
            }
        }
    }
    rmse = sqrt(rmse / ((N * N * N)-3)); // Normalize and take square root
    auto end_rmse = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> rmse_time = end_rmse - start_rmse;

    // Output results
    printf("Initialization time: %.2f seconds.\n", init_time.count());
    printf("Solver converged in %d iterations.\n", iter);
    printf("Final average update: %.6f\n", avg_update);
    printf("Final RMSE: %.6f\n", rmse);
    printf("Average iteration time: %.2f seconds.\n", avg_iteration_time);
    printf("RMSE computation time: %.2f seconds.\n", rmse_time.count());
    printf("Total solver time: %.2f seconds.\n", solver_time.count());
    printf("Total time (including initialization): %.2f seconds.\n", (init_time + solver_time + rmse_time).count());

    return 0;
}
