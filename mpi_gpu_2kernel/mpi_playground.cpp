#include <iostream>
#include <vector>

void print_subarray(double *array, int sizes[3], int subsizes[3], int starts[3])
{
  int x_size = sizes[0], y_size = sizes[1], z_size = sizes[2];
  int x_subsize = subsizes[0], y_subsize = subsizes[1], z_subsize = subsizes[2];
  int x_start = starts[0], y_start = starts[1], z_start = starts[2];

  std::cout << "Extracted subarray:" << std::endl;

  for (int i = 0; i < x_subsize; ++i)
  { // Iterate over the x dimension
    for (int j = 0; j < y_subsize; ++j)
    { // Iterate over the y dimension
      for (int k = 0; k < z_subsize; ++k)
      { // Iterate over the z dimension
        // Calculate the 1D index of the element in the original array
        int global_idx = (x_start + i) * (y_size * z_size) +
                         (y_start + j) * z_size +
                         (z_start + k);

        std::cout << array[global_idx] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl; // Separate planes
  }
}

int main()
{
  // Original 3D array dimensions
  int sizes[3] = {5, 5, 5}; // Example: 5x5x5 array

  // Subarray dimensions
  int subsizes[3] = {3, 1, 3}; // Extract a yz-plane of size 1x3x3

  // Starting indices (offset into the original array)
  int starts[3] = {1, 1, 1}; // Start at (1, 1, 1) in the original array

  // Simulate a flattened 3D array
  std::vector<double> array(5 * 5 * 5);
  for (int i = 0; i < 5; ++i)
  {
    for (int j = 0; j < 5; ++j)
    {
      for (int k = 0; k < 5; ++k)
      {
        array[i * (5 * 5) + j * 5 + k] = i * 100 + j * 10 + k;
      }
    }
  }

  // print the array
  for (int i = 0; i < 5; ++i)
  {
    for (int j = 0; j < 5; ++j)
    {
      for (int k = 0; k < 5; ++k)
      {
        std::cout << array[i * (5 * 5) + j * 5 + k] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  // Print the subarray
  print_subarray(array.data(), sizes, subsizes, starts);

  return 0;
}
