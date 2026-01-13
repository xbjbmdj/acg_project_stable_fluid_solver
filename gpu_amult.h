#pragma once

#include <cstddef>
#include <vector>

// Initialize GPU resources and upload matrices.
// J is expected as row-major with stride J_stride (elements per row), using
// modes indexed 0..num_modes-1 (caller can pass the same data where J[1]..J[6]
// live). scaleBase length >= num_modes (only first num_modes used).
bool gpu_amult_init(int n_dim,
    const float* Adiag,
    const float* Adiagplusi,
    const float* Adiagplusj,
    const float* Adiagplusk,
    const float* Adiagminusi,
    const float* Adiagminusj,
    const float* Adiagminusk,
    const float* J,           // pointer to J[0][0]
    std::size_t J_stride,     // elements per row in J (e.g., 22345678)
    const double* scaleBase,  // length >= num_modes
    int num_modes = 6);

// GPU version of A_mult: result = A * x using uploaded matrices.
void gpu_amult(const std::vector<double>& x, std::vector<double>& result);

// Fully GPU-resident Conjugate Gradient solve using the uploaded matrix.
// Returns false on failure or lack of initialization.
bool gpu_cg_solve(const std::vector<double>& d,
                  std::vector<double>& p,
                  double tol = 1e-6,
                  int max_iter = 2000,
                  const std::vector<double>* p0 = nullptr);

// Free GPU resources.
void gpu_amult_destroy();
