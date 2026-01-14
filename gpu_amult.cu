#include "gpu_amult.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <array>
#include <iostream>
#include <memory>
#include <limits>
#include <cmath>

namespace {
struct GpuContext {
    int n_dim = 0;
    int num_modes = 0;
    std::size_t total = 0; // n_dim^3

    // GPU处的缓冲区指针，用于存放向量
    float* d_Adiag = nullptr;
    float* d_Adiagplusi = nullptr;
    float* d_Adiagplusj = nullptr;
    float* d_Adiagplusk = nullptr;
    float* d_Adiagminusi = nullptr;
    float* d_Adiagminusj = nullptr;
    float* d_Adiagminusk = nullptr;

    float* d_J = nullptr;         // column-major (lda = num_modes)
    float* d_scaleBase = nullptr; // length num_modes

    float* d_x = nullptr;
    float* d_result = nullptr;
    float* d_temp = nullptr;   // num_modes
    float* d_scale = nullptr;  // num_modes

    cublasHandle_t handle = nullptr;
};

GpuContext g_ctx;

#define CHECK_CUDA(expr)                                                                 \
    do {                                                                                \
        cudaError_t err__ = (expr);                                                     \
        if (err__ != cudaSuccess) {                                                     \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__) << " at "        \
                      << __FILE__ << ":" << __LINE__ << std::endl;                     \
            return false;                                                               \
        }                                                                               \
    } while (0)

#define CHECK_CUBLAS(expr)                                                              \
    do {                                                                                \
        cublasStatus_t st__ = (expr);                                                   \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                            \
            std::cerr << "cuBLAS error code " << st__ << " at "                      \
                      << __FILE__ << ":" << __LINE__ << std::endl;                     \
            return false;                                                               \
        }                                                                               \
    } while (0)

__global__ void stencil_kernel(int n_dim, std::size_t total,
                               const float* diag,
                               const float* plusi, const float* plusj, const float* plusk,
                               const float* minusi, const float* minusj, const float* minusk,
                               const float* x, float* out) {
    std::size_t t = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;

    int plane = n_dim * n_dim;
    int i = t / plane;
    int rem = t - (std::size_t)i * plane;
    int j = rem / n_dim;
    int k = rem - j * n_dim;

    float val = diag[t] * x[t];
    if (i > 0) val += minusi[t] * x[t - plane];
    if (i + 1 < n_dim) val += plusi[t] * x[t + plane];
    if (j > 0) val += minusj[t] * x[t - n_dim];
    if (j + 1 < n_dim) val += plusj[t] * x[t + n_dim];
    if (k > 0) val += minusk[t] * x[t - 1];
    if (k + 1 < n_dim) val += plusk[t] * x[t + 1];
    out[t] = val;
}

__global__ void scale_kernel(const float* temp, const float* scaleBase, float* scale, int num_modes) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < num_modes) {
        scale[m] = scaleBase[m] * temp[m];
    }
}

// y = x (device copy)
__global__ void vec_copy(int n, const float* x, float* y) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) y[t] = x[t];
}

bool allocate_diag(std::size_t total, const float* src, float** dst) {
    if (!src) return false;
    CHECK_CUDA(cudaMalloc((void**)dst, total * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(*dst, src, total * sizeof(float), cudaMemcpyHostToDevice));
    return true;
}
} // namespace

bool gpu_amult_init(int n_dim,
    const float* Adiag,
    const float* Adiagplusi,
    const float* Adiagplusj,
    const float* Adiagplusk,
    const float* Adiagminusi,
    const float* Adiagminusj,
    const float* Adiagminusk,
    const float* J_host,
    std::size_t J_stride,
    const double* scaleBase_host,
    int num_modes) {

    gpu_amult_destroy(); // clean previous state if any

    if (n_dim <= 0 || num_modes <= 0) return false;

    g_ctx.n_dim = n_dim;
    g_ctx.num_modes = num_modes;
    g_ctx.total = (std::size_t)n_dim * n_dim * n_dim;

    // allocate diag parts
    if (!allocate_diag(g_ctx.total, Adiag, &g_ctx.d_Adiag)) return false;
    if (!allocate_diag(g_ctx.total, Adiagplusi, &g_ctx.d_Adiagplusi)) return false;
    if (!allocate_diag(g_ctx.total, Adiagplusj, &g_ctx.d_Adiagplusj)) return false;
    if (!allocate_diag(g_ctx.total, Adiagplusk, &g_ctx.d_Adiagplusk)) return false;
    if (!allocate_diag(g_ctx.total, Adiagminusi, &g_ctx.d_Adiagminusi)) return false;
    if (!allocate_diag(g_ctx.total, Adiagminusj, &g_ctx.d_Adiagminusj)) return false;
    if (!allocate_diag(g_ctx.total, Adiagminusk, &g_ctx.d_Adiagminusk)) return false;

    CHECK_CUDA(cudaMalloc((void**)&g_ctx.d_x, g_ctx.total * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&g_ctx.d_result, g_ctx.total * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&g_ctx.d_temp, num_modes * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&g_ctx.d_scale, num_modes * sizeof(float)));

    // scaleBase
    CHECK_CUDA(cudaMalloc((void**)&g_ctx.d_scaleBase, num_modes * sizeof(float)));
    {
        std::vector<float> sb(num_modes);
        for (int m = 0; m < num_modes; ++m) sb[m] = static_cast<float>(scaleBase_host[m]);
        CHECK_CUDA(cudaMemcpy(g_ctx.d_scaleBase, sb.data(), num_modes * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Prepare J in column-major layout for cuBLAS
    std::size_t lda = (std::size_t)num_modes;
    std::size_t elems = lda * g_ctx.total;
    CHECK_CUDA(cudaMalloc((void**)&g_ctx.d_J, elems * sizeof(float)));

    {
        // host scratch to reorder
        std::unique_ptr<float[]> hostJ(new (std::nothrow) float[elems]);
        if (!hostJ) {
            std::cerr << "Failed to allocate host buffer for J" << std::endl;
            gpu_amult_destroy();
            return false;
        }
        std::size_t stride = (J_stride == 0) ? g_ctx.total : J_stride;
        for (std::size_t c = 0; c < g_ctx.total; ++c) {
            const float* col = J_host + c; // start of column in row-major view
            for (int m = 0; m < num_modes; ++m) {
                hostJ[c * lda + m] = col[m * stride];
            }
        }
        CHECK_CUDA(cudaMemcpy(g_ctx.d_J, hostJ.get(), elems * sizeof(float), cudaMemcpyHostToDevice));
    }

    // cuBLAS handle
    CHECK_CUBLAS(cublasCreate(&g_ctx.handle));
    return true;
}

// void gpu_amult(const std::vector<double>& x, std::vector<double>& result) {
//     // Function to Calculate Ax
//     if (g_ctx.n_dim == 0) {
//         std::cerr << "gpu_amult called before init" << std::endl;
//         return;
//     }
//     if (x.size() < g_ctx.total) {
//         std::cerr << "gpu_amult: input size too small" << std::endl;
//         return;
//     }
//     result.resize(g_ctx.total);

//     // copy x (double) to device float buffer
//     std::vector<float> xf(g_ctx.total);
//     for (size_t i = 0; i < g_ctx.total; ++i) xf[i] = static_cast<float>(x[i]);
//     cudaMemcpy(g_ctx.d_x, xf.data(), g_ctx.total * sizeof(float), cudaMemcpyHostToDevice);

//     // stencil part
//     int threads = 256;
//     int blocks = (int)((g_ctx.total + threads - 1) / threads);
//     stencil_kernel<<<blocks, threads>>>(g_ctx.n_dim, g_ctx.total,
//         g_ctx.d_Adiag, g_ctx.d_Adiagplusi, g_ctx.d_Adiagplusj, g_ctx.d_Adiagplusk,
//         g_ctx.d_Adiagminusi, g_ctx.d_Adiagminusj, g_ctx.d_Adiagminusk,
//         g_ctx.d_x, g_ctx.d_result);

//     // temp = J * x
//     const float alpha = 1.0f;
//     const float beta0 = 0.0f;
//     cublasSgemv(g_ctx.handle, CUBLAS_OP_N, g_ctx.num_modes, (int)g_ctx.total,
//                 &alpha, g_ctx.d_J, g_ctx.num_modes, g_ctx.d_x, 1, &beta0, g_ctx.d_temp, 1);

//     // scale = scaleBase * temp
//     int mThreads = 128;
//     int mBlocks = (g_ctx.num_modes + mThreads - 1) / mThreads;
//     scale_kernel<<<mBlocks, mThreads>>>(g_ctx.d_temp, g_ctx.d_scaleBase, g_ctx.d_scale, g_ctx.num_modes);

//     // result += J^T * scale
//     const float beta1 = 1.0f;
//     cublasSgemv(g_ctx.handle, CUBLAS_OP_T, g_ctx.num_modes, (int)g_ctx.total,
//                 &alpha, g_ctx.d_J, g_ctx.num_modes, g_ctx.d_scale, 1, &beta1, g_ctx.d_result, 1);

//     // copy back (float -> double)
//     std::vector<float> rf(g_ctx.total);
//     cudaMemcpy(rf.data(), g_ctx.d_result, g_ctx.total * sizeof(float), cudaMemcpyDeviceToHost);
//     for (size_t i = 0; i < g_ctx.total; ++i) result[i] = static_cast<double>(rf[i]);
// }

// Device-only A*x using already-uploaded matrix, no host transfers.
static bool gpu_amult_device(const float* d_x, float* d_out) {
    if (g_ctx.n_dim == 0) return false;

    int threads = 256;
    int blocks = (int)((g_ctx.total + threads - 1) / threads);
    stencil_kernel<<<blocks, threads>>>(g_ctx.n_dim, g_ctx.total,
        g_ctx.d_Adiag, g_ctx.d_Adiagplusi, g_ctx.d_Adiagplusj, g_ctx.d_Adiagplusk,
        g_ctx.d_Adiagminusi, g_ctx.d_Adiagminusj, g_ctx.d_Adiagminusk,
        d_x, d_out);

    const float alpha = 1.0f;
    const float beta0 = 0.0f;
    cublasStatus_t st1 = cublasSgemv(g_ctx.handle, CUBLAS_OP_N, g_ctx.num_modes, (int)g_ctx.total,
                                     &alpha, g_ctx.d_J, g_ctx.num_modes, d_x, 1, &beta0, g_ctx.d_temp, 1);
    if (st1 != CUBLAS_STATUS_SUCCESS) return false;

    int mThreads = 128;
    int mBlocks = (g_ctx.num_modes + mThreads - 1) / mThreads;
    scale_kernel<<<mBlocks, mThreads>>>(g_ctx.d_temp, g_ctx.d_scaleBase, g_ctx.d_scale, g_ctx.num_modes);

    const float beta1 = 1.0f;
    cublasStatus_t st2 = cublasSgemv(g_ctx.handle, CUBLAS_OP_T, g_ctx.num_modes, (int)g_ctx.total,
                                     &alpha, g_ctx.d_J, g_ctx.num_modes, g_ctx.d_scale, 1, &beta1, d_out, 1);
    if (st2 != CUBLAS_STATUS_SUCCESS) return false;
    return true;
}

// Fully GPU-resident Conjugate Gradient using the uploaded matrix.
// the main function
bool gpu_cg_solve(const std::vector<double>& d_host,
                  std::vector<double>& p_host,
                  double tol,
                  int max_iter,
                  const std::vector<double>* p0) {
    
    if (g_ctx.n_dim == 0 || g_ctx.handle == nullptr) {
        std::cerr << "gpu_cg_solve called before gpu_amult_init" << std::endl;
        return false;
    }

    const size_t n = g_ctx.total;
    if (d_host.size() < n) {
        std::cerr << "gpu_cg_solve: d size mismatch" << std::endl;
        return false;
    }

    p_host.resize(n);

    // device buffers
    float *d_p = nullptr, *d_r = nullptr, *d_s = nullptr, *d_As = nullptr, *d_r_new = nullptr, *d_rhs = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_p, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_r, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_s, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_As, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_r_new, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_rhs, n * sizeof(float)));

    // upload RHS
    {
        std::vector<float> h_rhs(n);
        for (size_t i = 0; i < n; ++i) h_rhs[i] = static_cast<float>(d_host[i]);
        CHECK_CUDA(cudaMemcpy(d_rhs, h_rhs.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }

    // init p
    if (p0 && p0->size() >= n) {
        std::vector<float> h_p0(n);
        for (size_t i = 0; i < n; ++i) h_p0[i] = static_cast<float>((*p0)[i]);
        CHECK_CUDA(cudaMemcpy(d_p, h_p0.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        std::vector<float> ones(n, 1.0f);
        CHECK_CUDA(cudaMemcpy(d_p, ones.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }

    // r = d - A*p
    if (!gpu_amult_device(d_p, d_As)) {
        gpu_amult_destroy();
        return false;
    }
    CHECK_CUDA(cudaMemcpy(d_r, d_rhs, n * sizeof(float), cudaMemcpyDeviceToDevice));
    const float minus_one = -1.0f;
    CHECK_CUBLAS(cublasSaxpy(g_ctx.handle, (int)n, &minus_one, d_As, 1, d_r, 1));

    // s = r
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    vec_copy<<<blocks, threads>>>((int)n, d_r, d_s);

    float r_dot_r = 0.0f;
    CHECK_CUBLAS(cublasSdot(g_ctx.handle, (int)n, d_r, 1, d_r, 1, &r_dot_r));

    const double tol2 = tol * tol;

    for (int iter = 0; iter < max_iter; ++iter) {
        if (!gpu_amult_device(d_s, d_As)) {
            gpu_amult_destroy();
            return false;
        }

        float s_dot_As = 0.0f;
        CHECK_CUBLAS(cublasSdot(g_ctx.handle, (int)n, d_s, 1, d_As, 1, &s_dot_As));
        if (std::abs(s_dot_As) < 1e-15) {
            std::cout << "Warning: Division by near-zero in CG iteration " << iter << std::endl;
            break;
        }

        float alpha = r_dot_r / s_dot_As;

        // p += alpha * s
        CHECK_CUBLAS(cublasSaxpy(g_ctx.handle, (int)n, &alpha, d_s, 1, d_p, 1));

        // r_new = r - alpha * As
        CHECK_CUDA(cudaMemcpy(d_r_new, d_r, n * sizeof(float), cudaMemcpyDeviceToDevice));
        float neg_alpha = -alpha;
        CHECK_CUBLAS(cublasSaxpy(g_ctx.handle, (int)n, &neg_alpha, d_As, 1, d_r_new, 1));

        float r_new_dot_r_new = 0.0f;
        CHECK_CUBLAS(cublasSdot(g_ctx.handle, (int)n, d_r_new, 1, d_r_new, 1, &r_new_dot_r_new));

        if (r_new_dot_r_new < tol2) {
            std::vector<float> h_p(n);
            CHECK_CUDA(cudaMemcpy(h_p.data(), d_p, n * sizeof(float), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < n; ++i) p_host[i] = static_cast<double>(h_p[i]);
            cudaFree(d_p); cudaFree(d_r); cudaFree(d_s); cudaFree(d_As); cudaFree(d_r_new); cudaFree(d_rhs);
            std::cout << "CG converged in " << iter + 1 << " iterations, residual: " << std::sqrt(r_new_dot_r_new) << std::endl;
            return true;
        }

        float beta = r_new_dot_r_new / r_dot_r;

        // s = r_new + beta * s
        CHECK_CUBLAS(cublasSscal(g_ctx.handle, (int)n, &beta, d_s, 1));
        const float one = 1.0f;
        CHECK_CUBLAS(cublasSaxpy(g_ctx.handle, (int)n, &one, d_r_new, 1, d_s, 1));

        // swap r and r_new pointers
        std::swap(d_r, d_r_new);
        r_dot_r = r_new_dot_r_new;
    }

    // copy solution back
    {
        std::vector<float> h_p(n);
        CHECK_CUDA(cudaMemcpy(h_p.data(), d_p, n * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; ++i) p_host[i] = static_cast<double>(h_p[i]);
    }

    cudaFree(d_p); cudaFree(d_r); cudaFree(d_s); cudaFree(d_As); cudaFree(d_r_new); cudaFree(d_rhs);

    std::cout << "CG did not converge after " << max_iter << " iterations, final residual: " << std::sqrt(r_dot_r) << std::endl;
    return false;
}

void gpu_amult_destroy() {
    if (g_ctx.handle) {
        cublasDestroy(g_ctx.handle);
        g_ctx.handle = nullptr;
    }
    cudaFree(g_ctx.d_Adiag); g_ctx.d_Adiag = nullptr;
    cudaFree(g_ctx.d_Adiagplusi); g_ctx.d_Adiagplusi = nullptr;
    cudaFree(g_ctx.d_Adiagplusj); g_ctx.d_Adiagplusj = nullptr;
    cudaFree(g_ctx.d_Adiagplusk); g_ctx.d_Adiagplusk = nullptr;
    cudaFree(g_ctx.d_Adiagminusi); g_ctx.d_Adiagminusi = nullptr;
    cudaFree(g_ctx.d_Adiagminusj); g_ctx.d_Adiagminusj = nullptr;
    cudaFree(g_ctx.d_Adiagminusk); g_ctx.d_Adiagminusk = nullptr;

    cudaFree(g_ctx.d_J); g_ctx.d_J = nullptr;
    cudaFree(g_ctx.d_scaleBase); g_ctx.d_scaleBase = nullptr;

    cudaFree(g_ctx.d_x); g_ctx.d_x = nullptr;
    cudaFree(g_ctx.d_result); g_ctx.d_result = nullptr;
    cudaFree(g_ctx.d_temp); g_ctx.d_temp = nullptr;
    cudaFree(g_ctx.d_scale); g_ctx.d_scale = nullptr;

    g_ctx.n_dim = 0;
    g_ctx.num_modes = 0;
    g_ctx.total = 0;
}
