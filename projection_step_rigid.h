#include <vector>
#include <cmath>
#include <functional>
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <algorithm>

// 可选 OpenMP 加速宏，编译时可定义为 0 以禁用
#ifndef CG_USE_OPENMP
#define CG_USE_OPENMP 1
#endif

// 有些编译环境没有 <omp.h>，使用 __has_include 做保护；若缺失则禁用 OpenMP 支持
#if CG_USE_OPENMP
#if defined(__has_include)
#if __has_include(<omp.h>)
#include <omp.h>
#else
#undef CG_USE_OPENMP
#define CG_USE_OPENMP 0
#endif
#else
/* If __has_include is not available, attempt to include <omp.h>.
    If this fails during compile, define CG_USE_OPENMP=0 manually or
    compile with -DCG_USE_OPENMP=0 to avoid the missing header. */
#include <omp.h>
#endif
#endif

class ConjugateGradientSolver {
public:
    // 求解 Ap = d
    // A_mult: 函数对象，执行 y = A*x
    static bool solve(std::function<void(const std::vector<double>&, std::vector<double>&)> A_mult,
                     const std::vector<double>& d,
                     std::vector<double>& p,
                     double tol = 1e-5,
                     int max_iter = 500,
                     const std::vector<double>* p0 = nullptr) {

        int n = (int)d.size();

        // 初始化解向量 p
        if (p0 != nullptr) {
            p = *p0;  // 使用提供的初始猜测
        } else {
            p.resize(n, 1.0);  // 保持原行为（注意：原注释有误，非零向量）
        }

        // 复用缓冲，避免每次分配
        std::vector<double> r(n), s(n), As(n), r_new(n);

        // r = d - A*p
        A_mult(p, r);
#if CG_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n; ++i) r[i] = d[i] - r[i];

        // 初始搜索方向 s = r
        s = r;

        // 预计算 tol^2，避免每次 sqrt
        double tol2 = tol * tol;

        for (int iter = 0; iter < max_iter; ++iter) {
            
            A_mult(s, As);

            // 计算 r_dot_r 和 s_dot_As
            double r_dot_r = dot(r, r);
            double s_dot_As = dot(s, As);
            if (std::abs(s_dot_As) < 1e-15) {
                std::cout << "Warning: Division by near-zero in CG iteration " << iter << std::endl;
                return false;
            }
            double alpha = r_dot_r / s_dot_As;

            // 一次遍历完成：更新 p, 计算 r_new = r - alpha*As，并累加 r_new·r_new
            double r_new_dot_r_new = 0.0;
#if CG_USE_OPENMP
#pragma omp parallel for reduction(+:r_new_dot_r_new) schedule(static)
#endif
            for (int i = 0; i < n; ++i) {
                p[i] += alpha * s[i];
                r_new[i] = r[i] - alpha * As[i];
                r_new_dot_r_new += r_new[i] * r_new[i];
            }

            // 检查收敛（使用平方残差比较，避免 sqrt）
            if (r_new_dot_r_new < tol2) {
                double residual_norm = std::sqrt(r_new_dot_r_new);
                std::cout << "CG converged in " << iter + 1 << " iterations, residual: " << residual_norm << std::endl;
                p.swap(p); // no-op but keep semantic point where p is final
                return true;
            }

            double beta = r_new_dot_r_new / r_dot_r;

            // 更新搜索方向 s = r_new + beta * s
#if CG_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < n; ++i) {
                s[i] = r_new[i] + beta * s[i];
            }

            // 将 r_new 作为下一次的 r（复用缓冲）
            r.swap(r_new);
        }

        std::cout << "CG did not converge after " << max_iter << " iterations, final residual: " << std::sqrt(dot(r, r)) << std::endl;
        return false;
    }

private:
    // 计算向量的2-范数平方（内部使用）
    static double norm2(const std::vector<double>& v) {
        return dot(v, v);
    }

    // 计算向量的2-范数
    static double norm(const std::vector<double>& v) {
        double sum = 0.0;
    int n = (int)v.size();
#if CG_USE_OPENMP
#pragma omp parallel for reduction(+:sum) schedule(static)
#endif
    for (int i = 0; i < n; ++i) sum += v[i] * v[i];
        return std::sqrt(sum);
    }

    // 计算向量点积
    static double dot(const std::vector<double>& a, const std::vector<double>& b) {
        double result = 0.0;
        int n = (int)a.size();
#if CG_USE_OPENMP
#pragma omp parallel for reduction(+:result) schedule(static)
#endif
        for (int i = 0; i < n; ++i) result += a[i] * b[i];
        return result;
    }
};
class pressure_solver{

	public:
		int find_index(int i, int j, int k, int n_dim=5){
			return (i-1)*n_dim*n_dim+(j-1)*n_dim+k-1;
		}
		bool solve(bool print, int gridtype[123][123][123],  int n_dim, std::vector<double> d, std::vector<double>& p
            , float J[10][1234567], double constant_B, float rigid_m[10]){
        // runtime tuning to reduce per-thread memory and arenas (help WSL2)
        //setenv("MALLOC_ARENA_MAX", "1", 1);
        //setenv("OMP_STACKSIZE", "4M", 1);
        //std::cout<<"solving\n";
        int prefer_threads = 4;
        long nproc = sysconf(_SC_NPROCESSORS_ONLN);
        if (nproc > 0) prefer_threads = std::min(prefer_threads, (int)nproc);
#if CG_USE_OPENMP
        omp_set_num_threads(prefer_threads);
#endif
        //std::cout<<"solving\n";
            // Use heap-allocated vectors sized to the grid (avoid large stack arrays)
            const size_t total = (size_t)n_dim * n_dim * n_dim;
            // use float to halve memory for these coefficient arrays
            std::vector<float> Adiag(total, 0.0f), Adiagplusi(total, 0.0f), Adiagplusj(total, 0.0f), Adiagminusi(total, 0.0f), Adiagminusj(total, 0.0f);
            std::vector<float> Adiagplusk(total, 0.0f), Adiagminusk(total, 0.0f);

			for(int i=1;i<=n_dim;i++)
			for(int j=1;j<=n_dim;j++){
            for(int k=1;k<=n_dim;k++){
                if(gridtype[i][j][k]==1){
				    Adiag[find_index(i,j,k,n_dim)]=12-gridtype[i-1][j][k]-gridtype[i+1][j][k]-gridtype[i][j-1][k]-gridtype[i][j+1][k]-gridtype[i][j][k-1]-gridtype[i][j][k+1];
                    if(gridtype[i-1][j][k]==1) Adiagminusi[find_index(i,j,k,n_dim)]=-1;
				    if(gridtype[i+1][j][k]==1) Adiagplusi[find_index(i,j,k,n_dim)]=-1;
                    if(gridtype[i][j-1][k]==1) Adiagminusj[find_index(i,j,k,n_dim)]=-1;
                    if(gridtype[i][j+1][k]==1) Adiagplusj[find_index(i,j,k,n_dim)]=-1;
                    if(gridtype[i][j][k-1]==1) Adiagminusk[find_index(i,j,k,n_dim)]=-1;
                    if(gridtype[i][j][k+1]==1) Adiagplusk[find_index(i,j,k,n_dim)]=-1;
                }
                else if(gridtype[i][j][k]==0){
                    Adiag[find_index(i,j,k,n_dim)]=1.0;
                    d[find_index(i,j,k,n_dim)]=0.0;
                }
			} 
        }
        //std::cout<<"Successfully built A matrix components and d\n";
            auto A_mult = [&rigid_m, &constant_B,&J,&Adiag,&Adiagminusi,&Adiagminusj,&Adiagplusi,&Adiagplusj,&Adiagplusk,&Adiagminusk, n_dim](const std::vector<double>& x, std::vector<double>& result) {
                const size_t total = (size_t)n_dim * n_dim * n_dim;
                result.resize(total);

                // 1) base A*x from diag and neighbor entries
#if CG_USE_OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
                for (int i = 0; i < n_dim; ++i) {
                    for (int j = 0; j < n_dim; ++j) {
                        for (int k = 0; k < n_dim; ++k) {
                            size_t t = (size_t)i * n_dim * n_dim + j * n_dim + k;
                            double val = (double)Adiag[t] * x[t];
                            if (i - 1 >= 0) val += (double)Adiagminusi[t] * x[t - n_dim * n_dim];
                            if (i + 1 <= n_dim - 1) val += (double)Adiagplusi[t] * x[t + n_dim * n_dim];
                            if (j - 1 >= 0) val += (double)Adiagminusj[t] * x[t - n_dim];
                            if (j + 1 <= n_dim - 1) val += (double)Adiagplusj[t] * x[t + n_dim];
                            if (k - 1 >= 0) val += (double)Adiagminusk[t] * x[t - 1];
                            if (k + 1 <= n_dim - 1) val += (double)Adiagplusk[t] * x[t + 1];
                            result[t] = val;
                        }
                    }
                }

                // 2) add the contribution of J * M^{-1} * J^T
                const int num_modes = 6; 
                std::vector<double> temp(num_modes + 1, 0.0);
                for (int m = 1; m <= num_modes; ++m) {
                    double s = 0.0;
#if CG_USE_OPENMP
#pragma omp parallel for reduction(+:s) schedule(static)
#endif
                    for (size_t c = 0; c < total; ++c) s += (double)J[m][c] * x[c];
                    temp[m] = s;
                }
                for (int m = 1; m <= num_modes; ++m) {
                    double scale = constant_B * (1.0 / rigid_m[m]) * temp[m];
#if CG_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
                    for (size_t t = 0; t < total; ++t) result[t] += scale * (double)J[m][t];
                }
            };

			bool success = ConjugateGradientSolver::solve(A_mult, d, p);

            if(print){
                std::cout<<"Not Implemented\n";
            }
			return success;	
		}
	
};