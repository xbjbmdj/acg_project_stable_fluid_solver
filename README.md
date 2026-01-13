# acg_project_stable_fluid_solver
叉院第三学期《高等计算机图形学》课程项目：Stam的stable fluid算法。流体求解器。

## Usage

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc ..
cmake --build . -j
./demo3d_obj
```



Input two numbers, separated by a space. The first number is # of grids side length, the second number is the time steps you need to simulate. e.g. 50 40
