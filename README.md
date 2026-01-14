# acg_project_stable_fluid_solver
叉院第三学期《高等计算机图形学》课程项目：Stam的stable fluid算法。流体求解器。

## Code Structure

- CMakeLists.txt
- demo3d_obj.cpp: Main demo, including OBJ export.
- demo3d_ball.cpp: Another demo.
- fluid3d.h: 3D stable fluid solver core implementation.
- solid.h: Rigid body data, pose, inertia utilities.
- projection_step_rigid.h: Pressure projection coupled with rigid constraints.
- gpu_amult.h / gpu_amult.cu: GPU mat-vec and Conjugate Gradient solver.
- render_script.py: CLI-driven rendering/screenshot script.

## Usage

```bash
ulimit -s unlimited
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc ..
cmake --build . -j
./demo3d_obj
```

Before simulation, put your .obj file in ./build, and change the path in solid.h from "bunny_200.obj" to your filename.  

Input two numbers, separated by a space. The first number is # of grids side length, the second number is the time steps you need to simulate. e.g. 150 101

