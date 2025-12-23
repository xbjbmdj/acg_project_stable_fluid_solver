#include "fluid3d.h"
#include "solid.h"
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>
#include <sstream>
#include <iomanip>
#include <string>
#include <fstream>

//100*100*100每个时间步大约16秒（40个时间步总计用时9分41秒）

// Simple OBJ exporter: write a face on the interface where two adjacent voxels
// have different `filled` states. This is intentionally minimal: each face
// emits four vertices (no deduplication) and one quad face.

/*
oier@LAPTOP-U9912E3T:~/change3d/build$ cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native -ffast-math -fopenmp"
(base) oier@LAPTOP-U9912E3T:~/change3d/build$ cmake --build . -j
*/

void writeOBJFromFluid(const Fluid &f, const std::string &filename, double voxelSize = 1.0)
{
    std::ofstream out(filename, std::ios::out);
    if (!out) {
        std::cerr << "Failed to open OBJ file for writing: " << filename << std::endl;
        return;
    }

    int nx = f.numX - 2;
    int ny = f.numY - 2;
    int nz = f.numZ - 2;
    int maxN = std::max(std::max(nx, ny), nz);

    int offX = (maxN - nx) / 2;
    int offY = (maxN - ny) / 2;
    int offZ = (maxN - nz) / 2;

    // We'll output each face as four vertices and one face line. Track vertex index.
    size_t vid = 1; // OBJ is 1-based

    auto voxelToWorld = [&](int dx, int dy, int dz) {
        // convert voxel integer coord to world-space corner (min corner)
        double x = (dx - 0.5) * voxelSize;
        double y = (dy - 0.5) * voxelSize;
        double z = (dz - 0.5) * voxelSize;
        return std::array<double,3>{x,y,z};
    };

    // iterate logical fluid cells (1..num?-2)
    for (int j = 1; j <= f.numY - 2; ++j) {
        for (int k = 1; k <= f.numZ - 2; ++k) {
            for (int i = 1; i <= f.numX - 2; ++i) {
                bool me = f.filled[f.index(i,j,k)];
                int dx = (i - 1) + offX;
                int dy = (j - 1) + offY;
                int dz = (k - 1) + offZ;

                // neighbor +X
                bool neigh = false;
                if (i+1 <= f.numX-2) neigh = f.filled[f.index(i+1,j,k)];
                if (me != neigh) {
                    // face at x = (dx+0.5)
                    double x = (dx + 0.5) * voxelSize;
                    double y0 = (dy - 0.5) * voxelSize;
                    double y1 = (dy + 0.5) * voxelSize;
                    double z0 = (dz - 0.5) * voxelSize;
                    double z1 = (dz + 0.5) * voxelSize;
                    // order vertices so normal points outward from filled voxel when me==true
                    if (me) {
                        out << "v " << x << " " << y0 << " " << z0 << "\n";
                        out << "v " << x << " " << y0 << " " << z1 << "\n";
                        out << "v " << x << " " << y1 << " " << z1 << "\n";
                        out << "v " << x << " " << y1 << " " << z0 << "\n";
                    } else {
                        // flipped when filled is on the +X neighbor
                        out << "v " << x << " " << y0 << " " << z0 << "\n";
                        out << "v " << x << " " << y1 << " " << z0 << "\n";
                        out << "v " << x << " " << y1 << " " << z1 << "\n";
                        out << "v " << x << " " << y0 << " " << z1 << "\n";
                    }
                    out << "f " << vid << " " << vid+1 << " " << vid+2 << " " << vid+3 << "\n";
                    vid += 4;
                }

                // neighbor +Y
                neigh = false;
                if (j+1 <= f.numY-2) neigh = f.filled[f.index(i,j+1,k)];
                if (me != neigh) {
                    double y = (dy + 0.5) * voxelSize;
                    double x0 = (dx - 0.5) * voxelSize;
                    double x1 = (dx + 0.5) * voxelSize;
                    double z0 = (dz - 0.5) * voxelSize;
                    double z1 = (dz + 0.5) * voxelSize;
                    if (me) {
                        out << "v " << x0 << " " << y << " " << z0 << "\n";
                        out << "v " << x1 << " " << y << " " << z0 << "\n";
                        out << "v " << x1 << " " << y << " " << z1 << "\n";
                        out << "v " << x0 << " " << y << " " << z1 << "\n";
                    } else {
                        out << "v " << x0 << " " << y << " " << z0 << "\n";
                        out << "v " << x0 << " " << y << " " << z1 << "\n";
                        out << "v " << x1 << " " << y << " " << z1 << "\n";
                        out << "v " << x1 << " " << y << " " << z0 << "\n";
                    }
                    out << "f " << vid << " " << vid+1 << " " << vid+2 << " " << vid+3 << "\n";
                    vid += 4;
                }

                // neighbor +Z
                neigh = false;
                if (k+1 <= f.numZ-2) neigh = f.filled[f.index(i,j,k+1)];
                if (me != neigh) {
                    double z = (dz + 0.5) * voxelSize;
                    double x0 = (dx - 0.5) * voxelSize;
                    double x1 = (dx + 0.5) * voxelSize;
                    double y0 = (dy - 0.5) * voxelSize;
                    double y1 = (dy + 0.5) * voxelSize;
                    if (me) {
                        out << "v " << x0 << " " << y0 << " " << z << "\n";
                        out << "v " << x1 << " " << y0 << " " << z << "\n";
                        out << "v " << x1 << " " << y1 << " " << z << "\n";
                        out << "v " << x0 << " " << y1 << " " << z << "\n";
                    } else {
                        out << "v " << x0 << " " << y0 << " " << z << "\n";
                        out << "v " << x0 << " " << y1 << " " << z << "\n";
                        out << "v " << x1 << " " << y1 << " " << z << "\n";
                        out << "v " << x1 << " " << y0 << " " << z << "\n";
                    }
                    out << "f " << vid << " " << vid+1 << " " << vid+2 << " " << vid+3 << "\n";
                    vid += 4;
                }
            }
        }
    }

    out.close();
    std::cout << "Wrote OBJ: " << filename << " (faces: " << (vid-1)/4 << ")\n";
}

int main() {
  int n_input, max_step;
  std::cin >> n_input >> max_step;
  Fluid f(1.0f, n_input, n_input, n_input, 1.0f); //h=1.0
  float dt = 0.1f;
  float gravity = 0.0f;

  // Initialize RigidBody
  change3d::RigidBody rigidBody;
  rigidBody.setMass(250.0); // Example mass，如果mass=27，体积=27m^3，则密度和水完全一样
    // set body-space inertia for a 3m cube mass=10kg: I ~ 15 kg·m^2 on diag
  change3d::Mat3 Ibody(3000.0);
  rigidBody.setInertiaBody(Ibody);
  rigidBody.setPosition(change3d::Vec3(10.0, 20.00, 10.0)); // Example initial position of the center
  rigidBody.setLinearVelocity(change3d::Vec3(0.0, 0.0, 0.0)); // Example velocity

  // Associate RigidBody with Fluid
  f.m_rigid = &rigidBody;

  auto applyFreeSlip = [&](Fluid &F) {
    for (int i = 0; i < F.numX; ++i) {
      for (int j = 0; j < F.numY; ++j) {
        for (int k = 0; k < F.numZ; ++k) {
            F.u[F.index(i, j, k)] = F.v[F.index(i, j, k)] = F.w[F.index(i, j, k)] = 0.0f;
        }
      }
    }
    for (int i = 1; i <= 20; i++) {
      for (int j = 1; j <= 30; j++) {
        for (int k = 1; k <= 20; k++) {
            F.filled[F.index(i, j, k)] = true;
        }
      }
    }
    for(int i=1;i<=F.numX-2;i++)
    for(int j=1;j<=F.numY-2;j++)
    for(int k=1;k<=F.numZ-2;k++){
        F.is_container[F.index(i, j, k)] = false;
    }
    // for(int i=13;i<=18;i++)
    // for(int j=1;j<=15;j++)
    // for(int k=13;k<=18;k++){
    //     F.is_container[F.index(i, j, k)] = true;
    // }
    for (int i = 0; i < F.numX; ++i) for (int j = 0; j < F.numY; ++j) for (int k=0;k<F.numZ;++k) F.p[F.index(i,j,k)] = 0.0f;
  };

  applyFreeSlip(f);
  f.initialize();
    f.calculate_is_rigid_body();//计算刚体占据了哪些格子。
    f.treat_rigid_as_container();

  for (int step = 0; step < max_step; ++step) {
    std::cout << "Step " << step << "\n";

    f.integrate(dt, 0.0, -8.0, 0.0);
    
    
    f.calculate_extended_velocity();
    f.advect(dt);
    f.force_zero(0);
// std::cout << "Step " << step << ": Information before projection" << std::endl;
//     for(int i=4;i<=6;i++)
//       for(int j=7;j<=7;j++)
//         for(int k=4;k<=6;k++){
//             std::cout<<"filled["<<i<<","<<j<<","<<k<<"]="<<f.filled[f.index(i,j,k)]<<"\n";
//             std::cout<<"is_rigid_["<<i<<","<<j<<","<<k<<"]="<<f.is_rigid_body[f.index(i,j,k)]<<"\n";
//             std::cout<<"is_container_["<<i<<","<<j<<","<<k<<"]="<<f.is_container[f.index(i,j,k)]<<"\n";
//         }

    //统计有多少个格子被刚体占据并输出
    int rigid_occupied_count=0;
    for(int i=1;i<=f.numX-2;i++)
    for(int j=1;j<=f.numY-2;j++)
    for(int k=1;k<=f.numZ-2;k++){
        int c=f.index(i,j,k);
        if(f.is_rigid_body[c]){
            rigid_occupied_count++;
        }
    }
    std::cout<<"Rigid body occupies "<<rigid_occupied_count<<" cells.\n";
    f.Project(0, f.h, f.h, f.h, dt);

    //std::cout << "Step " << step << ": after Project" << std::endl;
    //已经包含了对于刚体位置的更新
    //输出刚体的中心的位置
    //change3d::Vec3 rigidPos_ = f.m_rigid->position();
    //std::cout << "Rigid Body Position: (" << rigidPos_.x << ", " << rigidPos_.y << ", " << rigidPos_.z << ")\n";

    f.force_zero(0);
    //std::cout << "Step " << step << ": after force_zero(0) (post-Project)" << std::endl;
    f.calculate_extended_velocity();
    //std::cout << "Step " << step << ": after calculate_extended_velocity (post-Project)" << std::endl;
    f.update_particles(dt);
    //std::cout << "Step " << step << ": after update_particles" << std::endl;
    // f.calculate_is_rigid_body();//计算刚体占据了哪些格子。
    // f.treat_rigid_as_container(1);
    f.check_filled(0);//检查哪些格子有液体
    f.after_check_filled();
    f.store_is_rigid_body();
    f.calculate_is_rigid_body();//计算刚体占据了哪些格子。
    f.treat_rigid_as_container();
    f.replace_rigid_body_fluid();
    //如果一个格子上一步is_rigid_body是true，这一帧却变成false了，说明刚体离开了这个格子，那么需要用流体填充这个格子


    
    // for(int i=4;i<=6;i++)
    //   for(int j=7;j<=7;j++)
    //     for(int k=4;k<=6;k++){
    //         std::cout<<"filled["<<i<<","<<j<<","<<k<<"]="<<f.filled[f.index(i,j,k)]<<"\n";
    //     }
    

    if (step % 5 == 0) {
        std::ostringstream oname;
        oname << "122306ball50_" << step << ".obj";
        writeOBJFromFluid(f, oname.str(), 1.0);
    }
    //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
  return 0;
}
