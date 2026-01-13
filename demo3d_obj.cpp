#include "fluid3d.h"
#include "solid.h"
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// 100*100*100每个时间步大约16秒（40个时间步总计用时9分41秒）
// 输入100 51 用时2分26秒77

// 修复代码中Container的错误，15分钟

// 12.28更改Writeobj为立方体，17/15
// 要不要改成矩阵，17/15分钟

// 1.13处理刚体碰撞，50/30分钟


//  Simple OBJ exporter: write a face on the interface where two adjacent voxels
//  have different `filled` states. This is intentionally minimal: each face
//  emits four vertices (no deduplication) and one quad face.

void writeOBJFromFluid(const Fluid &f, const std::string &filename,
                       double voxelSize = 1.0) {
  std::ofstream out(filename, std::ios::out);
  if (!out) {
    std::cerr << "Failed to open OBJ file for writing: " << filename
              << std::endl;
    return;
  }

  int nx = f.numX - 2;
  int ny = f.numY - 2;
  int nz = f.numZ - 2;
  int maxN = std::max(std::max(nx, ny), nz);

  int offX = (maxN - nx) / 2;
  int offY = (maxN - ny) / 2;
  int offZ = (maxN - nz) / 2;

  // We'll output each face as four vertices and one face line. Track vertex
  // index.
  size_t vid = 1; // OBJ is 1-based

  // Prepare MTL filename (same base name as OBJ) and write header linking it
  std::string mtlFilename = filename;
  size_t posdot = mtlFilename.rfind('.');
  if (posdot != std::string::npos) mtlFilename = mtlFilename.substr(0, posdot) + ".mtl";
  else mtlFilename += ".mtl";
  // mtllib line should reference the basename of the mtl file
  std::string mtlBase = mtlFilename;
  size_t sl = mtlBase.find_last_of("/\\");
  if (sl != std::string::npos) mtlBase = mtlBase.substr(sl + 1);
  out << "mtllib " << mtlBase << "\n";

  // create the MTL file next to the OBJ (defines Water and Ball materials)
  std::ofstream mout(mtlFilename, std::ios::out);
  if (mout) {
    mout << "newmtl Water\n";
    mout << "Ka 0.0 0.05 0.08\n";
    mout << "Kd 0.0 0.2 1.0\n";
    mout << "Ks 0.1 0.1 0.1\n";
    mout << "Ns 10.0\n";
    mout << "\n";
    mout << "newmtl Ball\n";
    mout << "Ka 0.1 0.0 0.0\n";
    mout << "Kd 1.0 0.0 0.0\n";
    mout << "Ks 0.3 0.3 0.3\n";
    mout << "Ns 50.0\n";
    mout.close();
  } else {
    std::cerr << "Warning: failed to write MTL file: " << mtlFilename << std::endl;
  }

  // 添加绿色长方体材质（用于导出的附加盒子）
  if (mout) {
    std::ofstream mout2(mtlFilename, std::ios::out | std::ios::app);
    if (mout2) {
      mout2 << "\n";
      mout2 << "newmtl GreenBox\n";
      mout2 << "Ka 0.0 0.05 0.0\n";
      mout2 << "Kd 0.0 0.8 0.0\n";
      mout2 << "Ks 0.1 0.1 0.1\n";
      mout2 << "Ns 20.0\n";
      mout2.close();
    }
  }

  // start with fluid material
  out << "g fluid\n";
  out << "usemtl Water\n";

  auto voxelToWorld = [&](int dx, int dy, int dz) {
    // convert voxel integer coord to world-space corner (min corner)
    double x = (dx - 0.5) * voxelSize;
    double y = (dy - 0.5) * voxelSize;
    double z = (dz - 0.5) * voxelSize;
    return std::array<double, 3>{x, y, z};
  };

  // iterate logical fluid cells (1..num?-2)
  for (int j = 1; j <= f.numY - 2; ++j) {
    for (int k = 1; k <= f.numZ - 2; ++k) {
      for (int i = 1; i <= f.numX - 2; ++i) {
        bool me = f.filled[f.index(i, j, k)];
        int dx = (i - 1) + offX;
        int dy = (j - 1) + offY;
        int dz = (k - 1) + offZ;

        // neighbor +X
        bool neigh = false;
        if (i + 1 <= f.numX - 2)
          neigh = f.filled[f.index(i + 1, j, k)];
        if (me != neigh) {
          // face at x = (dx+0.5)
          double x = (dx + 0.5) * voxelSize;
          double y0 = (dy - 0.5) * voxelSize;
          double y1 = (dy + 0.5) * voxelSize;
          double z0 = (dz - 0.5) * voxelSize;
          double z1 = (dz + 0.5) * voxelSize;
          // order vertices so normal points outward from filled voxel when
          // me==true
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
          out << "f " << vid << " " << vid + 1 << " " << vid + 2 << " "
              << vid + 3 << "\n";
          vid += 4;
        }

        // neighbor +Y
        neigh = false;
        if (j + 1 <= f.numY - 2)
          neigh = f.filled[f.index(i, j + 1, k)];
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
          out << "f " << vid << " " << vid + 1 << " " << vid + 2 << " "
              << vid + 3 << "\n";
          vid += 4;
        }

        // neighbor +Z
        neigh = false;
        if (k + 1 <= f.numZ - 2)
          neigh = f.filled[f.index(i, j, k + 1)];
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
          out << "f " << vid << " " << vid + 1 << " " << vid + 2 << " "
              << vid + 3 << "\n";
          vid += 4;
        }
      }
    }
  }
  // If a rigid body is attached, append a UV-sphere mesh for it.
  auto appendSphere = [&](std::ostream &out, size_t &vid,
                          double cx, double cy, double cz, double r,
                          int sectors = 48, int stacks = 24) {
    double PI = std::acos(-1.0);
    size_t start = vid;
    for (int i = 0; i <= stacks; ++i) {
      double phi = PI * double(i) / double(stacks); // 0..pi
      double sinPhi = std::sin(phi), cosPhi = std::cos(phi);
      for (int j = 0; j <= sectors; ++j) {
        double theta = 2.0 * PI * double(j) / double(sectors); // 0..2pi
        double sinT = std::sin(theta), cosT = std::cos(theta);
        double x = cx + r * sinPhi * cosT;
        double y = cy + r * cosPhi;
        double z = cz + r * sinPhi * sinT;
        out << "v " << x << " " << y << " " << z << "\n";
        ++vid;
      }
    }
    int rowVerts = sectors + 1;
    for (int i = 0; i < stacks; ++i) {
      for (int j = 0; j < sectors; ++j) {
        size_t v1 = start + i * rowVerts + j;
        size_t v2 = start + (i + 1) * rowVerts + j;
        size_t v3 = start + (i + 1) * rowVerts + (j + 1);
        size_t v4 = start + i * rowVerts + (j + 1);
        out << "f " << v1 << " " << v2 << " " << v3 << "\n";
        out << "f " << v1 << " " << v3 << " " << v4 << "\n";
      }
    }
  };

  if (f.m_rigid) {
    auto pos = f.m_rigid->position();
    auto ori = f.m_rigid->orientation();
    change3d::Mat3 R = ori.toMat3();
    const auto &bunny = change3d::getBunnyMesh();
    if (!bunny.loaded || bunny.verts.empty() || bunny.tris.empty()) {
      std::cerr << "Warning: bunny_200.obj not loaded; skipping rigid body export" << std::endl;
    } else {
      out << "g ball\n";
      out << "usemtl Ball\n";
      size_t start = vid;
      // write transformed vertices
      for (const auto &v : bunny.verts) {
        change3d::Vec3 w = R.mulVec(v) + pos; // centroid already baked into mesh
        out << "v " << w.x << " " << w.y << " " << w.z << "\n";
        ++vid;
      }
      // write faces using the transformed vertices
      for (const auto &tri : bunny.tris) {
        out << "f " << start + tri[0] << " " << start + tri[1] << " " << start + tri[2] << "\n";
      }
    }
  }

  // 在OBJ末尾添加一个与轴平行的长方体，两个对角为 (75,0,75) 和 (100,25,100)
  // 六个面使用材料 GreenBox（应在 MTL 中定义）
  {
    double minx = 80.0, miny = 0.0, minz = 80.0;
    double maxx = 120.0, maxy = 40.0, maxz = 120.0;
    out << "g AddedBox\n";
    out << "usemtl GreenBox\n";
    // vertices
    out << "v " << minx << " " << miny << " " << minz << "\n"; // v0
    out << "v " << maxx << " " << miny << " " << minz << "\n"; // v1
    out << "v " << maxx << " " << maxy << " " << minz << "\n"; // v2
    out << "v " << minx << " " << maxy << " " << minz << "\n"; // v3
    out << "v " << minx << " " << miny << " " << maxz << "\n"; // v4
    out << "v " << maxx << " " << miny << " " << maxz << "\n"; // v5
    out << "v " << maxx << " " << maxy << " " << maxz << "\n"; // v6
    out << "v " << minx << " " << maxy << " " << maxz << "\n"; // v7
    // faces (quads). Use current vid index
    out << "f " << vid << " " << vid + 1 << " " << vid + 2 << " " << vid + 3 << "\n"; // bottom (z=min)
    vid += 4;
    out << "f " << vid << " " << vid + 1 << " " << vid + 2 << " " << vid + 3 << "\n"; // top (z=max)
    vid += 4;
    // side faces
    out << "f " << (vid - 8) << " " << (vid - 4) << " " << (vid - 1) << " " << (vid - 5) << "\n";
    out << "f " << (vid - 7) << " " << (vid - 3) << " " << (vid - 2) << " " << (vid - 6) << "\n";
    out << "f " << (vid - 8) << " " << (vid - 7) << " " << (vid - 3) << " " << (vid - 4) << "\n";
    out << "f " << (vid - 6) << " " << (vid - 2) << " " << (vid - 1) << " " << (vid - 5) << "\n";
    // note: the index arithmetic above references the last 8 vertices we wrote
  }

  out.close();
  std::cout << "Wrote OBJ: " << filename << " (faces: " << (vid - 1) / 4
            << ")\n";
}

int main() {
  int n_input, max_step;
  std::cin >> n_input >> max_step;
  Fluid f(1.0f, n_input, n_input, n_input, 1.0f); // h=1.0
  float dt = 0.1f;
  float gravity = 0.0f;

  // Initialize RigidBody
  change3d::RigidBody rigidBody;
  rigidBody.setMass(
      3200.0); // Example mass，如果mass=27，体积=27m^3，则密度和水完全一样
              // set body-space inertia for a 3m cube mass=10kg: I ~ 15 kg·m^2
              // on diag  如果半径为9.4，体积大约为4000
  change3d::Mat3 Ibody(120000.0);//默认质量1000对应40000
  rigidBody.setInertiaBody(Ibody);
  rigidBody.setPosition(change3d::Vec3(
      30.0, 75.00, 30.0)); // Example initial position of the center
  rigidBody.setLinearVelocity(
      change3d::Vec3(0.0, 0.0, 0.0)); // Example velocity

  // Associate RigidBody with Fluid
  f.m_rigid = &rigidBody;

  auto applyFreeSlip = [&](Fluid &F) {
   //int maxi, maxj, maxk;
   // std::cin>>maxi>>maxj>>maxk;
    for (int i = 0; i < F.numX; ++i) {
      for (int j = 0; j < F.numY; ++j) {
        for (int k = 0; k < F.numZ; ++k) {
          F.u[F.index(i, j, k)] = F.v[F.index(i, j, k)] =
              F.w[F.index(i, j, k)] = 0.0f;
        }
      }
    }
    for (int i = 1; i <= 60; i++) {// 60, 50, 60
      for (int j = 1; j <= 90; j++) {
        for (int k = 1; k <= 60; k++) {
          F.filled[F.index(i, j, k)] = true;
        }
      }
    }
    for (int i = 1; i <= F.numX - 2; i++)
      for (int j = 1; j <= F.numY - 2; j++)
        for (int k = 1; k <= F.numZ - 2; k++) {
          F.is_container[F.index(i, j, k)] = false;
        }
    for(int i=80;i<=F.numX - 2;i++)
    for(int j=1;j<=40;j++)
    for(int k=80;k<=F.numZ - 2;k++){
        F.is_container[F.index(i, j, k)] = true;
    }
    for (int i = 0; i < F.numX; ++i)
      for (int j = 0; j < F.numY; ++j)
        for (int k = 0; k < F.numZ; ++k)
          F.p[F.index(i, j, k)] = 0.0f;
  };

  applyFreeSlip(f);

  f.calculate_is_rigid_body(); // 计算刚体占据了哪些格子。
  f.treat_rigid_as_container();
  f.initialize();

  for (int step = 0; step < max_step; ++step) {
    std::cout << "Step " << step << "\n";

    f.integrate(dt, 0.0, -8.0, 0.0);

    f.calculate_extended_velocity();
    //输出精确到纳秒的时间
    auto now = std::chrono::high_resolution_clock::now();
    std::cout << "Current time (ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() << "\n";

    f.advect(dt);
    f.force_zero(0);

    now = std::chrono::high_resolution_clock::now();
    std::cout << "0Advect time (ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() << "\n";
    // std::cout << "Step " << step << ": Information before projection" <<
    // std::endl;
    //     for(int i=4;i<=6;i++)
    //       for(int j=7;j<=7;j++)
    //         for(int k=4;k<=6;k++){
    //             std::cout<<"filled["<<i<<","<<j<<","<<k<<"]="<<f.filled[f.index(i,j,k)]<<"\n";
    //             std::cout<<"is_rigid_["<<i<<","<<j<<","<<k<<"]="<<f.is_rigid_body[f.index(i,j,k)]<<"\n";
    //             std::cout<<"is_container_["<<i<<","<<j<<","<<k<<"]="<<f.is_container[f.index(i,j,k)]<<"\n";
    //         }

    // 统计有多少个格子被刚体占据并输出
    // int rigid_occupied_count = 0;
    // for (int i = 1; i <= f.numX - 2; i++)
    //   for (int j = 1; j <= f.numY - 2; j++)
    //     for (int k = 1; k <= f.numZ - 2; k++) {
    //       int c = f.index(i, j, k);
    //       if (f.is_rigid_body[c]) {
    //         rigid_occupied_count++;
    //       }
    //     }
    // std::cout << "Rigid body occupies " << rigid_occupied_count << " cells.\n";
    f.Project(0, f.h, f.h, f.h, dt);

    now = std::chrono::high_resolution_clock::now();
    std::cout << "Project time (ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() << "\n";

    // std::cout << "Step " << step << ": after Project" << std::endl;
    // 已经包含了对于刚体位置的更新
    // 输出刚体的中心的位置
    // change3d::Vec3 rigidPos_ = f.m_rigid->position();
    // std::cout << "Rigid Body Position: (" << rigidPos_.x << ", " <<
    // rigidPos_.y << ", " << rigidPos_.z << ")\n";

    f.force_zero(0);
    // std::cout << "Step " << step << ": after force_zero(0) (post-Project)" <<
    // std::endl;
    f.calculate_extended_velocity();
    // std::cout << "Step " << step << ": after calculate_extended_velocity
    // (post-Project)" << std::endl;
    f.update_particles(dt/4);
    f.update_particles(dt/4);
    f.update_particles(dt/4);
    f.update_particles(dt/4);
    // std::cout << "Step " << step << ": after update_particles" << std::endl;
    //  f.calculate_is_rigid_body();//计算刚体占据了哪些格子。
    //  f.treat_rigid_as_container(1);
    f.check_filled(0); // 检查哪些格子有液体
    f.after_check_filled();
    f.store_is_rigid_body();
    f.calculate_is_rigid_body(); // 计算刚体占据了哪些格子。
    f.treat_rigid_as_container();
    f.replace_rigid_body_fluid();
    // 如果一个格子上一步is_rigid_body是true，这一帧却变成false了，说明刚体离开了这个格子，那么需要用流体填充这个格子

    // for(int i=4;i<=6;i++)
    //   for(int j=7;j<=7;j++)
    //     for(int k=4;k<=6;k++){
    //         std::cout<<"filled["<<i<<","<<j<<","<<k<<"]="<<f.filled[f.index(i,j,k)]<<"\n";
    //     }

    f.print_highest_fluid_block();

    //输出哪些高为47的网格是container

    // for (int i = 1; i <= f.numX - 2; i++)
    //   for (int k = 1; k <= f.numZ - 2; k++) {
    //     int c = f.index(i, 47, k);
    //     if (f.is_container[c]) {
    //       std::cout << "Container at (" << i << ",47," << k << ")\n";
    //     }
    //   }
    //   for (int i = 1; i <= f.numX - 2; i++)
    //   for (int k = 1; k <= f.numZ - 2; k++) {
    //     int c = f.index(i, 47, k);
    //     if (f.is_rigid_body[c]) {
    //       std::cout << "Rigid body at (" << i << ",47," << k << ")\n";
    //     }
    //   }
    if (step % 5 ==0) {
      std::ostringstream oname;
      oname << "011301bunny_120_" << step << ".obj";
      writeOBJFromFluid(f, oname.str(), 1.0);
    }
    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
  return 0;
}
// export PATH=$PATH:/snap/bin
// export PATH=/usr/local/cuda/bin:$PATH
// export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
// ulimit -s unlimited
// 渲染主要使用：blender --background --python ds2.py
// 视频生成主要使用：（在具有很多图片的那个文件夹里面）
//ffmpeg -framerate 10 -pattern_type glob -i 'ball_*_render.png' -c:v libx264 -pix_fmt yuv420p out.mp4

//mkdir -p build/longrun2 && find build -maxdepth 1 -type f -name '1228ball_*' -exec mv {} build/longrun/ \;

/*
mkdir -p tmp_seq
i=1
for f in $(printf "%s\n" 1229bunny_120_*_render.png | sort -V); do
  ln -sf "$PWD/$f" tmp_seq/$(printf "img%03d.png" $i)
  i=$((i+1))
done
ffmpeg -framerate 20 -i tmp_seq/img%03d.png -c:v libx264 -pix_fmt yuv420p out_20.mp4
rm -r tmp_seq
*/
/*
Step 80
Rigid body occupies 3371 cells.
CG converged in 189 iterations, residual: 1.78335e-05
Original Force Jp=8061.79
Gravity Added: -8000
-1066.38 61.7905 2727.12 -1994.72 861.336 232.475
Rigid Body Position Before Update: (34.0267, 7.75577, 33.2608)
Rigid Body Position After Update: (34.3285, 7.39662, 33.5666)
Rigid Body Linear Velocity After Update: (3.01742, -3.59145, 3.05842)
Highest fluid block at y = 91 (grid j=91)
Wrote OBJ: ball_80.obj (faces: 48171)

*/


//运行
/*
cd /home/oier/acgpj_change3d/build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native ..
cmake --build . -j
或者

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc ..
cmake --build . -j


*/
//140是最后一个有效时间步