#pragma once
#include <assert.h>

#include <algorithm>
#include <cmath>
#include <queue>
#include <vector>

#include "projection_step_rigid.h"
#include "solid.h"

struct Fluid {
  enum Field {
    U_FIELD = 0,
    V_FIELD = 1,
    extended_U_FIELD = 2,
    extended_V_FIELD = 3,
    S_FIELD = 4
  };

  float density;
  int numX, numY, numZ, numCells;
  float h;
  std::vector<float> u, v, w, newU, newV, newW, p, s, m, newM;
  std::vector<float> extendedu, extendedv, extendedw;
  std::vector<bool> filled, old_filled;
  std::vector<bool> is_container;
  std::vector<bool> is_rigid_body;
  std::vector<bool> old_is_rigid_body;
  std::vector<bool> danger;
  change3d::RigidBody *m_rigid =
      nullptr;  // optional pointer to an external rigid body
  std::vector<float> x_pos, y_pos, z_pos;
  std::vector<int> statistic;
  std::vector<float> phi;  // level set for free surface
  bool hasFreeSurface = false;
  float overRelaxation = 1.9f;  // default if not set externally
  // u[0.5,1]指的是u[0,1]，u[1.5,1]指的是u[1,1]
  // v[1,0.5]指的是v[1,0]，v[1,1.5]指的是v[1,1]

  Fluid(float density_, int nx, int ny, int nz, float h_)
      : density(density_), numX(nx + 2), numY(ny + 2), numZ(nz + 2), h(h_) {
    numCells = numX * numY * numZ;
    u.assign(numCells, 0.0f);
    v.assign(numCells, 0.0f);
    w.assign(numCells, 0.0f);
    danger.assign(numCells, false);
    newU.assign(numCells, 0.0f);
    newV.assign(numCells, 0.0f);
    newW.assign(numCells, 0.0f);
    extendedu.assign(numCells, 0.0f);
    extendedv.assign(numCells, 0.0f);
    extendedw.assign(numCells, 0.0f);
    p.assign(numCells, 0.0f);
    s.assign(numCells, 1.0f);
    m.assign(numCells, 1.0f);
    newM.assign(numCells, 0.0f);
    phi.assign(numCells, 0.0f);
    statistic.assign(numCells, 0);
    filled.assign(numCells, false);  // true means fluid cell by default
    is_container.assign(numCells,
                        true);  // 先全部设成true，初始化的时候需要注意
    is_rigid_body.assign(numCells, false);
  }
  int index(int i, int j, int k) const {
    return i + j * numX + k * numX * numY;
  }
  // 2D index wrapper for backward compatibility (assumes k=0 slice)
  int index(int i, int j) const { return i + j * numX; }
  void store_is_rigid_body() { old_is_rigid_body = is_rigid_body; }
  void print_all_rigid_occupied(bool print_rigid,
                                bool print_fluids_too = false) {
    if (print_rigid) {
      for (int i = 1; i <= numX - 2; i++)
        for (int j = 1; j <= numY - 2; j++)
          for (int k = 1; k <= numZ - 2; k++) {
            int c = index(i, j, k);
            if (is_rigid_body[c]) {
              std::cout << "Rigid occupies cell: (" << i << "," << j << "," << k
                        << ")\n";
            }
          }
    }
    if (print_fluids_too) {
      for (int i = 5; i <= 5; i++)
        for (int j = 1; j <= numY - 2; j++)
          for (int k = 5; k <= 5; k++) {
            int c = index(i, j, k);
            if (filled[c]) {
              std::cout << "Fluid occupies cell: (" << i << "," << j << "," << k
                        << ")\n";
            }
          }
    }
  }

  void replace_rigid_body_fluid() {
    // 对每个格子
    old_filled = filled;
    // 把danger都设为0
    std::fill(danger.begin(), danger.end(), false);
    for (int i = 1; i <= numX - 2; i++)
      for (int j = 1; j <= numY - 2; j++)
        for (int k = 1; k <= numZ - 2; k++) {
          int c = index(i, j, k);
          // 如果上一步是刚体，这一步不是刚体，那么用流体填充
          if (old_is_rigid_body[c] && !is_rigid_body[c]) {
            is_container[c] = false;
            filled[c] = false;
            // if(old_filled[index(i-1,j,k)] || old_filled[index(i+1,j,k)] ||
            //    old_filled[index(i,j-1,k)] || old_filled[index(i,j+1,k)] ||
            //    old_filled[index(i,j,k-1)] || old_filled[index(i,j,k+1)]){
            //     filled[c] = true;
            //     //std::cout<<"Filling cell ("<<i<<","<<j<<","<<k<<") with
            //     fluid as rigid body moved away.\n"; danger[c]=true;
            //   }
            //   else{
            //     filled[c] = false;
            //   }
          }
        }
  }
  void initialize() {
    for (int i = 1; i <= numX - 2; i++) {
      for (int j = 1; j <= numY - 2; j++) {
        for (int k = 1; k <= numZ - 2; k++) {
          if (filled[index(i, j, k)] == true) {
            float cx = i * h;
            float cy = j * h;
            float cz = k * h;
            float off = 0.25f * h;
            // push eight sub-cell particle positions (corner offsets)
            x_pos.push_back(cx - off);
            y_pos.push_back(cy - off);
            z_pos.push_back(cz - off);

            x_pos.push_back(cx - off);
            y_pos.push_back(cy - off);
            z_pos.push_back(cz + off);

            x_pos.push_back(cx - off);
            y_pos.push_back(cy + off);
            z_pos.push_back(cz - off);

            x_pos.push_back(cx - off);
            y_pos.push_back(cy + off);
            z_pos.push_back(cz + off);

            x_pos.push_back(cx + off);
            y_pos.push_back(cy - off);
            z_pos.push_back(cz - off);

            x_pos.push_back(cx + off);
            y_pos.push_back(cy - off);
            z_pos.push_back(cz + off);

            x_pos.push_back(cx + off);
            y_pos.push_back(cy + off);
            z_pos.push_back(cz - off);

            x_pos.push_back(cx + off);
            y_pos.push_back(cy + off);
            z_pos.push_back(cz + off);
          }
        }
      }
    }
  }

  void integrate(float dt, float gravity_x, float gravity_y, float gravity_z) {
    for (int i = 1; i <= numX - 3; ++i) {
      for (int j = 1; j <= numY - 2; ++j) {
        for (int k = 1; k <= numZ - 2; ++k) {
          int c = index(i, j, k);
          u[c] += gravity_x * dt;
        }
      }
    }

    for (int i = 1; i <= numX - 2; ++i) {
      for (int j = 1; j <= numY - 3; ++j) {
        for (int k = 1; k <= numZ - 2; ++k) {
          int c = index(i, j, k);
          v[c] += gravity_y * dt;
        }
      }
    }

    for (int i = 1; i <= numX - 2; ++i) {
      for (int j = 1; j <= numY - 2; ++j) {
        for (int k = 1; k <= numZ - 3; ++k) {
          int c = index(i, j, k);
          w[c] += gravity_z * dt;
        }
      }
    }
  }

  void calculate_extended_velocity() {  // Extrapolate from existing filled
                                        // cells using BFS
    // Propagate u, v, w values outward from filled cells on their
    // respective staggered grids using BFS in 3D.

    // --- extendedu: u-grid positions are i=[0..numX-2], j=[1..numY-2],
    // k=[1..numZ-2]
    {
      std::vector<char> seen(numCells, 0);
      std::queue<int> q;

      // Seed from filled cell centers: for each filled cell (ci,cj,ck), seed
      // u at ui = ci-1 and ui = ci (if in bounds) at same cj,ck.
      for (int ci = 1; ci <= numX - 2; ++ci) {
        for (int cj = 1; cj <= numY - 2; ++cj) {
          for (int ck = 1; ck <= numZ - 2; ++ck) {
            if (!filled[index(ci, cj, ck)]) continue;
            for (int ui = ci - 1; ui <= ci; ++ui) {
              if (ui < 0 || ui > numX - 2) continue;
              int uidx = index(ui, cj, ck);
              if (!seen[uidx]) {
                seen[uidx] = 1;
                extendedu[uidx] = u[uidx];
                q.push(uidx);
              }
            }
          }
        }
      }

      // BFS propagate to 6 neighbors on u-grid
      while (!q.empty()) {
        int cur = q.front();
        q.pop();
        int ci = cur % numX;
        int cj = (cur / numX) % numY;
        int ck = cur / (numX * numY);
        const int di[6] = {-1, 1, 0, 0, 0, 0};
        const int dj[6] = {0, 0, -1, 1, 0, 0};
        const int dk[6] = {0, 0, 0, 0, -1, 1};
        for (int m = 0; m < 6; ++m) {
          int ni = ci + di[m];
          int nj = cj + dj[m];
          int nk = ck + dk[m];
          if (ni < 0 || ni > numX - 2) continue;
          if (nj < 1 || nj > numY - 2) continue;
          if (nk < 1 || nk > numZ - 2) continue;
          int nidx = index(ni, nj, nk);
          if (seen[nidx]) continue;
          extendedu[nidx] = extendedu[cur];
          seen[nidx] = 1;
          q.push(nidx);
        }
      }
    }

    // --- extendedv: v-grid positions are i=[1..numX-2], j=[0..numY-2],
    // k=[1..numZ-2]
    {
      std::vector<char> seenV(numCells, 0);
      std::queue<int> qv;

      for (int ci = 1; ci <= numX - 2; ++ci) {
        for (int cj = 1; cj <= numY - 2; ++cj) {
          for (int ck = 1; ck <= numZ - 2; ++ck) {
            if (!filled[index(ci, cj, ck)]) continue;
            for (int vj = cj - 1; vj <= cj; ++vj) {
              if (vj < 0 || vj > numY - 2) continue;
              int vidx = index(ci, vj, ck);
              if (!seenV[vidx]) {
                seenV[vidx] = 1;
                extendedv[vidx] = v[vidx];
                qv.push(vidx);
              }
            }
          }
        }
      }

      // BFS propagate to 6 neighbors on v-grid
      while (!qv.empty()) {
        int cur = qv.front();
        qv.pop();
        int ci = cur % numX;
        int cj = (cur / numX) % numY;
        int ck = cur / (numX * numY);
        const int di[6] = {-1, 1, 0, 0, 0, 0};
        const int dj[6] = {0, 0, -1, 1, 0, 0};
        const int dk[6] = {0, 0, 0, 0, -1, 1};
        for (int m = 0; m < 6; ++m) {
          int ni = ci + di[m];
          int nj = cj + dj[m];
          int nk = ck + dk[m];
          if (ni < 1 || ni > numX - 2) continue;
          if (nj < 0 || nj > numY - 2) continue;
          if (nk < 1 || nk > numZ - 2) continue;
          int nidx = index(ni, nj, nk);
          if (seenV[nidx]) continue;
          extendedv[nidx] = extendedv[cur];
          seenV[nidx] = 1;
          qv.push(nidx);
        }
      }
    }

    // --- extendedw: w-grid positions are i=[1..numX-2], j=[1..numY-2],
    // k=[0..numZ-2]
    {
      std::vector<char> seenW(numCells, 0);
      std::queue<int> qw;

      for (int ci = 1; ci <= numX - 2; ++ci) {
        for (int cj = 1; cj <= numY - 2; ++cj) {
          for (int ck = 1; ck <= numZ - 2; ++ck) {
            if (!filled[index(ci, cj, ck)]) continue;
            for (int wk = ck - 1; wk <= ck; ++wk) {
              if (wk < 0 || wk > numZ - 2) continue;
              int widx = index(ci, cj, wk);
              if (!seenW[widx]) {
                seenW[widx] = 1;
                extendedw[widx] = w[widx];
                qw.push(widx);
              }
            }
          }
        }
      }

      // BFS propagate to 6 neighbors on w-grid
      while (!qw.empty()) {
        int cur = qw.front();
        qw.pop();
        int ci = cur % numX;
        int cj = (cur / numX) % numY;
        int ck = cur / (numX * numY);
        const int di[6] = {-1, 1, 0, 0, 0, 0};
        const int dj[6] = {0, 0, -1, 1, 0, 0};
        const int dk[6] = {0, 0, 0, 0, -1, 1};
        for (int m = 0; m < 6; ++m) {
          int ni = ci + di[m];
          int nj = cj + dj[m];
          int nk = ck + dk[m];
          if (ni < 1 || ni > numX - 2) continue;
          if (nj < 1 || nj > numY - 2) continue;
          if (nk < 0 || nk > numZ - 2) continue;
          int nidx = index(ni, nj, nk);
          if (seenW[nidx]) continue;
          extendedw[nidx] = extendedw[cur];
          seenW[nidx] = 1;
          qw.push(nidx);
        }
      }
    }
  }

  std::vector<float> speed_at(float x, float y, float z) {
    //[1,2]就是(h,2h)
    float u_val = sampleU_e(x / h - 0.5, y / h, z / h);
    float v_val = sampleV_e(x / h, y / h - 0.5, z / h);
    float w_val = sampleW_e(x / h, y / h, z / h - 0.5);
    std::vector<float> res;
    res.push_back(u_val);
    res.push_back(v_val);
    res.push_back(w_val);
    return res;
  }

  // 3D version: returns {i,j,k} clamped to interior cell indices
  // (i,j,k)的中心就是(ih,jh,kh)
  std::vector<int> getCellIndices(float x, float y, float z) {
    x /= h;
    y /= h;
    z /= h;
    int closest_to_x = static_cast<int>(std::round(x));
    int closest_to_y = static_cast<int>(std::round(y));
    int closest_to_z = static_cast<int>(std::round(z));
    int i = std::max(1, std::min(closest_to_x, numX - 2));
    int j = std::max(1, std::min(closest_to_y, numY - 2));
    int k = std::max(1, std::min(closest_to_z, numZ - 2));
    return {i, j, k};
  }
  std::vector<float> marker_particle_position_step(
      float x, float y, float z,
      float dt) {  // 原来的position_after_dt(float x, float y, float z, float
                   // dt) {
    float x_, y_, z_;
    std::vector<float> val = speed_at(x, y, z);
    x_ = x + val[0] * dt;
    y_ = y + val[1] * dt;
    z_ = z + val[2] * dt;
    std::vector<float> res;
    res.push_back(x_);
    res.push_back(y_);
    res.push_back(z_);
    return res;
  }
  // std::vector<float> position_after_dt(float x, float y, float z, float dt) {
  //   std::vector<float> mid_pos = marker_particle_position_step(x, y, z, dt *
  //   0.5f); std::vector<float> end_pos =
  //       marker_particle_position_step(mid_pos[0], mid_pos[1], mid_pos[2],
  //       dt);
  //   return end_pos;
  // }
  bool push_backwards(float &x, float &y, float &z) {
    if (x < 0.5 * h) {
      x = h - x;
    }
    if (x > (numX - 1.5) * h) {
      x = 2 * (numX - 1.5) * h - x;
    }
    if (y < 0.5 * h) {
      y = h - y;
    }
    if (y > (numY - 1.5) * h) {
      y = 2 * (numY - 1.5) * h - y;
    }
    if (z < 0.5 * h) {
      z = h - z;
    }
    if (z > (numZ - 1.5) * h) {
      z = 2 * (numZ - 1.5) * h - z;
    }
    return true;
  }
  // 更新marker_particles的位置
  void update_particles(float dt) {
    for (size_t idx = 0; idx < x_pos.size(); idx++) {
      float tmp1, tmp2, tmp3;
      tmp1 = x_pos[idx];
      tmp2 = y_pos[idx];
      tmp3 = z_pos[idx];
      std::vector<float> pos =
          marker_particle_position_step(x_pos[idx], y_pos[idx], z_pos[idx], dt);
      x_pos[idx] = pos[0];
      y_pos[idx] = pos[1];
      z_pos[idx] = pos[2];
      // 回弹粒子
      push_backwards(x_pos[idx], y_pos[idx], z_pos[idx]);
      // 如果在is_container的格子里面，直接用tmp1,2,3
      std::vector<int> cell_idx =
          getCellIndices(x_pos[idx], y_pos[idx], z_pos[idx]);
      if (is_container[index(cell_idx[0], cell_idx[1], cell_idx[2])] == true) {
        x_pos[idx] = tmp1;
        y_pos[idx] = tmp2;
        z_pos[idx] = tmp3;
      }
    }
    // std::cout<<x_pos[584884]<<" "<<y_pos[584884]<<" "<<z_pos[584884]<<"\n";
    return;
  }

  void check_filled(bool need_print) {
    old_filled = filled;
    for (int i = 0; i < numCells; i++) statistic[i] = 0;
    for (size_t idx = 0; idx < x_pos.size(); idx++) {
      std::vector<int> pos = getCellIndices(x_pos[idx], y_pos[idx], z_pos[idx]);
      statistic[index(pos[0], pos[1], pos[2])] = idx + 1;
    }
    for (int k = numZ - 2; k >= 1; --k) {
      if (need_print) std::cout << "slice k=" << k << std::endl;
      for (int j = numY - 2; j >= 1; --j) {
        for (int i = 1; i <= numX - 2; ++i) {
          if (need_print) std::cout << statistic[index(i, j, k)] << " ";
          filled[index(i, j, k)] = (statistic[index(i, j, k)] >= 1);
        }
        if (need_print) std::cout << std::endl;
      }
    }

    for (int i = 1; i <= numX - 2; i++)
      for (int j = 1; j <= numY - 2; j++)
        for (int k = 1; k <= numZ - 2; k++) {
          if (is_container[index(i, j, k)] == true) {
            filled[index(i, j, k)] = false;
          }
        }
  }

  void after_check_filled() {
    u = extendedu;
    v = extendedv;
    w = extendedw;
    force_zero(0);
    return;
  }

  //////////////////////////////////////////////////////////////////////////////////
  // 必须有numX=numY=numZ
  int find_index_2(int i, int j, int k) {  // 用于J的传入，传入J时就把格式弄好
    return (i - 1) * (numX - 2) * (numX - 2) + (j - 1) * (numX - 2) + k - 1;
  }
  void Project(bool need_print, float dx, float dy, float dz, float dt) {
    pressure_solver P;
    std::vector<double> p_sol;
    int gridtype[345][345][345];  // 1表示正常，2表示固体，0表示空气
    float J[10][23455678];
    // initialize J to zero to avoid using uninitialized garbage values
    std::fill(&J[0][0], &J[0][0] + 10 * 23455678, 0.0f);
    float rc_x, rc_y, rc_z;  // rigid_center
    float rigid_v[10] = {0.0f};

    // if an external rigid body is attached, sync its state into local
    // variables
    if (m_rigid) {
      auto pos = m_rigid->position();
      rc_x = static_cast<float>(pos.x);
      rc_y = static_cast<float>(pos.y);
      rc_z = static_cast<float>(pos.z);
      auto lin = m_rigid->linearVelocity();
      auto ang = m_rigid->angularVelocity();
      // note: code elsewhere expects rigid_v indexed 1..6
      rigid_v[1] = static_cast<float>(lin.x);
      rigid_v[2] = static_cast<float>(lin.y);
      rigid_v[3] = static_cast<float>(lin.z);
      rigid_v[4] = static_cast<float>(ang.x);
      rigid_v[5] = static_cast<float>(ang.y);
      rigid_v[6] = static_cast<float>(ang.z);
    }
    // 用rigid_m表示广义质量
    float rigid_m[10] = {0.0f};
    if (m_rigid) {
      rigid_m[1] = static_cast<float>(m_rigid->mass());
      rigid_m[2] = static_cast<float>(m_rigid->mass());
      rigid_m[3] = static_cast<float>(m_rigid->mass());
      auto I = m_rigid->m_inertiaBody;
      rigid_m[4] = static_cast<float>(I(0, 0));
      rigid_m[5] = static_cast<float>(I(1, 1));
      rigid_m[6] = static_cast<float>(I(2, 2));
    }

    float fc_x, fc_y, fc_z;  // face_center
    int d_n_dim = (numX - 2) * (numY - 2) * (numZ - 2);
    std::vector<double> d((numX - 2) * (numY - 2) * (numZ - 2), 0.0);
    // 我们先计算\nabla \cdot (u,v)
    // 然后求出标量场p（已知\nabla^2 p=(u,v)的散度）
    // 然后把(u,v)减去\nabla p使得速度的散度是0
    std::vector<float> divergence(numCells, 0.0f);
    // Calculate divergence
    for (int j = 1; j < numY - 1; j++) {
      for (int i = 1; i < numX - 1; i++) {
        for (int k = 1; k < numZ - 1; k++) {
          int idx = index(i, j, k);
          float du_dx = (u[index(i, j, k)] - u[index(i - 1, j, k)]) / (h);
          float dv_dy = (v[index(i, j, k)] - v[index(i, j - 1, k)]) / (h);
          float dw_dz = (w[index(i, j, k)] - w[index(i, j, k - 1)]) / (h);
          divergence[idx] = du_dx + dv_dy + dw_dz;  // 右手边的式子
        }
      }
    }
    // check_divergence();

    for (int j = 1; j < numY - 1; j++) {
      for (int i = 1; i < numX - 1; i++) {
        for (int k = 1; k < numZ - 1; k++) {
          int idx = find_index_2(i, j, k);       //(i-1)+(j-1)*(numX-2);
          d[idx] = -divergence[index(i, j, k)];  // 注意有个负号！
        }
      }
    }

    // 求出J的过程
    for (int i = 1; i <= numX - 3; i++)
      for (int j = 1; j <= numY - 2; j++)
        for (int k = 1; k <= numZ - 2; k++)  // 右边的那个格子是刚体
          if (is_rigid_body[index(i, j, k)] == false &&
              filled[index(i, j, k)] == true &&
              is_rigid_body[index(i + 1, j, k)] == true) {
            int j_index = find_index_2(i, j, k);
            fc_y = j * h;
            fc_z = k * h;
            fc_x = i * h + 0.5 * h;
            // J_{1i}+=A(-1),J_{4i,5i,6i}+=A(x_1,x_2,x_3)\times
            // (-1,0,0)=A(0,-x_3,+x_2)
            J[1][j_index] += h * h * (-1);
            J[4][j_index] += h * h * (0);
            J[5][j_index] += h * h * (-fc_z + rc_z);
            // std::cout<<fc_z<<" "<<rc_z<<"\n";
            J[6][find_index_2(i, j, k)] += h * h * (fc_y - rc_y);
          }
    for (int i = 2; i <= numX - 2; i++)
      for (int j = 1; j <= numY - 2; j++)
        for (int k = 1; k <= numZ - 2; k++)  // 左边的那个格子是刚体
          if (is_rigid_body[index(i, j, k)] == false &&
              filled[index(i, j, k)] == true &&
              is_rigid_body[index(i - 1, j, k)] == true) {
            int j_index = find_index_2(i, j, k);
            fc_y = j * h;
            fc_z = k * h;
            fc_x = i * h - 0.5 * h;
            J[1][j_index] += h * h * (1);
            J[5][j_index] += h * h * (fc_z - rc_z);
            J[6][j_index] += h * h * (-fc_y + rc_y);
          }
    for (int i = 1; i <= numX - 2; i++)
      for (int j = 1; j <= numY - 3; j++)
        for (int k = 1; k <= numZ - 2; k++)  // 右边的那个格子是刚体
          if (is_rigid_body[index(i, j, k)] == false &&
              filled[index(i, j, k)] == true &&
              is_rigid_body[index(i, j + 1, k)] == true) {
            int j_index = find_index_2(i, j, k);
            fc_y = j * h + 0.5 * h;
            fc_z = k * h;
            fc_x = i * h;
            J[2][j_index] += h * h * (-1);
            J[4][j_index] += h * h * (fc_z - rc_z);
            J[6][j_index] -= h * h * (fc_x - rc_x);
          }
    for (int i = 1; i <= numX - 2; i++)
      for (int j = 2; j <= numY - 2; j++)
        for (int k = 1; k <= numZ - 2; k++)  // 右边的那个格子是刚体
          if (is_rigid_body[index(i, j, k)] == false &&
              filled[index(i, j, k)] == true &&
              is_rigid_body[index(i, j - 1, k)] == true) {
            int j_index = find_index_2(i, j, k);
            fc_y = j * h - 0.5 * h;
            fc_z = k * h;
            fc_x = i * h;
            J[2][j_index] += h * h;
            J[4][j_index] -= h * h * (fc_z - rc_z);
            J[6][j_index] += h * h * (fc_x - rc_x);
          }
    for (int i = 1; i <= numX - 2; i++)
      for (int j = 1; j <= numY - 2; j++)
        for (int k = 1; k <= numZ - 3; k++)  // 右边的那个格子是刚体
          if (is_rigid_body[index(i, j, k)] == false &&
              filled[index(i, j, k)] == true &&
              is_rigid_body[index(i, j, k + 1)] == true) {
            int j_index = find_index_2(i, j, k);
            fc_y = j * h;
            fc_z = k * h + 0.5 * h;
            fc_x = i * h;
            J[3][j_index] += h * h * (-1);
            J[4][j_index] -= h * h * (fc_y - rc_y);
            J[5][j_index] += h * h * (fc_x - rc_x);
          }
    for (int i = 1; i <= numX - 2; i++)
      for (int j = 1; j <= numY - 2; j++)    // 49.7713
        for (int k = 2; k <= numZ - 2; k++)  // 右边的那个格子是刚体
          if (is_rigid_body[index(i, j, k)] == false &&
              filled[index(i, j, k)] == true &&
              is_rigid_body[index(i, j, k - 1)] == true) {
            int j_index = find_index_2(i, j, k);
            fc_y = j * h;
            fc_z = k * h - 0.5 * h;
            fc_x = i * h;
            J[3][j_index] += h * h;
            J[4][j_index] += h * h * (fc_y - rc_y);
            J[5][j_index] -= h * h * (fc_x - rc_x);
          }
    // 把J的每一项都乘以0.5
    for (size_t i = 1; i <= 6; ++i) {         // Iterate over rows of J
      for (size_t j = 0; j < d_n_dim; ++j) {  // Iterate over columns of J
        J[i][j] *= -0.5f;
      }
    }

    // 左边加上J的东西，右边什么都不加，63.156→54.3374
    // 左边加上0.5J的东西，右边什么都不加，70.1317→48.7201

    // 散度d还需要减去density*J^T*V_rigid，最终我们得到的【散度】是实际散度-\rho
    // J^T V
    for (size_t i = 1; i <= 6; ++i) {         // Iterate over rows of J
      for (size_t j = 0; j < d_n_dim; ++j) {  // Iterate over columns of J
        d[j] -= density * J[i][j] * rigid_v[i];
      }
    }

    for (int i = 0; i < numX; i++)
      for (int j = 0; j < numY; j++)
        for (int k = 0; k < numZ; k++) {
          gridtype[i][j][k] = 2;
        }

    for (int i = 1; i <= numX - 2; i++)
      for (int j = 1; j <= numY - 2; j++)
        for (int k = 1; k <= numZ - 2; k++) {
          if (is_container[index(i, j, k)])
            gridtype[i][j][k] = 2;  // 如果是固体或者容器壁就是2
          else
            gridtype[i][j][k] =
                filled[index(i, j, k)];  // 如果是空气就是0，液体就是1
        }

    double constant_B;
    constant_B = density * density * h * h * h;
    bool success = P.solve(need_print, gridtype, numX - 2, d, p_sol, J,
                           constant_B, rigid_m);
    // 得到的p_sol的下标的格式还是projection_solver中的格式
    if (!success) {
      std::cout << "Pressure solve failed!" << std::endl;
      return;
    }
    // If a rigid body is attached, compute generalized impulse from J and p_sol
    // and apply it
    if (m_rigid) {
      float impulse[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      for (int i = 1; i <= 6; ++i) {  // 第i行的值
        double acc = 0.0;
        for (int j = 0; j < d_n_dim; ++j) {
          double delta_acc = static_cast<double>(J[i][j]) * p_sol[j] * density *
                             dx * dx * dx / dt;
          acc += delta_acc;  // 所谓的p_sol还不是真实的p
          // if(i==2 && J[i][j]!=0) std::cout<<J[i][j]<<"*"<<p_sol[j] * density
          // * dx * dx * dx / dt<<"="<<delta_acc<<"\n";
        }
        impulse[i - 1] = static_cast<float>(acc) * 2;  // 这里乘了一个2
      }
      // 上面的是Jp，是力
      // 把重力也作为冲量加到impulse[1]上
      std::cout << "Original Force Jp=" << impulse[1]
                << "\n";  // 流体对刚体的作用力的y分量
      impulse[1] += m_rigid->mass() * -8.0;  // 这个也是力

      std::cout << "Gravity Added: " << m_rigid->mass() * -8.0 << "\n";
      std::cout << impulse[0] << " " << impulse[1] << " " << impulse[2] << " "
                << impulse[3] << " " << impulse[4] << " " << impulse[5] << "\n";
      change3d::Vec3 rigidPosBeforeUpdate = m_rigid->position();
      std::cout << "Rigid Body Position Before Update: ("
                << rigidPosBeforeUpdate.x << ", " << rigidPosBeforeUpdate.y
                << ", " << rigidPosBeforeUpdate.z << ")\n";
      // 输出刚体当前旋转（以 Roll, Pitch, Yaw 角表示，单位为度）
      {
        auto q = m_rigid->orientation();
        double ww = q.w, xx = q.x, yy = q.y, zz = q.z;
        double roll = std::atan2(2.0 * (ww * xx + yy * zz),
                                 1.0 - 2.0 * (xx * xx + yy * yy));
        double pitch =
            std::asin(std::max(-1.0, std::min(1.0, 2.0 * (ww * yy - zz * xx))));
        double yaw = std::atan2(2.0 * (ww * zz + xx * yy),
                                1.0 - 2.0 * (yy * yy + zz * zz));
        double deg = 180.0 / std::acos(-1.0);
        std::cout << "Rigid Body Orientation (deg) Roll,Pitch,Yaw: "
                  << roll * deg << ", " << pitch * deg << ", " << yaw * deg
                  << "\n";
      }
      m_rigid->update(impulse,
                      dt);  // 更新刚体！！！！！！！！！！

      change3d::Vec3 rigidPosAfterUpdate = m_rigid->position();
      std::cout << "Rigid Body Position After Update: ("
                << rigidPosAfterUpdate.x << ", " << rigidPosAfterUpdate.y
                << ", " << rigidPosAfterUpdate.z << ")\n";
      change3d::Vec3 rigidVelAfterUpdate = m_rigid->linearVelocity();
      std::cout << "Rigid Body Linear Velocity After Update: ("
                << rigidVelAfterUpdate.x << ", " << rigidVelAfterUpdate.y
                << ", " << rigidVelAfterUpdate.z << ")\n";
    }  // end if

    for (int i = 0; i < numCells; i++) p[i] = 0.0f;

    for (int i = 1; i < numX - 1; i++) {
      for (int j = 1; j < numY - 1; j++) {
        for (int k = 1; k < numZ - 1; k++) {
          int idx = (i - 1) * (numZ - 2) * (numY - 2) + (j - 1) * (numZ - 2) +
                    k - 1;  //(i-1)+(j-1)*(numX-2);
          p[index(i, j, k)] = p_sol[idx] * density * dx * dx * dx /
                              dt;  // density=1.0f×1×1×1/0.1
        }  // 答案需要乘以\rho*dx^3 / dt
      }
    }

    // Subtract gradient of pressure from velocity field
    for (int k = 1; k <= numZ - 2; k++)
      for (int j = 1; j <= numY - 2; j++)
        for (int i = 1; i <= numX - 3; i++) {
          u[index(i, j, k)] -= (dt / density / h) *
                               (p[index(i + 1, j, k)] -
                                p[index(i, j, k)]);  // 多除了一个h，警钟长鸣
        }
    for (int k = 1; k <= numZ - 2; k++)
      for (int j = 1; j <= numY - 3; j++)
        for (int i = 1; i <= numX - 2; i++) {
          v[index(i, j, k)] -=
              (dt / density / h) * (p[index(i, j + 1, k)] - p[index(i, j, k)]);
        }
    for (int k = 1; k <= numZ - 3; k++)
      for (int j = 1; j <= numY - 2; j++)
        for (int i = 1; i <= numX - 2; i++) {
          w[index(i, j, k)] -=
              (dt / density / h) * (p[index(i, j, k + 1)] - p[index(i, j, k)]);
        }
  }

  void check_divergence() {
    std::cout << "Not implemented\n";
    // std::vector<float> divergence(numCells, 0.0f);
    // for (int j = 1; j < numY - 1; j++) {
    //   for (int i = 1; i < numX - 1; i++) {
    //     int idx = i + j * numX;
    //     float du_dx = (u[index(i, j)] - u[index(i - 1, j)]) / (h);
    //     float dv_dy = (v[index(i, j)] - v[index(i, j - 1)]) / (h);
    //     divergence[idx] = du_dx + dv_dy;
    //   }
    // }
    // for (int i = 1; i < numX - 1; i++) {
    //   for (int j = 1; j < numY - 1; j++) {
    //     // if(abs(divergence[index(i,j)])>1e-5f)
    //     std::cout << divergence[index(i, j)] << " ";
    //     // else std::cout<<"0 ";
    //   }
    //   std::cout << std::endl;
    // }
  }
  void extrapolate() {
    // Extrapolate u along y-boundaries for all z slices
    for (int i = 0; i < numX; ++i) {
      for (int k = 0; k < numZ; ++k) {
        u[index(i, 0, k)] = u[index(i, 1, k)];
        u[index(i, numY - 1, k)] = u[index(i, numY - 2, k)];
      }
    }

    // Extrapolate v along x-boundaries for all z slices
    for (int j = 0; j < numY; ++j) {
      for (int k = 0; k < numZ; ++k) {
        v[index(0, j, k)] = v[index(1, j, k)];
        v[index(numX - 1, j, k)] = v[index(numX - 2, j, k)];
      }
    }

    // Extrapolate w along z-boundaries for all x,y positions
    for (int i = 0; i < numX; ++i) {
      for (int j = 0; j < numY; ++j) {
        w[index(i, j, 0)] = w[index(i, j, 1)];
        w[index(i, j, numZ - 1)] = w[index(i, j, numZ - 2)];
      }
    }
  }

  // 3D trilinear sampling for extended u-field. The third coordinate `k`
  // corresponds to the z-index in grid units. A default `k` is provided
  // so existing 2-arg calls remain valid (they sample the middle-ish slice).
  float sampleU_e(float i, float j, float k = 1.0f) {
    // u ∈[0,numX-2] * [1,numY-2] * [1,numZ-2]
    if (i < 0.0f) i = 0.0f;
    if (i > numX - 2.0f) i = numX - 2.0f;
    if (j < 1.0f) j = 1.0f;
    if (j > numY - 2.0f) j = numY - 2.0f;
    if (k < 1.0f) k = 1.0f;
    if (k > numZ - 2.0f) k = numZ - 2.0f;

    float i_r = i - std::floor(i);
    float j_r = j - std::floor(j);
    float k_r = k - std::floor(k);
    int i0 = static_cast<int>(std::floor(i));
    int j0 = static_cast<int>(std::floor(j));
    int k0 = static_cast<int>(std::floor(k));
    int i1 = std::min(i0 + 1, numX - 2);
    int j1 = std::min(j0 + 1, numY - 2);
    int k1 = std::min(k0 + 1, numZ - 2);

    float u000 = extendedu[index(i0, j0, k0)];
    float u100 = extendedu[index(i1, j0, k0)];
    float u010 = extendedu[index(i0, j1, k0)];
    float u110 = extendedu[index(i1, j1, k0)];

    float u001 = extendedu[index(i0, j0, k1)];
    float u101 = extendedu[index(i1, j0, k1)];
    float u011 = extendedu[index(i0, j1, k1)];
    float u111 = extendedu[index(i1, j1, k1)];

    // Interpolate along i
    float c00 = u000 * (1 - i_r) + u100 * i_r;
    float c10 = u010 * (1 - i_r) + u110 * i_r;
    float c01 = u001 * (1 - i_r) + u101 * i_r;
    float c11 = u011 * (1 - i_r) + u111 * i_r;

    // Interpolate along j
    float c0 = c00 * (1 - j_r) + c10 * j_r;
    float c1 = c01 * (1 - j_r) + c11 * j_r;

    // Interpolate along k and return
    return c0 * (1 - k_r) + c1 * k_r;
  }

  float sampleW_e(float i, float j, float k = 1.0f) {
    // w ∈[1,numX-2] * [1,numY-2] * [0,numZ-2]
    if (i < 1.0f) i = 1.0f;
    if (i > numX - 2.0f) i = numX - 2.0f;
    if (j < 1.0f) j = 1.0f;
    if (j > numY - 2.0f) j = numY - 2.0f;
    if (k < 0.0f) k = 0.0f;
    if (k > numZ - 2.0f) k = numZ - 2.0f;

    float i_r = i - std::floor(i);
    float j_r = j - std::floor(j);
    float k_r = k - std::floor(k);
    int i0 = static_cast<int>(std::floor(i));
    int j0 = static_cast<int>(std::floor(j));
    int k0 = static_cast<int>(std::floor(k));
    int i1 = std::min(i0 + 1, numX - 2);
    int j1 = std::min(j0 + 1, numY - 2);
    int k1 = std::min(k0 + 1, numZ - 2);

    float w000 = extendedw[index(i0, j0, k0)];
    float w100 = extendedw[index(i1, j0, k0)];
    float w010 = extendedw[index(i0, j1, k0)];
    float w110 = extendedw[index(i1, j1, k0)];

    float w001 = extendedw[index(i0, j0, k1)];
    float w101 = extendedw[index(i1, j0, k1)];
    float w011 = extendedw[index(i0, j1, k1)];
    float w111 = extendedw[index(i1, j1, k1)];

    // Interpolate along i
    float c00 = w000 * (1 - i_r) + w100 * i_r;
    float c10 = w010 * (1 - i_r) + w110 * i_r;
    float c01 = w001 * (1 - i_r) + w101 * i_r;
    float c11 = w011 * (1 - i_r) + w111 * i_r;

    // Interpolate along j
    float c0 = c00 * (1 - j_r) + c10 * j_r;
    float c1 = c01 * (1 - j_r) + c11 * j_r;

    // Interpolate along k and return
    return c0 * (1 - k_r) + c1 * k_r;
  }

  float sampleV_e(float i, float j, float k = 1.0f) {
    // v ∈[1,numX-2] * [0,numY-2] * [1,numZ-2]
    if (j < 0.0f) j = 0.0f;
    if (j > numY - 2.0f) j = numY - 2.0f;
    if (i < 1.0f) i = 1.0f;
    if (i > numX - 2.0f) i = numX - 2.0f;
    if (k < 1.0f) k = 1.0f;
    if (k > numZ - 2.0f) k = numZ - 2.0f;

    float i_r = i - std::floor(i);
    float j_r = j - std::floor(j);
    float k_r = k - std::floor(k);
    int i0 = static_cast<int>(std::floor(i));
    int j0 = static_cast<int>(std::floor(j));
    int k0 = static_cast<int>(std::floor(k));
    int i1 = std::min(i0 + 1, numX - 2);
    int j1 = std::min(j0 + 1, numY - 2);
    int k1 = std::min(k0 + 1, numZ - 2);

    float v000 = extendedv[index(i0, j0, k0)];
    float v100 = extendedv[index(i1, j0, k0)];
    float v010 = extendedv[index(i0, j1, k0)];
    float v110 = extendedv[index(i1, j1, k0)];

    float v001 = extendedv[index(i0, j0, k1)];
    float v101 = extendedv[index(i1, j0, k1)];
    float v011 = extendedv[index(i0, j1, k1)];
    float v111 = extendedv[index(i1, j1, k1)];

    // Interpolate along i
    float c00 = v000 * (1 - i_r) + v100 * i_r;
    float c10 = v010 * (1 - i_r) + v110 * i_r;
    float c01 = v001 * (1 - i_r) + v101 * i_r;
    float c11 = v011 * (1 - i_r) + v111 * i_r;

    // Interpolate along j
    float c0 = c00 * (1 - j_r) + c10 * j_r;
    float c1 = c01 * (1 - j_r) + c11 * j_r;

    // Interpolate along k and return
    return c0 * (1 - k_r) + c1 * k_r;
  }

  // 3D averaged velocities. Provide both 3-arg versions (i,j,k) and
  // 2-arg wrappers for backward compatibility (use middle slice k=1).
  float avgU(int i, int j, int k) const {
    // average u on the 2x2x2 neighborhood in j and k directions
    return (u[index(i - 1, j, k)] + u[index(i - 1, j + 1, k)] +
            u[index(i, j, k)] + u[index(i, j + 1, k)] +
            u[index(i - 1, j, k + 1)] + u[index(i - 1, j + 1, k + 1)] +
            u[index(i, j, k + 1)] + u[index(i, j + 1, k + 1)]) *
           0.125f;
  }

  float avgV(int i, int j, int k) const {
    // average v on the 2x2x2 neighborhood in i and k directions
    return (v[index(i, j - 1, k)] + v[index(i, j, k)] +
            v[index(i + 1, j - 1, k)] + v[index(i + 1, j, k)] +
            v[index(i, j - 1, k + 1)] + v[index(i, j, k + 1)] +
            v[index(i + 1, j - 1, k + 1)] + v[index(i + 1, j, k + 1)]) *
           0.125f;
  }

  // average w on the 2x2x2 neighborhood in i and j directions (k-centered)
  float avgW(int i, int j, int k) const {
    return (w[index(i, j, k - 1)] + w[index(i, j, k)] +
            w[index(i + 1, j, k - 1)] + w[index(i + 1, j, k)] +
            w[index(i, j + 1, k - 1)] + w[index(i, j + 1, k)] +
            w[index(i + 1, j + 1, k - 1)] + w[index(i + 1, j + 1, k)]) *
           0.125f;
  }
  void calculate_is_rigid_body() {
    for (int i = 1; i <= numX - 2; i++) {
      for (int j = 1; j <= numY - 2; j++) {
        for (int k = 1; k <= numZ - 2; k++) {
          float cx, cy, cz;  // center
          cx = i * h, cy = j * h, cz = k * h;
          if (m_rigid) {
            // auto pos = m_rigid->position();
            // float rc_x = static_cast<float>(pos.x);
            // float rc_y = static_cast<float>(pos.y);
            // float rc_z = static_cast<float>(pos.z);
            // float dist = std::sqrt((cx - rc_x) * (cx - rc_x) +
            //                        (cy - rc_y) * (cy - rc_y) +
            //                        (cz - rc_z) * (cz - rc_z));
            if (m_rigid->is_inside(cx, cy, cz)) {
              is_rigid_body[index(i, j, k)] = true;
            } else {
              is_rigid_body[index(i, j, k)] = false;
            }
          }
        }
      }
    }
    
  }
  void force_zero(
      bool need_print) {  // 把固体内部的速度都设为0，边界上的不设为0
    // u:内部的是[1,numX-3]*[1,numY-2]*[1,numZ-2]，边界上的是i=0和i=numX-2
    // 第一个格子对应的u是i=0和i=1的
    for (int j = 1; j <= numY - 2; j++) {
      for (int k = 1; k <= numZ - 2; k++) {
        u[index(0, j, k)] = 0.0f;
        u[index(numX - 2, j, k)] = 0.0f;
      }
    }
    for (int i = 1; i <= numX - 2; i++) {
      for (int k = 1; k <= numZ - 2; k++) {
        v[index(i, 0, k)] = 0.0f;
        v[index(i, numY - 2, k)] = 0.0f;
      }
    }
    for (int i = 1; i <= numX - 2; i++) {
      for (int j = 1; j <= numY - 2; j++) {
        w[index(i, j, 0)] = 0.0f;
        w[index(i, j, numZ - 2)] = 0.0f;
      }
    }
    for (int i = 1; i <= numX - 3; i++) {
      for (int j = 1; j <= numY - 2; j++) {
        for (int k = 1; k <= numZ - 2; k++) {
          if (is_container[index(i, j, k)] ||
              is_container[index(i + 1, j, k)]) {
            // 譬如说（0,j,k）是固体，（1,j,k）是流体，那么u(0,j,k)就应该是0
            u[index(i, j, k)] = 0.0f;
          }
        }
      }
    }
    for (int i = 1; i <= numX - 2; i++) {
      for (int j = 1; j <= numY - 3; j++) {
        for (int k = 1; k <= numZ - 2; k++) {
          if (is_container[index(i, j, k)] ||
              is_container[index(i, j + 1, k)]) {
            // 譬如说（i,0,k）是固体，（i,1,k）是流体，那么v(i,0,k)就应该是0
            v[index(i, j, k)] = 0.0f;
          }
        }
      }
    }
    for (int i = 1; i <= numX - 2; i++) {
      for (int j = 1; j <= numY - 2; j++) {
        for (int k = 1; k <= numZ - 3; k++) {
          if (is_container[index(i, j, k)] ||
              is_container[index(i, j, k + 1)]) {
            // 譬如说（i,j,0）是固体，（i,j,1）是流体，那么w(i,j,0)就应该是0
            w[index(i, j, k)] = 0.0f;
          }
        }
      }
    }
  }

  void advect(float dt) {
    newU = u;
    newV = v;
    newW = w;
    float h2 = 0.5f * h;
    for (int i = 1; i <= numX - 3; i++) {
      for (int j = 1; j <= numY - 2; j++) {
        for (int k = 1; k <= numZ - 2; k++) {
          int c = index(i, j, k);
          float x = i * h + h2;
          float y = j * h;
          float z = k * h;
          float uu = u[c];
          float vv = avgV(i, j, k);
          float ww = avgW(i, j, k);
          x -= dt * uu;
          y -= dt * vv;
          z -= dt * ww;
          newU[c] = sampleU_e(x / h - 0.5f, y / h, z / h);
        }
      }
    }
    for (int i = 1; i <= numX - 2; ++i) {
      for (int j = 1; j <= numY - 3; ++j) {
        for (int k = 1; k <= numZ - 2; ++k) {
          int c = index(i, j, k);
          float x = i * h;
          float y = j * h + h2;
          float z = k * h;
          float uu = avgU(i, j, k);
          float vv = v[c];
          float ww = avgW(i, j, k);
          x -= dt * uu;
          y -= dt * vv;
          z -= dt * ww;
          newV[c] = sampleV_e(x / h, y / h - 0.5f, z / h);
        }
      }
    }

    // Advect W
    for (int i = 1; i <= numX - 2; ++i) {
      for (int j = 1; j <= numY - 2; ++j) {
        for (int k = 1; k <= numZ - 3; ++k) {
          int c = index(i, j, k);
          float x = i * h;
          float y = j * h;
          float z = k * h + h2;
          float uu = avgU(i, j, k);
          float vv = avgV(i, j, k);
          float ww = w[c];
          x -= dt * uu;
          y -= dt * vv;
          z -= dt * ww;
          newW[c] = sampleW_e(x / h, y / h, z / h - 0.5f);
        }
      }
    }
    u.swap(newU);
    v.swap(newV);
    w.swap(newW);
  }
  void print_highest_fluid_block() {
    // Find the highest j (y-index) that contains any fluid cell and print
    int highest_j = -1;
    for (int j = numY - 2; j >= 1; --j) {
      bool found = false;
      for (int i = 1; i <= 49 && !found; ++i) {
        for (int k = 1; k <= 49; ++k) {
          if (filled[index(i, j, k)]) {
            found = true;
            break;
          }
        }
      }
      if (found) {
        highest_j = j;
        break;
      }
    }
    if (highest_j == -1) {
      std::cout << "No fluid cells found\n";
    } else {
      float height = highest_j * h;  // cell center y-coordinate
      std::cout << "Highest fluid block at y = " << height
                << " (grid j=" << highest_j << ")\n";
    }
  }
  void treat_rigid_as_container(bool print = false) {
    int cnt0 = 0;
    for (int i = 1; i <= numX - 2; i++)
      for (int j = 1; j <= numY - 2; j++)
        for (int k = 1; k <= numZ - 2; k++) {
          if (is_rigid_body[index(i, j, k)] == true) {
            if (print) {
              std::cout << i << " " << j << " " << k << " is container    ";
              cnt0++;
            }
            is_container[index(i, j, k)] = true;
            filled[index(i, j, k)] = false;
          }
        }
    if (print) {
      std::cout << "Total rigid body cells treated as container: " << cnt0
                << "\n";
    }
  }
};
