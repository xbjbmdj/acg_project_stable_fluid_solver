#ifndef CHANGE3D_SOLID_H
#define CHANGE3D_SOLID_H

#include <cmath>
#include <algorithm>
#include <array>


namespace change3d {

struct Vec3 {
    double x=0, y=0, z=0;
    Vec3() = default;
    Vec3(double x_, double y_, double z_): x(x_), y(y_), z(z_) {}
    Vec3 operator+(const Vec3& o) const { return Vec3(x+o.x, y+o.y, z+o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x-o.x, y-o.y, z-o.z); }
    Vec3 operator*(double s) const { return Vec3(x*s, y*s, z*s); }
    Vec3 operator/(double s) const { return Vec3(x/s, y/s, z/s); }
    Vec3& operator+=(const Vec3& o){ x+=o.x; y+=o.y; z+=o.z; return *this; }
    Vec3& operator-=(const Vec3& o){ x-=o.x; y-=o.y; z-=o.z; return *this; }
};

inline Vec3 operator*(double s, const Vec3& v){ return v*s; }

inline double dot(const Vec3& a, const Vec3& b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b){
    return Vec3(a.y*b.z - a.z*b.y,
                a.z*b.x - a.x*b.z,
                a.x*b.y - a.y*b.x);
}
inline double norm(const Vec3& v){ return std::sqrt(dot(v,v)); }
inline Vec3 normalize(const Vec3& v){ double n = norm(v); return n>0? v / n : v; }

struct Mat3 {
    // row-major
    std::array<double,9> m{};
    Mat3() { m = {1,0,0, 0,1,0, 0,0,1}; }
    Mat3(double diag){ m = {diag,0,0, 0,diag,0, 0,0,diag}; }
    double& operator()(int r,int c){ return m[r*3+c]; }
    double operator()(int r,int c) const { return m[r*3+c]; }
    Vec3 mulVec(const Vec3& v) const {
        return Vec3(
            m[0]*v.x + m[1]*v.y + m[2]*v.z,
            m[3]*v.x + m[4]*v.y + m[5]*v.z,
            m[6]*v.x + m[7]*v.y + m[8]*v.z);
    }
    Mat3 transpose() const {
        Mat3 r; r.m = {m[0],m[3],m[6], m[1],m[4],m[7], m[2],m[5],m[8]}; return r;
    }
};

struct Quat {
    double w=1, x=0, y=0, z=0;
    Quat() = default;
    Quat(double w_, double x_, double y_, double z_): w(w_), x(x_), y(y_), z(z_) {}
    static Quat fromAxisAngle(const Vec3& axis, double angle){
        Vec3 a = change3d::normalize(axis);
        double s = std::sin(angle*0.5);
        return Quat(std::cos(angle*0.5), a.x*s, a.y*s, a.z*s);
    }
    Quat operator*(const Quat& o) const {
        return Quat(
            w*o.w - x*o.x - y*o.y - z*o.z,
            w*o.x + x*o.w + y*o.z - z*o.y,
            w*o.y - x*o.z + y*o.w + z*o.x,
            w*o.z + x*o.y - y*o.x + z*o.w);
    }
    Quat operator*(double s) const { return Quat(w*s, x*s, y*s, z*s); }
    Quat operator+(const Quat& o) const { return Quat(w+o.w, x+o.x, y+o.y, z+o.z); }
    void normalize(){ double n = std::sqrt(w*w + x*x + y*y + z*z); if(n>0){ w/=n; x/=n; y/=n; z/=n; } }
    Mat3 toMat3() const {
        Mat3 R(0.0);
        double ww=w*w, xx=x*x, yy=y*y, zz=z*z;
        R(0,0) = ww+xx-yy-zz; R(0,1) = 2*(x*y - w*z);   R(0,2) = 2*(x*z + w*y);
        R(1,0) = 2*(x*y + w*z);   R(1,1) = ww-xx+yy-zz; R(1,2) = 2*(y*z - w*x);
        R(2,0) = 2*(x*z - w*y);   R(2,1) = 2*(y*z + w*x); R(2,2) = ww-xx-yy+zz;
        return R;
    }
};

class RigidBody {
public:
    RigidBody();

    // mass / inertia
    void setMass(double m);
    double mass() const { return m_mass; }
    void setInertiaBody(const Mat3& Ibody); // inertia expressed in body space

    // pose & velocities
    void setPosition(const Vec3& p){ m_pos = p; }
    Vec3 position() const { return m_pos; }
    void setOrientation(const Quat& q){ m_ori = q; m_ori.normalize(); }
    Quat orientation() const { return m_ori; }
    void setLinearVelocity(const Vec3& v){ m_linVel = v; }
    Vec3 linearVelocity() const { return m_linVel; }
    void setAngularVelocity(const Vec3& w){ m_angVel = w; }
    Vec3 angularVelocity() const { return m_angVel; }

    // external forces: force in world coords, application point in world coords
    void addForceAtPoint(const Vec3& force, const Vec3& pointWorld);
    void addForce(const Vec3& force); // applies at center
    void clearForces();
    void addTorque(const Vec3& torque);
    void applyForceAndTorque(const Vec3& force, const Vec3& torque);
    void applyForceAndTorqueAndIntegrate(const Vec3& force, const Vec3& torque, double dt);

    // integrate explicit Euler semi-implicit for dt
    void integrate(double dt);

    // apply generalized impulse (or force integrated over dt):
    // impulse[0..2] = linear component, impulse[3..5] = angular component
    // This treats `impulse` as a force if you provide a non-1 `dt`:
    // the applied change is M_S^{-1} * (impulse * dt).
    void update(const float impulse[6], float dt);

    // access accumulated
    Vec3 accumulatedForce() const { return m_force; }
    Vec3 accumulatedTorque() const { return m_torque; }
    Mat3 m_inertiaBody;        // body-space inertia
    Mat3 m_invInertiaBody;     // body-space inverse inertia
    
private:
    double m_mass = 1.0;
    double m_invMass = 1.0;


    Vec3 m_pos;
    Quat m_ori;
    Vec3 m_linVel;
    Vec3 m_angVel; // world space angular velocity

    Vec3 m_force;
    Vec3 m_torque;

    // helper
    Mat3 worldInverseInertia() const {
        Mat3 R = m_ori.toMat3();
        Mat3 Rt = R.transpose();
        // Iw^{-1} = R * Ibody^{-1} * R^T
        // multiply: temp = R * m_invInertiaBody
        Mat3 temp;
        for(int r=0;r<3;++r) for(int c=0;c<3;++c){
            double v=0;
            for(int k=0;k<3;++k) v += R(r,k) * m_invInertiaBody(k,c);
            temp(r,c)=v;
        }
        Mat3 IwInv;
        for(int r=0;r<3;++r) for(int c=0;c<3;++c){
            double v=0;
            for(int k=0;k<3;++k) v += temp(r,k) * Rt(k,c);
            IwInv(r,c)=v;
        }
        return IwInv;
    }
};

// Implementation
inline RigidBody::RigidBody()
    : m_mass(1.0), m_invMass(1.0), m_inertiaBody(1.0), m_invInertiaBody(1.0), m_pos(), m_ori(), m_linVel(), m_angVel(), m_force(), m_torque()
{
}

inline void RigidBody::setMass(double m){
    m_mass = std::max(1e-12, m);
    m_invMass = 1.0 / m_mass;
}

inline void RigidBody::setInertiaBody(const Mat3& Ibody){
    m_inertiaBody = Ibody;
    // naive inverse (for small 3x3 matrix). Use analytic inverse for symmetric positive-definite if needed.
    // Here we compute the inverse by Cramer's rule for generality.
    const double* a = m_inertiaBody.m.data();
    double A00=a[0], A01=a[1], A02=a[2];
    double A10=a[3], A11=a[4], A12=a[5];
    double A20=a[6], A21=a[7], A22=a[8];
    double det = A00*(A11*A22 - A12*A21) - A01*(A10*A22 - A12*A20) + A02*(A10*A21 - A11*A20);
    if (std::abs(det) < 1e-18) {
        // fallback to identity scaled
        m_invInertiaBody = Mat3(1.0);
        return;
    }
    double invdet = 1.0/det;
    Mat3 inv;
    inv(0,0) =  (A11*A22 - A12*A21)*invdet;
    inv(0,1) = -(A01*A22 - A02*A21)*invdet;
    inv(0,2) =  (A01*A12 - A02*A11)*invdet;
    inv(1,0) = -(A10*A22 - A12*A20)*invdet;
    inv(1,1) =  (A00*A22 - A02*A20)*invdet;
    inv(1,2) = -(A00*A12 - A02*A10)*invdet;
    inv(2,0) =  (A10*A21 - A11*A20)*invdet;
    inv(2,1) = -(A00*A21 - A01*A20)*invdet;
    inv(2,2) =  (A00*A11 - A01*A10)*invdet;
    m_invInertiaBody = inv;
}

inline void RigidBody::addForceAtPoint(const Vec3& force, const Vec3& pointWorld){
    m_force += force;
    Vec3 r = pointWorld - m_pos; // lever arm in world
    m_torque += cross(r, force);
}

inline void RigidBody::addForce(const Vec3& force){ m_force += force; }

inline void RigidBody::clearForces(){ m_force = Vec3(); m_torque = Vec3(); }

inline void RigidBody::addTorque(const Vec3& torque){ m_torque += torque; }

inline void RigidBody::applyForceAndTorque(const Vec3& force, const Vec3& torque){
    m_force += force;
    m_torque += torque;
}

inline void RigidBody::applyForceAndTorqueAndIntegrate(const Vec3& force, const Vec3& torque, double dt){
    applyForceAndTorque(force, torque);
    integrate(dt);
}

inline void RigidBody::integrate(double dt){
    if(dt <= 0) return;
    // linear
    Vec3 accel = m_force * m_invMass;
    m_linVel += accel * dt;
    m_pos += m_linVel * dt;

    // angular: compute world-space inverse inertia and use torque
    Mat3 IwInv = worldInverseInertia();
    // angAcc = IwInv * torque
    Vec3 angAcc = IwInv.mulVec(m_torque);
    m_angVel += angAcc * dt;

    // update orientation: q_dot = 0.5 * omega_quat * q
    Quat omegaQ(0.0, m_angVel.x, m_angVel.y, m_angVel.z);
    Quat qdot = omegaQ * m_ori;
    qdot = qdot * 0.5;
    m_ori = m_ori + qdot * dt;
    m_ori.normalize();

    // clear accumulators for next step
    clearForces();
}

inline void RigidBody::update(const float impulse[6], float dt=0.1){
    // scale impulse by dt so callers can pass either a force (with dt) or
    // a true impulse (pass dt=1.0)
    double s = static_cast<double>(dt);

    // linear: use inverse mass
    m_linVel.x += (static_cast<double>(impulse[0]) * s) * m_invMass;
    m_linVel.y += (static_cast<double>(impulse[1]) * s) * m_invMass;
    m_linVel.z += (static_cast<double>(impulse[2]) * s) * m_invMass;

    // angular: assume M_S diagonal -> use diagonal of world inverse inertia
    Mat3 IwInv = worldInverseInertia();
    double invIxx = IwInv(0,0);
    double invIyy = IwInv(1,1);
    double invIzz = IwInv(2,2);
    m_angVel.x += invIxx * (static_cast<double>(impulse[3]) * s);
    m_angVel.y += invIyy * (static_cast<double>(impulse[4]) * s);
    m_angVel.z += invIzz * (static_cast<double>(impulse[5]) * s);

    // integrate pose using updated velocities
    // update position (linear velocity already updated above)
    m_pos += m_linVel * s;

    // update orientation via quaternion kinematics: q_dot = 0.5 * omega_quat * q
    Quat omegaQ(0.0, m_angVel.x, m_angVel.y, m_angVel.z);
    Quat qdot = omegaQ * m_ori;
    qdot = qdot * 0.5;
    m_ori = m_ori + qdot * s;
    m_ori.normalize();
}

} // namespace change3d

#endif // CHANGE3D_SOLID_H
