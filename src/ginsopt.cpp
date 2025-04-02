#include "ginsopt.h"



IMUPreintegrator::IMUPreintegrator() {
    Reset();
}

void IMUPreintegrator::Reset() {
    delta_p_.setZero();
    delta_q_.setIdentity();
    delta_v_.setZero();

    covariance_.setZero();
    dp_dba_.setZero();
    dp_dbg_.setZero();
    dv_dba_.setZero();
    dv_dbg_.setZero();
    dq_dbg_.setZero();

    gyro_bias_.setZero();
    accel_bias_.setZero();
}


void IMUPreintegrator::Integrate(const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro, double dt) {
    // 去除零偏的测量值
    Eigen::Vector3d un_accel = accel - accel_bias_;
    Eigen::Vector3d un_gyro = gyro - gyro_bias_;

    // 更新旋转变化量: Δq_{k+1} = Δq_k ⊗ [0.5*ω*dt, 1]
    Eigen::Vector3d delta_angle = un_gyro * dt;
    Eigen::Quaterniond delta_q_dt;
    double delta_angle_norm = delta_angle.norm();
    if (delta_angle_norm > 1e-12) {
        delta_q_dt = Eigen::Quaterniond(
        cos(delta_angle_norm * 0.5),
        sin(delta_angle_norm * 0.5) * delta_angle.x() / delta_angle_norm,
        sin(delta_angle_norm * 0.5) * delta_angle.y() / delta_angle_norm,
        sin(delta_angle_norm * 0.5) * delta_angle.z() / delta_angle_norm);
    } else {
        delta_q_dt = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
    }
    delta_q_ = delta_q_ * delta_q_dt;
    delta_q_.normalize();

    // 更新速度变化量: Δv_{k+1} = Δv_k + Δq_k * a * dt
    delta_v_ += delta_q_ * un_accel * dt;

    // 更新位置变化量: Δp_{k+1} = Δp_k + Δv_k * dt + 0.5 * Δq_k * a * dt²
    delta_p_ += delta_v_ * dt + 0.5 * (delta_q_ * un_accel) * dt * dt;

    // 协方差传播 (简化版)
    Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Identity();
    Eigen::Matrix<double, 9, 6> G = Eigen::Matrix<double, 9, 6>::Zero();

    // 构建状态转移矩阵F
    F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
    F.block<3, 3>(3, 6) = -delta_q_.toRotationMatrix() * skewSymmetric(un_accel) * dt;
    F.block<3, 3>(6, 3) = -delta_q_.toRotationMatrix() * dt;

    // 构建噪声矩阵G
    G.block<3, 3>(3, 0) = delta_q_.toRotationMatrix() * dt;
    G.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity() * dt;

    // 噪声矩阵Q
    Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
    Q.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * gyro_noise_var_ * dt * dt;
    Q.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * accel_noise_var_ * dt * dt;

    // 协方差传播
    covariance_ = F * covariance_ * F.transpose() + G * Q * G.transpose();

    // 更新雅可比矩阵 (用于零偏校正)
    dp_dba_ += dv_dba_ * dt - 0.5 * delta_q_.toRotationMatrix() * dt * dt;
    dp_dbg_ += dv_dbg_ * dt - 0.5 * delta_q_.toRotationMatrix() * skewSymmetric(un_accel) * dp_dbg_ * dt * dt;
    dv_dba_ += -delta_q_.toRotationMatrix() * dt;
    dv_dbg_ += -delta_q_.toRotationMatrix() * skewSymmetric(un_accel) * dv_dbg_ * dt;
    dq_dbg_ = delta_q_dt.toRotationMatrix().transpose() * dq_dbg_ - rightJacobian(delta_angle) * dt;
}


bool IMUFactor::Evaluate(double const* const* parameters,
                         double* residuals,
                         double** jacobians) const {
    // 解析参数块
    // pose_i: [px, py, pz, qx, qy, qz, qw]
    Eigen::Map<const Eigen::Vector3d> pi(parameters[0]);
    Eigen::Map<const Eigen::Quaterniond> qi(parameters[0] + 3);

    // speed_bias_i: [vx, vy, vz, bgx, bgy, bgz, bax, bay, baz]
    Eigen::Map<const Eigen::Vector3d> vi(parameters[1]);
    Eigen::Map<const Eigen::Vector3d> bgi(parameters[1] + 3);
    Eigen::Map<const Eigen::Vector3d> bai(parameters[1] + 6);

    // pose_j: [px, py, pz, qx, qy, qz, qw]
    Eigen::Map<const Eigen::Vector3d> pj(parameters[2]);
    Eigen::Map<const Eigen::Quaterniond> qj(parameters[2] + 3);

    // speed_bias_j: [vx, vy, vz, bgx, bgy, bgz, bax, bay, baz]
    Eigen::Map<const Eigen::Vector3d> vj(parameters[3]);
    Eigen::Map<const Eigen::Vector3d> bgj(parameters[3] + 3);
    Eigen::Map<const Eigen::Vector3d> baj(parameters[3] + 6);

    // 计算残差
    Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);

    // 位置残差 (3维)
    residual.block<3, 1>(0, 0) = qi.inverse() * (pj - pi - vi * preint_.delta_t() - 0.5 * gravity_ * preint_.delta_t() * preint_.delta_t()) 
                                    - preint_.delta_p();

    // 旋转残差 (3维，使用旋转向量)
    residual.block<3, 1>(3, 0) = 2.0 * (preint_.delta_q().inverse() * (qi.inverse() * qj)).vec();

    // 速度残差 (3维)
    residual.block<3, 1>(6, 0) = 
    qi.inverse() * (vj - vi - gravity_ * preint_.delta_t()) - preint_.delta_v();

    // 零偏残差 (6维)
    residual.block<3, 1>(9, 0) = bgj - bgi;
    residual.block<3, 1>(12, 0) = baj - bai;

    // 计算雅可比（如果请求）
    if (jacobians) {
    // 获取预积分的雅可比
        const Eigen::Matrix3d dp_dbg = preint_.dp_dbg();
        const Eigen::Matrix3d dp_dba = preint_.dp_dba();
        const Eigen::Matrix3d dv_dbg = preint_.dv_dbg();
        const Eigen::Matrix3d dv_dba = preint_.dv_dba();
        const Eigen::Matrix3d dq_dbg = preint_.dq_dbg();

        // 雅可比矩阵布局 [J_pose_i, J_speed_bias_i, J_pose_j, J_speed_bias_j]
        if (jacobians[0]) { // 关于pose_i的雅可比
            Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
            jacobian_pose_i.setZero();

            // 位置部分
            jacobian_pose_i.block<3, 3>(0, 0) = -qi.inverse().toRotationMatrix();
            jacobian_pose_i.block<3, 3>(0, 3) = 
            skewSymmetric(qi.inverse() * (pj - pi - vi * preint_.delta_t() - 0.5 * gravity_ * preint_.delta_t() * preint_.delta_t()));

            // 旋转部分
            Eigen::Matrix3d inv_r = preint_.delta_q().inverse().toRotationMatrix();
            Eigen::Matrix3d jac = -inv_r * qi.inverse().toRotationMatrix();
            jacobian_pose_i.block<3, 3>(3, 3) = jac;

            // 速度部分
            jacobian_pose_i.block<3, 3>(6, 3) = 
            skewSymmetric(qi.inverse() * (vj - vi - gravity_ * preint_.delta_t()));
        }

    // 其他雅可比矩阵类似实现...
    // 实际实现中需要完整实现所有雅可比矩阵
    }

    return true;
}


bool GNSSPosFactor::Evaluate(double const* const* parameters,
                                double* residuals,
                                double** jacobians) const {
    // 解析参数块 [px, py, pz, qx, qy, qz, qw]
    Eigen::Map<const Eigen::Vector3d> p(parameters[0]);
    
    // 计算残差 (直接位置差)
    Eigen::Map<Eigen::Vector3d> residual(residuals);
    residual = (p - pos_) * weight_;
    
    // 计算雅可比（如果请求）
    if (jacobians && jacobians[0]) {
        Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian(jacobians[0]);
        jacobian.setZero();
        jacobian.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * weight_;
        // 旋转部分对位置无影响，保持为零
    }
    
    return true;
}

bool GNSSVelFactor::Evaluate(double const* const* parameters,
                                 double* residuals,
                                 double** jacobians) const {
    // 解析参数块 [px, py, pz, qx, qy, qz, qw]
    // 注意：速度通常存储在单独的参数块，这里简化为与位姿一起
    
    // 假设速度是参数块的一部分（实际实现可能需要调整）
    Eigen::Map<const Eigen::Vector3d> v(parameters[0] + 7); // 需要根据实际存储调整
    
    // 计算残差
    Eigen::Map<Eigen::Vector3d> residual(residuals);
    residual = (v - vel_) * weight_;
    
    // 计算雅可比
    if (jacobians && jacobians[0]) {
        Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian(jacobians[0]);
        jacobian.setZero();
        jacobian.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * weight_;
    }
    
    return true;
}


bool MarginalizationFactor::Evaluate(double const* const* parameters,
                                   double* residuals,
                                   double** jacobians) const {
    // 1. 解析参数块
    // pose: [px, py, pz, qx, qy, qz, qw]
    Eigen::Map<const Eigen::Vector3d> p(parameters[0]);
    Eigen::Map<const Eigen::Quaterniond> q(parameters[0] + 3);
    
    // speed_bias: [vx, vy, vz, bgx, bgy, bgz, bax, bay, baz]
    Eigen::Map<const Eigen::Vector3d> v(parameters[1]);
    Eigen::Map<const Eigen::Vector3d> bg(parameters[1] + 3);
    Eigen::Map<const Eigen::Vector3d> ba(parameters[1] + 6);

    // 2. 计算残差: r = b + H * [pose; speed_bias]
    Eigen::Map<Eigen::Matrix<double, 15, 1>> r(residuals);
    r = b_marg_;
    
    // 提取H矩阵的对应块
    const Eigen::Matrix<double, 15, 7> H_pose = H_marg_.leftCols(7);
    const Eigen::Matrix<double, 15, 9> H_speed_bias = H_marg_.rightCols(9);
    
    // 构造参数向量
    Eigen::Matrix<double, 7, 1> pose_vec;
    pose_vec << p, q.coeffs(); // [px,py,pz, qx,qy,qz,qw]
    
    Eigen::Matrix<double, 9, 1> speed_bias_vec;
    speed_bias_vec << v, bg, ba;
    
    // 累加H*x
    r += H_pose * pose_vec + H_speed_bias * speed_bias_vec;

    // 3. 计算雅可比（如果请求）
    if (jacobians) {
        if (jacobians[0]) { // 关于位姿的雅可比
            Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> J_pose(jacobians[0]);
            J_pose = H_pose;
        }
        
        if (jacobians[1]) { // 关于速度零偏的雅可比
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J_speed_bias(jacobians[1]);
            J_speed_bias = H_speed_bias;
        }
    }
    
    return true;
}


void SlidingWindow::PerformMarginalization() {
    // ... 计算H_marg和b_marg ...

    // 确保H矩阵尺寸正确 (15x16)
    CHECK_EQ(H_marg_.rows(), 15);
    CHECK_EQ(H_marg_.cols(), 16); // 7(pose) + 9(speed_bias)

    // 创建边缘化因子
    auto* factor = new MarginalizationFactor(H_marg_, b_marg_);
    
    // 添加到优化问题
    // problem_.AddResidualBlock(
    //     factor,
    //     nullptr,  // 损失函数
    //     oldest_pose_param,    // 7维位姿参数块
    //     oldest_speed_bias_param  // 9维速度零偏参数块
    // );
}