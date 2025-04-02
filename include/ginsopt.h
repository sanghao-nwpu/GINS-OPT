#pragma once
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <deque>

// 1. 基本类型定义
struct IMUData { 
    double timestamp;
    Eigen::Vector3d gyro;
    Eigen::Vector3d acc;    
};
struct GNSSData { 
    double timestamp;
    Eigen::Vector3d pos;
    Eigen::Vector3d pos_dev;
    Eigen::Vector3d vel;
    double vel_forward;
    double vel_track;
    double vel_upward;
    int sat_num;
    int type;   // 0: None, 1: Single, 4: RTK fixed, 5: RTK float
};
struct State { 
    double timestamp;
    Eigen::Vector3d pos;
    Eigen::Quaterniond quat;
    Eigen::Vector3d vel;
    Eigen::Vector3d gyro_bias;
    Eigen::Vector3d acc_bias;
};

// 1. 通用函数
Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m << 0, -v.z(), v.y(),
         v.z(), 0, -v.x(),
         -v.y(), v.x(), 0;
    return m;
}

Eigen::Matrix3d rightJacobian(const Eigen::Vector3d& phi) {
    double angle = phi.norm();
    if (angle < 1e-5) {
        return Eigen::Matrix3d::Identity();
    }
    Eigen::Matrix3d J;
    Eigen::Vector3d axis = phi.normalized();
    J = sin(angle)/angle * Eigen::Matrix3d::Identity() 
        + (1 - sin(angle)/angle) * axis * axis.transpose() 
        + (1 - cos(angle))/angle * skewSymmetric(axis);
    return J;
}


// 2. IMU预积分器（简化版）
class IMUPreintegrator {
public:
    // 构造函数
    IMUPreintegrator();
    
    // 核心功能
    void Integrate(const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro, double dt);
    void Reset();
    
    // 预积分结果获取
    const double& delta_t() const { return delta_t_; }
    const Eigen::Vector3d& delta_p() const { return delta_p_; }
    const Eigen::Quaterniond& delta_q() const { return delta_q_; }
    const Eigen::Vector3d& delta_v() const { return delta_v_; }
    const Eigen::Matrix<double, 9, 9>& covariance() const { return covariance_; }
    const Eigen::Vector3d& dp_dba() const { return dp_dba_; }
    const Eigen::Vector3d& dp_dbg() const { return dp_dbg_; }
    const Eigen::Vector3d& dv_dba() const { return dv_dba_; }
    const Eigen::Vector3d& dv_dbg() const { return dv_dbg_; }
    const Eigen::Vector3d& dq_dbg() const { return dq_dbg_; }
    
private:
    
    double delta_t_;               // 预积分时长

    // 预积分状态
    Eigen::Vector3d delta_p_;      // 位置变化量
    Eigen::Quaterniond delta_q_;   // 旋转变化量
    Eigen::Vector3d delta_v_;      // 速度变化量
    
    // 噪声参数
    double accel_noise_var_;
    double gyro_noise_var_;
    double accel_bias_noise_var_;
    double gyro_bias_noise_var_;
    
    // 协方差矩阵
    Eigen::Matrix<double, 9, 9> covariance_;
    
    // 零偏
    Eigen::Vector3d gyro_bias_;
    Eigen::Vector3d accel_bias_;
    
    // 雅可比矩阵
    Eigen::Matrix3d dp_dba_;    // ∂Δp/∂ba
    Eigen::Matrix3d dp_dbg_;    // ∂Δp/∂bg
    Eigen::Matrix3d dv_dba_;    // ∂Δv/∂ba
    Eigen::Matrix3d dv_dbg_;    // ∂Δv/∂bg
    Eigen::Matrix3d dq_dbg_;    // ∂Δq/∂bg
};

// 3. 因子类（内联实现）
class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9> {
public:
    IMUFactor(const IMUPreintegrator& preint, const Eigen::Vector3d& gravity)
        : preint_(preint), gravity_(gravity) {}
    
    virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const override;
    
private:
    const IMUPreintegrator& preint_;
    const Eigen::Vector3d gravity_;
};

class GNSSPosFactor : public ceres::SizedCostFunction<3, 7> {
public:
    GNSSPosFactor(const Eigen::Vector3d& pos, const Eigen::Vector3d& dev)
        : pos_(pos), weight_(dev.cwiseInverse()) {}
    
    virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const override;
    
private:
    const Eigen::Vector3d pos_;
    const Eigen::Vector3d weight_;
};

class GNSSVelFactor : public ceres::SizedCostFunction<3, 7> {
public:
    GNSSVelFactor(const Eigen::Vector3d& vel, const Eigen::Vector3d& dev)
        : vel_(vel), weight_(dev.cwiseInverse()) {}
    
    virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const override;
    
private:
    const Eigen::Vector3d vel_;
    const Eigen::Vector3d weight_;
};


class MarginalizationFactor : public ceres::SizedCostFunction<15, 7, 9> {
public:
    // 构造函数
    MarginalizationFactor(const Eigen::MatrixXd& H_marg, 
                        const Eigen::VectorXd& b_marg)
        : H_marg_(H_marg), 
          b_marg_(b_marg) {
        // 验证尺寸
        CHECK_EQ(H_marg_.cols(), 16) << "H矩阵应为16列(7+9)"; 
        CHECK_EQ(b_marg_.rows(), 15) << "b向量应为15维";
    }

    virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const override;

private:
    const Eigen::MatrixXd H_marg_; // 16列矩阵(7+9)
    const Eigen::VectorXd b_marg_; // 15维向量
};


class SlidingWindow {
public:
    // 窗口状态（位姿+速度/零偏）
    struct WindowState {
        double timestamp;
        Eigen::Vector3d p;
        Eigen::Quaterniond q;
        Eigen::Vector3d v;
        Eigen::Vector3d bg;
        Eigen::Vector3d ba;
        IMUPreintegrator preint; // 与前一个状态的预积分
    };

    // 初始化窗口
    explicit SlidingWindow(size_t max_size = 10) : max_size_(max_size) {}
    
    // 添加新状态
    void AddState(const WindowState& state) {
        if (!states_.empty()) {
            // 计算与前一帧的时间差
            double dt = state.timestamp - states_.back().timestamp;
            if (dt <= 0) throw std::runtime_error("Non-increasing timestamps");
        }
        states_.push_back(state);
        if (states_.size() > max_size_) {
            MarginalizeOldest();
        }
    }

    // 边缘化最老状态
    void MarginalizeOldest() {
        if (states_.size() <= 2) return; // 至少保留两帧
        
        // 执行边缘化（具体实现见下文）
        PerformMarginalization();
        
        // 移除最老状态
        states_.pop_front();
    }

    // 获取窗口状态
    const std::deque<WindowState>& states() const { return states_; }
    size_t Size() const { return states_.size(); }

private:
    void PerformMarginalization();
    
    std::deque<WindowState> states_;
    size_t max_size_;
    Eigen::MatrixXd H_marg_; // 边缘化先验的H矩阵
    Eigen::VectorXd b_marg_; // 边缘化先验的b向量
};


// 4. 优化器核心（简化接口）
class Optimizer {
public:
    void AddIMUMeasurement(/*...*/);
    void AddGNSSMeasurement(/*...*/);
    void Optimize();
private:
    ceres::Problem problem_;
};