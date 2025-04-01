#pragma once
#include <Eigen/Dense>
#include <ceres/ceres.h>

// 1. 基本类型定义
struct IMUData { /*...*/ };
struct GNSSData { /*...*/ };
struct State { /*...*/ };

// 2. IMU预积分器（简化版）
class IMUPreintegrator {
public:
    void Integrate(const IMUData& imu, double dt);
    // 仅保留核心成员变量...
};

// 3. 因子类（内联实现）
class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9> {
public:
    // 直接在此实现Evaluate函数...
};

class GNSSFactor { /*...*/ };

// 4. 优化器核心（简化接口）
class Optimizer {
public:
    void AddIMUMeasurement(/*...*/);
    void AddGNSSMeasurement(/*...*/);
    void Optimize();
private:
    ceres::Problem problem_;
};