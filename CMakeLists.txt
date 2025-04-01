cmake_minimum_required(VERSION 3.10)
project(GINS-OPT LANGUAGES CXX)

# 1. 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. 查找必需依赖
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

# 3. 包含头文件目录
include_directories(
    ${EIGEN3_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    include
)

# 4. 添加可执行文件
add_executable(ginsopt
    main.cpp
    src/ginsopt.cpp
)

# 5. 链接库
target_link_libraries(ginsopt
    ${CERES_LIBRARIES}
)

# 6. 安装规则（可选）
install(TARGETS ginsopt DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)