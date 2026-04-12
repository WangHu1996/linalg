cmake_minimum_required(VERSION 3.15)
project(mdvector VERSION 1.0.0 LANGUAGES CXX)

# 基础配置
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 引入mdvector
include_directories(${PROJECT_SOURCE_DIR}/mdvector/include)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/mdvector/cmake")
include(PlatformSetting)
include(CompilerOption)
include(SelectSimd)

# 头文件
include_directories(${PROJECT_SOURCE_DIR}/include)

# 测试文件
add_subdirectory(src)
