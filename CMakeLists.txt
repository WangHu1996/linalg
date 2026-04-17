cmake_minimum_required(VERSION 3.15)
project(linalg VERSION 1.0.0 LANGUAGES CXX)

# 基础配置
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 引入mdarray
include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.deps)
option(BUILD_MD_DEMO "" OFF)
option(BUILD_MD_UTEST "" OFF)
option(BUILD_MD_BENCHMARK "" OFF)
fetchcontent_declare(
    mdarray
    GIT_REPOSITORY  https://github.com/WangHu1996/mdarray.git
    GIT_TAG         6e8125c8f8742a9f044e69c119404338eb993ae2
    SOURCE_DIR      ${CMAKE_CURRENT_SOURCE_DIR}/.deps/mdarray
)
fetchcontent_makeavailable(mdarray)
fetchcontent_getproperties(mdarray SOURCE_DIR MDARRAY_SOURCE_DIR)
message(STATUS "mdarray source dir: ${MDARRAY_SOURCE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MDARRAY_SOURCE_DIR}/cmake")
include(PlatformSetting)
include(CompilerOption)
include(SelectSimd)

# 头文件
include_directories(${PROJECT_SOURCE_DIR}/include)

# 示例
add_subdirectory(demo)
