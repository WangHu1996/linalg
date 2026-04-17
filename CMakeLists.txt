cmake_minimum_required(VERSION 3.15)
project(linalg VERSION 1.0.0 LANGUAGES CXX)

# 基础配置
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 引入mdvector
include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.deps)
option(BUILD_MD_DEMO "" OFF)
option(BUILD_MD_UTEST "" OFF)
option(BUILD_MD_BENCHMARK "" OFF)
fetchcontent_declare(
    mdvector
    GIT_REPOSITORY  https://github.com/WangHu1996/mdvector.git
    GIT_TAG         afad00b179a642194e5d4e7d213d4fac4b10bac7
    SOURCE_DIR      ${CMAKE_CURRENT_SOURCE_DIR}/.deps/mdvector
)
fetchcontent_makeavailable(mdvector)
fetchcontent_getproperties(mdvector SOURCE_DIR MDVECTOR_SOURCE_DIR)
message(STATUS "mdvector source dir: ${MDVECTOR_SOURCE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MDVECTOR_SOURCE_DIR}/cmake")
include(PlatformSetting)
include(CompilerOption)
include(SelectSimd)

# 头文件
include_directories(${PROJECT_SOURCE_DIR}/include)

# 测试文件
add_subdirectory(src)
