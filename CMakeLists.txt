cmake_minimum_required(VERSION 3.15)
project(change3d LANGUAGES CXX CUDA)
# CUDA 13.1 dropped sm_52; target the detected GPU (compute capability 8.9)
set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

# By default do not require OpenVDB; enable with -DUSE_OPENVDB=ON
option(USE_OPENVDB "Link with OpenVDB" OFF)

if(USE_OPENVDB)
  if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    message(WARNING "No CMAKE_TOOLCHAIN_FILE specified. To use vcpkg pass -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg.cmake")
  endif()
  find_package(OpenVDB CONFIG REQUIRED)
  # Some OpenVDB builds use BLOSC compression; link it explicitly if available.
  find_package(blosc CONFIG QUIET)
endif()

# Minimal target: demo3d_obj (no external deps by default)
add_executable(demo3d_obj demo3d_obj.cpp)
target_include_directories(demo3d_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(demo3d_obj PRIVATE OpenMP::OpenMP_CXX)

# CUDA + cuBLAS for GPU A_mult
find_package(CUDAToolkit REQUIRED)
add_library(gpu_amult STATIC gpu_amult.cu gpu_amult.h)
set_target_properties(gpu_amult PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON)
target_link_libraries(gpu_amult PRIVATE CUDA::cudart CUDA::cublas)
target_include_directories(gpu_amult PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(demo3d_obj PRIVATE gpu_amult)

if(USE_OPENVDB)
  target_link_libraries(demo3d_obj PRIVATE OpenVDB::openvdb)
  if(TARGET blosc_static)
    target_link_libraries(demo3d_obj PRIVATE blosc_static)
  elseif(TARGET blosc)
    target_link_libraries(demo3d_obj PRIVATE blosc)
  endif()
endif()

# Other tools (optional) can be added by enabling USE_OPENVDB and editing this file.
