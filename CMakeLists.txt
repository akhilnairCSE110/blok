cmake_minimum_required(VERSION 3.28)
project(blok LANGUAGES CXX CUDA)
find_package(CUDAToolkit 12.8 REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)

add_executable(blok-kimi-exec src/kimi_exec.cu)
add_subdirectory(sub_dir/uGDS EXCLUDE_FROM_ALL)
target_link_libraries(blok-kimi-exec PRIVATE ugds)
target_compile_options(blok-kimi-exec PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic,-Werror;-Xptxas=-O3,-v>
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror>)
