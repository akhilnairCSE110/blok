cmake_minimum_required(VERSION 3.28)
project(blok LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)

add_executable(blok-byte-probe src/cuda_byte_probe.cu)
find_path(UGDS_INCLUDE_DIR ugds.h)
find_library(UGDS_LIBRARY ugds)
if(UGDS_INCLUDE_DIR AND UGDS_LIBRARY)
  target_include_directories(blok-byte-probe PRIVATE ${UGDS_INCLUDE_DIR})
  target_link_libraries(blok-byte-probe PRIVATE ${UGDS_LIBRARY})
  target_compile_definitions(blok-byte-probe PRIVATE BLOK_HAVE_UGDS=1)
endif()
target_compile_options(blok-byte-probe PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-Xptxas=-O3,-v>
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror>
)
