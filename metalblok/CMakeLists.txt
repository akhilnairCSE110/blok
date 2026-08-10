cmake_minimum_required(VERSION 3.24)
project(metalblok LANGUAGES C CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)

if(NOT APPLE)
  message(FATAL_ERROR "MetalBlok targets Apple Silicon/macOS")
endif()

option(METALBLOK_OFFLINE_METALLIB "Compile an offline metallib with xcrun" OFF)
option(METALBLOK_FAST_MATH "Enable fast-math only after correctness parity" OFF)

add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter
                    -fno-exceptions -fno-rtti -O2)
if(METALBLOK_FAST_MATH)
  add_compile_options(-ffast-math)
endif()

set(METAL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels.metal)
if(METALBLOK_OFFLINE_METALLIB)
  set(METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/kernels.air)
  set(METALLIB ${CMAKE_CURRENT_BINARY_DIR}/metalblok.metallib)
  add_custom_command(OUTPUT ${METAL_AIR}
    COMMAND xcrun metal -c ${METAL_SRC} -o ${METAL_AIR}
    DEPENDS ${METAL_SRC} VERBATIM)
  add_custom_command(OUTPUT ${METALLIB}
    COMMAND xcrun metallib ${METAL_AIR} -o ${METALLIB}
    DEPENDS ${METAL_AIR} VERBATIM)
  add_custom_target(metalblok_metallib ALL DEPENDS ${METALLIB})
  set(METALBLOK_KERNEL_PATH ${METALLIB})
  set(METALBLOK_KERNEL_IS_SOURCE 0)
else()
  # newLibraryWithSource uses Apple's runtime compiler. This deliberately
  # keeps bring-up independent of the optional command-line Metal component.
  set(METALBLOK_KERNEL_PATH ${METAL_SRC})
  set(METALBLOK_KERNEL_IS_SOURCE 1)
endif()

add_executable(metalblok
  src/main.cpp
  src/preflight.cpp
  src/model.cpp
  src/gguf.cpp
  src/gguf_dequant.cpp
  src/gguf_model.cpp
  src/gguf_runtime.cpp
  src/tokenizer.cpp
  src/metal_ctx.mm
  src/mla.cpp
  src/moe.cpp
  src/streamer.cpp
  src/kvcache.cpp
  src/pread_ring.cpp)

if(METALBLOK_OFFLINE_METALLIB)
  add_dependencies(metalblok metalblok_metallib)
endif()

target_link_libraries(metalblok PRIVATE
  "-framework Metal" "-framework Foundation" "-framework CoreGraphics")
target_compile_definitions(metalblok PRIVATE
  METALBLOK_KERNEL_PATH="${METALBLOK_KERNEL_PATH}"
  METALBLOK_KERNEL_IS_SOURCE=${METALBLOK_KERNEL_IS_SOURCE})
set_source_files_properties(src/metal_ctx.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")

add_executable(pread_ring_test tools/pread_ring_test.cpp src/pread_ring.cpp)
add_executable(model_preflight_test tests/model_preflight_test.cpp src/preflight.cpp)
target_include_directories(model_preflight_test PRIVATE src)
add_executable(router_ref_test tests/router_ref_test.cpp)
target_include_directories(router_ref_test PRIVATE src)

enable_testing()
add_test(NAME model_preflight COMMAND model_preflight_test)
add_test(NAME router_reference COMMAND router_ref_test)
