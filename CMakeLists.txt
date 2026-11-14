cmake_minimum_required(VERSION 3.24)

option(TINF_CUDA "Build the CUDA engine (requires the CUDA toolkit)" ON)

# Default GPU architecture: the GPU(s) in this machine. Override with e.g.
#   -DCMAKE_CUDA_ARCHITECTURES=86        (RTX 30xx)    or    "75;86;89"   (portable build)
if(TINF_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(tinyinfer VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_property(_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _multi_config AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# put every executable in build/bin (no per-config subfolder with Visual Studio)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<0:>)

if(MSVC)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W3> $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
else()
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
endif()

find_package(OpenMP)

# ---------------------------------------------------------------- core (pure C++)
add_library(tinf_core STATIC
  src/model_file.cpp
  src/tokenizer.cpp
  src/sampler.cpp
  src/quant.cpp
  src/cpu_engine.cpp)
target_include_directories(tinf_core PUBLIC include)
if(OpenMP_CXX_FOUND)
  target_link_libraries(tinf_core PUBLIC OpenMP::OpenMP_CXX)
endif()

enable_testing()

add_executable(tinf_cpu_tests tests/test_cpu.cpp)
target_link_libraries(tinf_cpu_tests PRIVATE tinf_core)
add_test(NAME cpu_tests COMMAND tinf_cpu_tests)

# ---------------------------------------------------------------- CUDA
if(TINF_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  add_library(tinf_cuda STATIC src/cuda/kernels.cu src/cuda/gpu_engine.cu)
  target_include_directories(tinf_cuda PUBLIC src/cuda)
  target_link_libraries(tinf_cuda PUBLIC tinf_core CUDA::cudart_static)
  target_compile_definitions(tinf_cuda PUBLIC TINF_WITH_CUDA=1)
  target_compile_options(tinf_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

  add_executable(tinyinfer src/main.cpp)
  target_link_libraries(tinyinfer PRIVATE tinf_cuda)

  add_executable(tinf_tests tests/test_kernels.cu)
  target_link_libraries(tinf_tests PRIVATE tinf_cuda)
  # GPU tests need a GPU; run with `ctest -L gpu`
  add_test(NAME gpu_kernel_tests COMMAND tinf_tests)
  set_tests_properties(gpu_kernel_tests PROPERTIES LABELS gpu)

  add_executable(tinf_bench_kernels bench/bench_kernels.cu)
  target_link_libraries(tinf_bench_kernels PRIVATE tinf_cuda)

  message(STATUS "tinyinfer: CUDA ${CUDAToolkit_VERSION}, architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
  add_executable(tinyinfer src/main.cpp)
  target_link_libraries(tinyinfer PRIVATE tinf_core)
  message(STATUS "tinyinfer: CPU-only build (TINF_CUDA=OFF)")
endif()
