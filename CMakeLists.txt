cmake_minimum_required(VERSION 3.20)

project(solar_system_simulator VERSION 0.1.0 LANGUAGES CXX)

option(NBODY_ENABLE_OPENMP "Build the OpenMP force backend" OFF)
option(NBODY_ENABLE_CUDA "Build the CUDA force backend" OFF)
option(NBODY_ENABLE_MPI "Build the optional MPI backend (future stage)" OFF)
option(NBODY_BUILD_VISUALIZATION "Build the GLFW/OpenGL viewer (future stage)" OFF)

foreach(feature IN ITEMS NBODY_ENABLE_MPI NBODY_BUILD_VISUALIZATION)
    if(${feature})
        message(FATAL_ERROR "${feature}=ON is not implemented in the foundation stage; leave it OFF")
    endif()
endforeach()

add_library(nbody_core
    src/diagnostics.cpp
    src/scenarios.cpp
    src/simulation.cpp
)
target_include_directories(nbody_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(nbody_core PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(nbody_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(nbody_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(NBODY_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "NBODY_ENABLE_CUDA=ON, but no usable CUDA compiler was found. "
            "Install the NVIDIA CUDA toolkit or configure with NBODY_ENABLE_CUDA=OFF."
        )
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(nbody_cuda backends/cuda/cuda_backend.cu)
    target_include_directories(nbody_cuda PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_features(nbody_cuda PUBLIC cxx_std_17)
    target_link_libraries(nbody_cuda PUBLIC nbody_core CUDA::cudart)
    set_target_properties(nbody_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
        CUDA_EXTENSIONS NO
    )

    add_executable(nbody_cuda_benchmark src/cuda_benchmark.cpp)
    target_link_libraries(nbody_cuda_benchmark PRIVATE nbody_cuda)
    target_compile_definitions(nbody_cuda_benchmark PRIVATE
        NBODY_BUILD_CONFIGURATION="$<CONFIG>"
    )
endif()
add_executable(nbody_headless src/main.cpp)
target_link_libraries(nbody_headless PRIVATE nbody_core)

add_executable(nbody_benchmark src/benchmark.cpp)
target_link_libraries(nbody_benchmark PRIVATE nbody_core)
target_compile_definitions(nbody_benchmark PRIVATE
    NBODY_BUILD_CONFIGURATION="$<CONFIG>"
)

add_executable(nbody_conservation_probe src/conservation_probe.cpp)
target_link_libraries(nbody_conservation_probe PRIVATE nbody_core)
target_compile_definitions(nbody_conservation_probe PRIVATE
    NBODY_BUILD_CONFIGURATION="$<CONFIG>"
)

if(NBODY_ENABLE_OPENMP)
    find_package(OpenMP QUIET COMPONENTS CXX)
    if(NOT OpenMP_CXX_FOUND)
        message(FATAL_ERROR
            "NBODY_ENABLE_OPENMP=ON, but the selected C++ compiler has no usable OpenMP support. "
            "Install/enable the compiler OpenMP runtime or configure with NBODY_ENABLE_OPENMP=OFF."
        )
    endif()

    add_library(nbody_openmp backends/openmp/openmp_backend.cpp)
    target_include_directories(nbody_openmp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_features(nbody_openmp PUBLIC cxx_std_17)
    target_link_libraries(nbody_openmp PUBLIC nbody_core OpenMP::OpenMP_CXX)
    target_compile_definitions(nbody_benchmark PRIVATE NBODY_HAS_OPENMP=1)
    target_link_libraries(nbody_benchmark PRIVATE nbody_openmp)

    if(MSVC)
        target_compile_options(nbody_openmp PRIVATE /W4 /permissive-)
    else()
        target_compile_options(nbody_openmp PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(nbody_tests tests/test_two_body.cpp)
    target_link_libraries(nbody_tests PRIVATE nbody_core)
    add_test(NAME nbody_two_body COMMAND nbody_tests)

    if(NBODY_ENABLE_OPENMP)
        add_executable(nbody_openmp_tests tests/test_openmp.cpp)
        target_link_libraries(nbody_openmp_tests PRIVATE nbody_openmp)
        add_test(NAME nbody_openmp_validation COMMAND nbody_openmp_tests)
    endif()

    if(NBODY_ENABLE_CUDA)
        add_executable(nbody_cuda_tests tests/test_cuda.cpp)
        target_link_libraries(nbody_cuda_tests PRIVATE nbody_cuda)
        add_test(NAME nbody_cuda_validation COMMAND nbody_cuda_tests)
    endif()
endif()
