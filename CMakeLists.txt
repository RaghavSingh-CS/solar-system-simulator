cmake_minimum_required(VERSION 3.20)

project(solar_system_simulator VERSION 0.1.0 LANGUAGES CXX)

option(NBODY_ENABLE_OPENMP "Build the OpenMP force backend (future stage)" OFF)
option(NBODY_ENABLE_CUDA "Build the CUDA force backend (future stage)" OFF)
option(NBODY_ENABLE_MPI "Build the optional MPI backend (future stage)" OFF)
option(NBODY_BUILD_VISUALIZATION "Build the GLFW/OpenGL viewer (future stage)" OFF)

foreach(feature IN ITEMS NBODY_ENABLE_OPENMP NBODY_ENABLE_CUDA NBODY_ENABLE_MPI NBODY_BUILD_VISUALIZATION)
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
add_executable(nbody_headless src/main.cpp)
target_link_libraries(nbody_headless PRIVATE nbody_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(nbody_tests tests/test_two_body.cpp)
    target_link_libraries(nbody_tests PRIVATE nbody_core)
    add_test(NAME nbody_two_body COMMAND nbody_tests)
endif()
