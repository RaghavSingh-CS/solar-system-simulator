cmake_minimum_required(VERSION 3.20)

project(solar_system_simulator VERSION 0.1.0 LANGUAGES CXX)

option(NBODY_ENABLE_OPENMP "Build the OpenMP force backend" OFF)
option(NBODY_ENABLE_CUDA "Build the CUDA force backend" OFF)
option(NBODY_ENABLE_MPI "Build the optional MPI distributed-memory backend" OFF)
option(NBODY_BUILD_VISUALIZATION "Build the optional GLFW/OpenGL viewer" OFF)
set(NBODY_STAGE5_SOURCE_SHA "uncontrolled" CACHE STRING
    "Source commit embedded in Stage 5 benchmark/test binaries")
set(NBODY_STAGE5_PROTOCOL_VERSION "none" CACHE STRING
    "Stage 5 protocol embedded in benchmark/test binaries")

function(nbody_embed_stage5_identity target)
    file(TO_CMAKE_PATH "${CMAKE_CXX_COMPILER}" nbody_stage5_cxx_compiler)
    target_compile_definitions(${target} PRIVATE
        NBODY_STAGE5_SOURCE_SHA="${NBODY_STAGE5_SOURCE_SHA}"
        NBODY_STAGE5_PROTOCOL_VERSION="${NBODY_STAGE5_PROTOCOL_VERSION}"
        NBODY_STAGE5_CXX_COMPILER="${nbody_stage5_cxx_compiler}"
        NBODY_STAGE5_CXX_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
        NBODY_STAGE5_CXX_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
        NBODY_STAGE5_GENERATOR="${CMAKE_GENERATOR}"
    )
endfunction()

add_library(nbody_core
    src/diagnostics.cpp
    src/scenarios.cpp
    src/solar_system.cpp
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
    nbody_embed_stage5_identity(nbody_cuda_benchmark)
    file(TO_CMAKE_PATH "${CMAKE_CUDA_COMPILER}" nbody_stage5_cuda_compiler)
    if(CMAKE_CUDA_HOST_COMPILER)
        file(TO_CMAKE_PATH "${CMAKE_CUDA_HOST_COMPILER}" nbody_stage5_cuda_host_compiler)
    else()
        file(TO_CMAKE_PATH "${CMAKE_CXX_COMPILER}" nbody_stage5_cuda_host_compiler)
    endif()
    target_compile_definitions(nbody_cuda_benchmark PRIVATE
        NBODY_STAGE5_CUDA_COMPILER="${nbody_stage5_cuda_compiler}"
        NBODY_STAGE5_CUDA_COMPILER_ID="${CMAKE_CUDA_COMPILER_ID}"
        NBODY_STAGE5_CUDA_COMPILER_VERSION="${CMAKE_CUDA_COMPILER_VERSION}"
        NBODY_STAGE5_CUDA_HOST_COMPILER="${nbody_stage5_cuda_host_compiler}"
        NBODY_STAGE5_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}"
    )
endif()
add_executable(nbody_headless src/main.cpp)
target_link_libraries(nbody_headless PRIVATE nbody_core)

add_executable(nbody_benchmark src/benchmark.cpp)
target_link_libraries(nbody_benchmark PRIVATE nbody_core)
target_compile_definitions(nbody_benchmark PRIVATE
    NBODY_BUILD_CONFIGURATION="$<CONFIG>"
)
nbody_embed_stage5_identity(nbody_benchmark)

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
    string(REPLACE ";" "," nbody_stage5_openmp_libraries "${OpenMP_CXX_LIB_NAMES}")
    target_compile_definitions(nbody_benchmark PRIVATE
        NBODY_STAGE5_OPENMP_FLAGS="${OpenMP_CXX_FLAGS}"
        NBODY_STAGE5_OPENMP_LIBRARIES="${nbody_stage5_openmp_libraries}"
    )
    target_link_libraries(nbody_benchmark PRIVATE nbody_openmp)

    if(MSVC)
        target_compile_options(nbody_openmp PRIVATE /W4 /permissive-)
    else()
        target_compile_options(nbody_openmp PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

if(NBODY_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    add_library(nbody_mpi backends/mpi/mpi_backend.cpp)
    target_include_directories(nbody_mpi PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_features(nbody_mpi PUBLIC cxx_std_17)
    target_link_libraries(nbody_mpi PUBLIC nbody_core MPI::MPI_CXX)
    if(NBODY_ENABLE_OPENMP)
        target_compile_definitions(nbody_mpi PRIVATE NBODY_MPI_HAS_OPENMP=1)
        target_link_libraries(nbody_mpi PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(MSVC)
        target_compile_options(nbody_mpi PRIVATE /W4 /permissive-)
    else()
        target_compile_options(nbody_mpi PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_executable(nbody_mpi_benchmark src/mpi_benchmark.cpp)
    target_link_libraries(nbody_mpi_benchmark PRIVATE nbody_mpi)
    target_compile_definitions(nbody_mpi_benchmark PRIVATE
        NBODY_BUILD_CONFIGURATION="$<CONFIG>"
    )
endif()

if(NBODY_BUILD_VISUALIZATION)
    message(STATUS
        "Visualization enabled: fetching pinned GLFW, GLM, and Dear ImGui dependencies if absent; using vendored GLAD loader"
    )
    enable_language(C)
    include(FetchContent)

    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG a74efa0d5628b74adc0426af4c5710e287fa7c2c
        GIT_PROGRESS TRUE
    )

    set(GLM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLM_BUILD_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG 0af55ccecd98d4e5a8d1fad7de25ba429d60e863
        GIT_PROGRESS TRUE
    )

    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG f1cc2ae15e53a861a874c3034aae6798fde194ab
        GIT_PROGRESS TRUE
    )

    FetchContent_MakeAvailable(glfw glm imgui)

    # Generated reproducibly from GLAD 2.0.6 at the pinned revision documented
    # in third_party/glad/README.md. Keeping the generated loader in-tree avoids
    # requiring Python/Jinja2 during a clean visualization build.
    add_library(nbody_glad STATIC third_party/glad/src/gl.c)
    target_include_directories(nbody_glad PUBLIC third_party/glad/include)

    add_library(nbody_imgui STATIC
        "${imgui_SOURCE_DIR}/imgui.cpp"
        "${imgui_SOURCE_DIR}/imgui_draw.cpp"
        "${imgui_SOURCE_DIR}/imgui_tables.cpp"
        "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
        "${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp"
        "${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp"
    )
    target_include_directories(nbody_imgui PUBLIC
        "${imgui_SOURCE_DIR}"
        "${imgui_SOURCE_DIR}/backends"
    )
    target_compile_features(nbody_imgui PUBLIC cxx_std_17)
    target_link_libraries(nbody_imgui PUBLIC glfw)

    add_library(nbody_visualization_model
        visualization/camera.cpp
        visualization/conservation.cpp
        visualization/compute_comparison.cpp
        visualization/controller.cpp
        visualization/presentation.cpp
        visualization/render_space.cpp
        visualization/scenario.cpp
        visualization/trails.cpp
        visualization/workload.cpp
    )
    target_include_directories(nbody_visualization_model PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_features(nbody_visualization_model PUBLIC cxx_std_17)
    target_link_libraries(nbody_visualization_model PUBLIC nbody_core glm::glm)
    if(NBODY_ENABLE_OPENMP)
        target_compile_definitions(nbody_visualization_model PRIVATE NBODY_VIS_HAS_OPENMP=1)
        target_link_libraries(nbody_visualization_model PRIVATE nbody_openmp)
    endif()
    if(NBODY_ENABLE_CUDA)
        target_compile_definitions(nbody_visualization_model PRIVATE NBODY_VIS_HAS_CUDA=1)
        target_link_libraries(nbody_visualization_model PRIVATE nbody_cuda)
    endif()

    add_executable(nbody_visualizer
        visualization/main.cpp
        visualization/renderer.cpp
        visualization/renderer.hpp
    )
    target_compile_features(nbody_visualizer PRIVATE cxx_std_17)
    target_link_libraries(nbody_visualizer PRIVATE
        nbody_visualization_model
        nbody_glad
        nbody_imgui
        glfw
    )

    foreach(target IN ITEMS nbody_visualization_model nbody_visualizer)
        if(MSVC)
            target_compile_options(${target} PRIVATE /W4 /permissive-)
        else()
            target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        endif()
    endforeach()
endif()

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME nbody_stage5_v2_tools
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_stage5_v2_tools.py
        )
        add_test(
            NAME nbody_stage6d_tools
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_stage6d_tools.py
        )
    endif()

    add_executable(nbody_tests tests/test_two_body.cpp)
    target_link_libraries(nbody_tests PRIVATE nbody_core)
    target_compile_definitions(nbody_tests PRIVATE NBODY_BUILD_CONFIGURATION="$<CONFIG>")
    nbody_embed_stage5_identity(nbody_tests)
    add_test(NAME nbody_two_body COMMAND nbody_tests)

    add_executable(nbody_solar_system_tests tests/test_solar_system.cpp)
    target_link_libraries(nbody_solar_system_tests PRIVATE nbody_core)
    target_compile_definitions(nbody_solar_system_tests PRIVATE NBODY_BUILD_CONFIGURATION="$<CONFIG>")
    nbody_embed_stage5_identity(nbody_solar_system_tests)
    add_test(NAME nbody_solar_system COMMAND nbody_solar_system_tests)

    if(NBODY_ENABLE_OPENMP)
        add_executable(nbody_openmp_tests tests/test_openmp.cpp)
        target_link_libraries(nbody_openmp_tests PRIVATE nbody_openmp)
        target_compile_definitions(nbody_openmp_tests PRIVATE NBODY_BUILD_CONFIGURATION="$<CONFIG>")
        nbody_embed_stage5_identity(nbody_openmp_tests)
        add_test(NAME nbody_openmp_validation COMMAND nbody_openmp_tests)
    endif()

    if(NBODY_ENABLE_CUDA)
        add_executable(nbody_cuda_tests tests/test_cuda.cpp)
        target_link_libraries(nbody_cuda_tests PRIVATE nbody_cuda)
        target_compile_definitions(nbody_cuda_tests PRIVATE NBODY_BUILD_CONFIGURATION="$<CONFIG>")
        nbody_embed_stage5_identity(nbody_cuda_tests)
        add_test(NAME nbody_cuda_validation COMMAND nbody_cuda_tests)
        set_tests_properties(nbody_cuda_validation PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    if(NBODY_ENABLE_MPI)
        add_executable(nbody_mpi_tests tests/test_mpi.cpp)
        target_link_libraries(nbody_mpi_tests PRIVATE nbody_mpi)
        foreach(rank_count IN ITEMS 1 2 4)
            add_test(
                NAME nbody_mpi_${rank_count}_rank
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${rank_count}
                        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:nbody_mpi_tests> ${MPIEXEC_POSTFLAGS}
            )
            set_tests_properties(nbody_mpi_${rank_count}_rank PROPERTIES PROCESSORS ${rank_count})
        endforeach()
        if(NBODY_ENABLE_OPENMP)
            add_test(
                NAME nbody_mpi_openmp_hybrid
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:nbody_mpi_tests>
                        --threads 2 ${MPIEXEC_POSTFLAGS}
            )
            set_tests_properties(nbody_mpi_openmp_hybrid PROPERTIES PROCESSORS 4)
        endif()
    endif()

    if(NBODY_BUILD_VISUALIZATION)
        add_executable(nbody_visualization_tests tests/test_visualization.cpp)
        target_link_libraries(nbody_visualization_tests PRIVATE nbody_visualization_model)
        add_test(NAME nbody_visualization_model COMMAND nbody_visualization_tests)
        if(NBODY_ENABLE_CUDA)
            add_executable(nbody_visualization_cuda_tests tests/test_visualization_cuda.cpp)
            target_link_libraries(nbody_visualization_cuda_tests PRIVATE nbody_visualization_model)
            add_test(NAME nbody_visualization_cuda COMMAND nbody_visualization_cuda_tests)
            set_tests_properties(nbody_visualization_cuda PROPERTIES SKIP_RETURN_CODE 77)
        endif()
    endif()
endif()
