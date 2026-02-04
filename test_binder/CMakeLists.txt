cmake_minimum_required(VERSION 3.16)
project(mylib_kokkos)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Required for linking Kokkos into a shared library (Python module)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
include(ExternalProject)

# ============================================================================
# Kokkos - use pre-built from _deps or download via FetchContent
# ============================================================================

# Check for pre-built Kokkos in ../_deps/kokkos-install
set(KOKKOS_PREBUILT_DIR "${CMAKE_SOURCE_DIR}/../_deps/kokkos-install")

if(EXISTS "${KOKKOS_PREBUILT_DIR}/lib/cmake/Kokkos/KokkosConfig.cmake" OR
   EXISTS "${KOKKOS_PREBUILT_DIR}/lib64/cmake/Kokkos/KokkosConfig.cmake")
    message(STATUS "Using pre-built Kokkos from ${KOKKOS_PREBUILT_DIR}")
    set(Kokkos_ROOT "${KOKKOS_PREBUILT_DIR}")
    find_package(Kokkos REQUIRED)
else()
    message(STATUS "Pre-built Kokkos not found, using FetchContent...")
    message(STATUS "  (Run ../build_kokkos.sh to build Kokkos once for faster rebuilds)")

    FetchContent_Declare(
        kokkos
        GIT_REPOSITORY https://github.com/kokkos/kokkos.git
        GIT_TAG        4.5.01
    )

    # Configure Kokkos options
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "")
    set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "")

    FetchContent_MakeAvailable(kokkos)
endif()

# ============================================================================
# Python and pybind11
# ============================================================================
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_result
)

if(NOT pybind11_result EQUAL 0)
    message(FATAL_ERROR "Could not find pybind11. Install with: pip install pybind11")
endif()

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE pybind11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 REQUIRED)

# ============================================================================
# LLVM/Clang - required for Binder
# ============================================================================
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")

# ============================================================================
# Binder - build from source
# ============================================================================
set(BINDER_INSTALL_DIR "${CMAKE_BINARY_DIR}/binder-install")
set(BINDER_EXECUTABLE "${BINDER_INSTALL_DIR}/bin/binder")

ExternalProject_Add(binder_project
    GIT_REPOSITORY https://github.com/RosettaCommons/binder.git
    GIT_TAG        master
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${BINDER_INSTALL_DIR}
        -DCMAKE_CXX_STANDARD=17
        -DLLVM_DIR=${LLVM_DIR}
        -DClang_DIR=${Clang_DIR}
        -DCMAKE_BUILD_TYPE=Release
        -DBINDER_ENABLE_TEST=OFF
    BUILD_BYPRODUCTS ${BINDER_EXECUTABLE}
)

message(STATUS "Binder will be built from source")
message(STATUS "Binder executable will be at: ${BINDER_EXECUTABLE}")

# ============================================================================
# Binding generation - runs automatically
# ============================================================================

# Get Kokkos include directory
get_target_property(KOKKOS_INCLUDE_DIRS Kokkos::kokkoscore INTERFACE_INCLUDE_DIRECTORIES)
list(GET KOKKOS_INCLUDE_DIRS 0 KOKKOS_INCLUDE_DIR)

set(GENERATED_BINDINGS "${CMAKE_SOURCE_DIR}/generated/mylib.cpp")

add_custom_command(
    OUTPUT ${GENERATED_BINDINGS}
    COMMAND ${BINDER_EXECUTABLE}
        --root-module mylib
        --prefix ${CMAKE_SOURCE_DIR}/generated/
        --config ${CMAKE_SOURCE_DIR}/binder.config
        --single-file
        ${CMAKE_SOURCE_DIR}/src/all_includes.hpp
        --
        -std=c++17
        -I${CMAKE_SOURCE_DIR}/src
        -I${KOKKOS_INCLUDE_DIR}
        -I${pybind11_INCLUDE_DIR}
        -DNDEBUG
    DEPENDS
        ${CMAKE_SOURCE_DIR}/src/mylib.hpp
        ${CMAKE_SOURCE_DIR}/src/all_includes.hpp
        ${CMAKE_SOURCE_DIR}/binder.config
        binder_project
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating pybind11 bindings with Binder"
    VERBATIM
)

add_custom_target(generate_bindings
    DEPENDS ${GENERATED_BINDINGS}
)

# ============================================================================
# Build the Python module
# ============================================================================
pybind11_add_module(mylib
    src/mylib.cpp           # Library implementation
    generated/mylib.cpp     # Python bindings (auto-generated)
)

# Ensure bindings are generated before building the module
add_dependencies(mylib generate_bindings)

target_include_directories(mylib PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(mylib PRIVATE Kokkos::kokkos)

set_target_properties(mylib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python
)
