cmake_minimum_required(VERSION 3.16)
project(mylib_kokkos)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

# ============================================================================
# Kokkos - fetch or use pre-built
# ============================================================================
set(KOKKOS_PREBUILT_DIR "${CMAKE_SOURCE_DIR}/../_deps/kokkos-install")

if(EXISTS "${KOKKOS_PREBUILT_DIR}/lib/cmake/Kokkos/KokkosConfig.cmake" OR
   EXISTS "${KOKKOS_PREBUILT_DIR}/lib64/cmake/Kokkos/KokkosConfig.cmake")
    message(STATUS "Using pre-built Kokkos from ${KOKKOS_PREBUILT_DIR}")
    set(Kokkos_ROOT "${KOKKOS_PREBUILT_DIR}")
    find_package(Kokkos REQUIRED)
else()
    message(STATUS "Fetching Kokkos via FetchContent...")
    FetchContent_Declare(
        kokkos
        GIT_REPOSITORY https://github.com/kokkos/kokkos.git
        GIT_TAG        4.5.01
    )
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

find_package(pybind11 REQUIRED)

# ============================================================================
# Build the Python module
# ============================================================================
pybind11_add_module(mylib
    src/mylib.cpp
    src/bindings.cpp
)

target_include_directories(mylib PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(mylib PRIVATE Kokkos::kokkos)

set_target_properties(mylib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python
)
