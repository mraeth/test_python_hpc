cmake_minimum_required(VERSION 3.15)
project(mylib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python3 interpreter and development files
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Locate pybind11 from the venv (pass PYTHON_EXECUTABLE as cmake arg if needed)
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

# Create the Python module
pybind11_add_module(mylib src/bindings.cpp)

# Install to the python directory for easy importing
set_target_properties(mylib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python
)
