#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mylib.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mylib, m) {
    m.doc() = "Kokkos-based array library with Python bindings";

    // Kokkos initialization functions
    m.def("initialize", &initialize_kokkos,
          "Initialize Kokkos runtime");
    m.def("initialize_kokkos", &initialize_kokkos,
          "Initialize Kokkos runtime (alias)");
    m.def("is_kokkos_initialized", &is_kokkos_initialized,
          "Check if Kokkos is initialized");

    // Array1D class
    py::class_<Array1D>(m, "Array1D")
        .def(py::init<size_t>(), py::arg("size"),
             "Create a 1D array of given size")
        .def("size", &Array1D::size,
             "Get array size")
        .def("get", &Array1D::get, py::arg("i"),
             "Get value at index i")
        .def("set", &Array1D::set, py::arg("i"), py::arg("val"),
             "Set value at index i")
        .def("sync_to_device", &Array1D::sync_to_device,
             "Copy data from host to device")
        .def("sync_to_host", &Array1D::sync_to_host,
             "Copy data from device to host")
        .def("from_list", &Array1D::from_list, py::arg("values"),
             "Fill array from Python list")
        .def("to_list", &Array1D::to_list,
             "Convert array to Python list");

    // Kernel functions
    m.def("dot_product", &dot_product, py::arg("a"), py::arg("b"),
          "Compute dot product of two arrays using Kokkos");
}
