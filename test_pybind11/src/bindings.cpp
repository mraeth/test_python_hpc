#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // For automatic std::vector conversion
#include "mylib.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mylib, m) {
    m.doc() = "Example C++ library with Python bindings";

    // Expose the Point struct
    py::class_<Point>(m, "Point")
        .def(py::init<double, double, double>(),
             py::arg("x") = 0, py::arg("y") = 0, py::arg("z") = 0)
        .def_readwrite("x", &Point::x)
        .def_readwrite("y", &Point::y)
        .def_readwrite("z", &Point::z)
        .def("norm", &Point::norm)
        .def("__repr__", [](const Point& p) {
            return "Point(" + std::to_string(p.x) + ", " +
                   std::to_string(p.y) + ", " + std::to_string(p.z) + ")";
        });

    // Expose the functions
    m.def("dot_product", &dot_product, "Compute dot product of two points",
          py::arg("a"), py::arg("b"));

    m.def("sum_of_norms", &sum_of_norms, "Compute sum of norms of a list of points",
          py::arg("points"));
}
