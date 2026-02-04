#pragma once

#include <vector>
#include <string>

// A simple data structure
struct Point {
    double x, y, z;

    Point(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}

    double norm() const {
        return std::sqrt(x*x + y*y + z*z);
    }
};

// A simple kernel function
inline double dot_product(const Point& a, const Point& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Another kernel that processes a vector of points
inline double sum_of_norms(const std::vector<Point>& points) {
    double sum = 0.0;
    for (const auto& p : points) {
        sum += p.norm();
    }
    return sum;
}
