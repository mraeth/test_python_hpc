#!/usr/bin/env python3
"""Example script demonstrating C++ library usage from Python."""

import mylib

# Create some points using the C++ struct
p1 = mylib.Point(1.0, 2.0, 3.0)
p2 = mylib.Point(4.0, 5.0, 6.0)

print(f"Point 1: {p1}")
print(f"Point 2: {p2}")

# Call C++ methods
print(f"Norm of p1: {p1.norm()}")

# Call C++ functions
dot = mylib.dot_product(p1, p2)
print(f"Dot product: {dot}")

# Pass a list of points to C++
points = [mylib.Point(1, 0, 0), mylib.Point(0, 1, 0), mylib.Point(0, 0, 1)]
total = mylib.sum_of_norms(points)
print(f"Sum of norms: {total}")

# Modify point attributes directly
p1.x = 10.0
print(f"Modified p1: {p1}")
