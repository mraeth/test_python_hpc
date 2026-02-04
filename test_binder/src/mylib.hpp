#pragma once

#include <Kokkos_Core.hpp>
#include <vector>
#include <cstddef>

// ============================================================================
// Kokkos initialization
// ============================================================================

void initialize_kokkos();
bool is_kokkos_initialized();

// ============================================================================
// 1D array wrapper using Kokkos::View
// ============================================================================

class Array1D {
public:
    using ViewType = Kokkos::View<double*>;
    using HostMirror = ViewType::HostMirror;

private:
    ViewType data_;
    HostMirror host_data_;
    size_t size_;

public:
    Array1D(size_t n);

    size_t size() const;
    void set(size_t i, double val);
    double get(size_t i) const;

    void sync_to_device();
    void sync_to_host();

    void from_list(const std::vector<double>& values);
    std::vector<double> to_list() const;

    ViewType& view();
    const ViewType& view() const;
};

// ============================================================================
// Kokkos kernel
// ============================================================================

double dot_product(const Array1D& a, const Array1D& b);
