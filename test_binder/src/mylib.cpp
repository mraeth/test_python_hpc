#include "mylib.hpp"
#include <stdexcept>

// ============================================================================
// Kokkos initialization
// ============================================================================

void initialize_kokkos() {
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
    }
}

bool is_kokkos_initialized() {
    return Kokkos::is_initialized();
}

// ============================================================================
// Array1D implementation
// ============================================================================

Array1D::Array1D(size_t n) : data_("array", n), size_(n) {
    host_data_ = Kokkos::create_mirror_view(data_);
}

size_t Array1D::size() const {
    return size_;
}

void Array1D::set(size_t i, double val) {
    if (i >= size_) throw std::out_of_range("Index out of bounds");
    host_data_(i) = val;
}

double Array1D::get(size_t i) const {
    if (i >= size_) throw std::out_of_range("Index out of bounds");
    return host_data_(i);
}

void Array1D::sync_to_device() {
    Kokkos::deep_copy(data_, host_data_);
}

void Array1D::sync_to_host() {
    Kokkos::deep_copy(host_data_, data_);
}

void Array1D::from_list(const std::vector<double>& values) {
    if (values.size() != size_) {
        throw std::runtime_error("Size mismatch");
    }
    for (size_t i = 0; i < size_; ++i) {
        host_data_(i) = values[i];
    }
    sync_to_device();
}

std::vector<double> Array1D::to_list() const {
    std::vector<double> result(size_);
    for (size_t i = 0; i < size_; ++i) {
        result[i] = host_data_(i);
    }
    return result;
}

Array1D::ViewType& Array1D::view() {
    return data_;
}

const Array1D::ViewType& Array1D::view() const {
    return data_;
}

// ============================================================================
// Kernel implementation
// ============================================================================

double dot_product(const Array1D& a, const Array1D& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Size mismatch in dot_product");
    }

    auto va = a.view();
    auto vb = b.view();
    size_t n = a.size();
    double result = 0.0;

    Kokkos::parallel_reduce("dot_product", n,
        KOKKOS_LAMBDA(const int i, double& lsum) {
            lsum += va(i) * vb(i);
        },
        result
    );

    return result;
}
