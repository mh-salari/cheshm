// Binding-side helper: wrap a ``std::vector<double>`` as a numpy 1-D
// ndarray that owns its underlying buffer via a nanobind capsule.

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace cheshm::Daugman
{

inline nanobind::ndarray<nanobind::numpy, double, nanobind::ndim<1>> nb_to_numpy(std::vector<double> data)
{
    auto owner = std::make_unique<std::vector<double>>(std::move(data));
    double* ptr = owner->data();
    const std::size_t shape[1] = {owner->size()};
    nanobind::capsule cap(owner.release(), [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    return nanobind::ndarray<nanobind::numpy, double, nanobind::ndim<1>>(ptr, 1, shape, cap);
}

} // namespace cheshm::Daugman
