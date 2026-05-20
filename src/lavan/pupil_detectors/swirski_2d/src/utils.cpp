// utils.cpp — deterministic ``random`` functions used by the Swirski
// 2D RANSAC step. The seeded overload reseeds a local engine; the
// unseeded one shares a translation-unit-local ``std::mt19937`` whose
// state persists across calls within the process.

#include "swirski_2d/utils.hpp"

#include <random>

namespace lavan::swirski_2d {

static std::mt19937 static_gen;

int random(int min, int max)
{
    std::uniform_int_distribution<> distribution(min, max);
    return distribution(static_gen);
}

int random(int min, int max, unsigned int seed)
{
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> distribution(min, max);
    return distribution(gen);
}

}  // namespace lavan::swirski_2d
