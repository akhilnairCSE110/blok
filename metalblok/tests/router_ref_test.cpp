#include "router_ref.hpp"

#include <cmath>
#include <vector>

int main() {
    constexpr uint32_t N = 256;
    std::vector<float> logits(N, -8.0f), bias(N, 0.0f);
    // Make group 7 contain the global best corrected expert, but only one
    // strong member. Four groups with two strong members must win group
    // selection, proving this is not a global top-k implementation.
    bias[224] = 10.0f;
    for (uint32_t g = 0; g < 4; ++g) {
        bias[g * 32] = 6.0f - static_cast<float>(g) * 0.1f;
        bias[g * 32 + 1] = 6.0f - static_cast<float>(g) * 0.1f;
        logits[g * 32] = 1.0f;
        logits[g * 32 + 1] = 0.5f;
    }
    std::vector<blade::RoutedExpert> routed;
    if (!blade::route_grouped_sigmoid(logits.data(), bias.data(), N, 8, 8, 4,
                                      2.5f, true, routed)) return 1;
    if (routed.size() != 8) return 2;
    float sum = 0.0f;
    for (const auto& r : routed) {
        if (r.id >= 128 || r.id == 224) return 3;
        sum += r.weight;
    }
    if (std::fabs(sum - 2.5f) > 1e-5f) return 4;
    return 0;
}
