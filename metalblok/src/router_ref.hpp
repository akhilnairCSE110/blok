#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace blade {

struct RoutedExpert {
    uint32_t id = 0;
    float weight = 0.0f;
};

// Exact DeepSeek-V3/R1 noaux_tc routing. Corrected scores choose experts;
// uncorrected sigmoid probabilities determine mixture weights.
inline bool route_grouped_sigmoid(const float* logits, const float* bias,
                                  uint32_t n_experts, uint32_t top_k,
                                  uint32_t n_groups, uint32_t top_groups,
                                  float scale, bool normalize,
                                  std::vector<RoutedExpert>& out) {
    if (!logits || !bias || n_experts == 0 || top_k == 0 ||
        n_groups == 0 || n_experts % n_groups != 0 ||
        top_k > n_experts || top_groups == 0 || top_groups > n_groups) {
        return false;
    }
    const uint32_t per_group = n_experts / n_groups;
    std::vector<float> prob(n_experts), corrected(n_experts);
    std::vector<std::pair<float, uint32_t>> group_scores;
    group_scores.reserve(n_groups);

    for (uint32_t e = 0; e < n_experts; ++e) {
        prob[e] = 1.0f / (1.0f + std::exp(-logits[e]));
        corrected[e] = prob[e] + bias[e];
    }
    for (uint32_t g = 0; g < n_groups; ++g) {
        float first = -std::numeric_limits<float>::infinity();
        float second = first;
        for (uint32_t j = 0; j < per_group; ++j) {
            float v = corrected[g * per_group + j];
            if (v > first) { second = first; first = v; }
            else if (v > second) second = v;
        }
        group_scores.emplace_back(first + second, g);
    }
    auto higher = [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    };
    std::sort(group_scores.begin(), group_scores.end(), higher);
    std::vector<uint8_t> keep(n_groups, 0);
    for (uint32_t i = 0; i < top_groups; ++i) keep[group_scores[i].second] = 1;

    std::vector<std::pair<float, uint32_t>> candidates;
    candidates.reserve(top_groups * per_group);
    for (uint32_t e = 0; e < n_experts; ++e) {
        if (keep[e / per_group]) candidates.emplace_back(corrected[e], e);
    }
    std::sort(candidates.begin(), candidates.end(), higher);
    if (candidates.size() < top_k) return false;

    float denom = 0.0f;
    for (uint32_t i = 0; i < top_k; ++i) denom += prob[candidates[i].second];
    out.resize(top_k);
    for (uint32_t i = 0; i < top_k; ++i) {
        uint32_t e = candidates[i].second;
        float w = prob[e];
        if (normalize && denom > 0.0f) w /= denom;
        out[i] = {e, w * scale};
    }
    return true;
}

} // namespace blade
