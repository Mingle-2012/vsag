#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "logger.h"
#include "simd/basic_func.h"
#include "spdlog/spdlog.h"
#include "vsag/vsag.h"

using namespace vsag;

template <typename T>
std::tuple<std::vector<T>, int64_t, int64_t>
read_vecs(const std::string& filename) {
    std::ifstream is(filename, std::ios::binary);
    if (!is.good()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return { {}, 0, 0 };
    }
    std::vector<T> data;
    is.seekg(0, std::ios::end);
    size_t size = is.tellg();
    is.seekg(0, std::ios::beg);
    unsigned dim;
    is.read(reinterpret_cast<char*>(&dim), sizeof(unsigned int));
    unsigned line = sizeof(T) * dim + sizeof(unsigned int);
    unsigned N = size / line;
    data.resize(N * dim);
    for (unsigned i = 0; i < N; ++i) {
        is.seekg(sizeof(unsigned int), std::ios::cur);
        is.read(reinterpret_cast<char*>(data.data() + i * dim), sizeof(T) * dim);
    }
    is.close();
    std::cout << "Read " << N << " vectors of dimension " << dim << " from file " << filename << std::endl;
    return { data, dim, N };
}

static const double THRESHOLD_ERROR = 2e-6;

static float
get_recall(const float* distances,
           const float* ground_truth_distances,
           size_t recall_num,
           size_t top_k) {
    std::vector<float> gt_distances(ground_truth_distances, ground_truth_distances + top_k);
    std::sort(gt_distances.begin(), gt_distances.end());
    float threshold = gt_distances[top_k - 1];
    size_t count = 0;
    for (size_t i = 0; i < recall_num; ++i) {
        if (distances[i] <= threshold + THRESHOLD_ERROR) {
            ++count;
        }
    }
    return static_cast<float>(count) / static_cast<float>(top_k);
}

constexpr static const char* search_param_hgraph = R"(
        {{
            "hgraph": {{
                "ef_search": {}
            }}
        }})";

constexpr static const char* search_param_hnsw = R"(
        {{
            "hnsw": {{
                "ef_search": {}
            }}
        }})";

constexpr static const char* search_param_diskann = R"(
{{
    "diskann": {{
        "ef_search": {},
        "beam_search": 4,
        "io_limit": 50
    }}
}}
)";

void
test_search_performance(const DatasetPtr& dataset,
                        const IndexPtr& index,
                        const std::string &search_param_json,
                        const std::string &query,
                        const std::string &gt,
                        int k = 10) {
    logger::info("Start testing search performance: {}", index->GetStats());
    auto [query_vectors, dim, num_queries] = read_vecs<float>(query);
    auto [gt_vectors, gt_dim, num_gt] = read_vecs<int>(gt);
    auto search_L = {20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 500, 600, 700, 800};
    for (auto L : search_L) {
        int round = 3;
        auto search_param = fmt::format(search_param_json, L, false);
        float qps = 0.0f, recall = 0.0f;
        for (int x = 0 ; x < round; ++x) {
            double time_cost_strong = 0.0;
            float correct = 0;
            for (int i = 0; i < num_queries; ++i) {
                auto q = Dataset::Make();
                q->Dim(dim)->Float32Vectors(query_vectors.data() + i * dim)->NumElements(1)->Owner(false);
                auto st = std::chrono::high_resolution_clock::now();
                auto qr = index->KnnSearch(q, k, search_param);
                auto ed = std::chrono::high_resolution_clock::now();
                time_cost_strong += std::chrono::duration<double>(ed - st).count();

                auto distance_func = [](const void* query1, const void* query2, const void* qty_ptr) -> float {
                    return std::sqrt(vsag::L2Sqr(query1, query2, qty_ptr));
                };
                auto gt_distances = std::shared_ptr<float[]>(new float[k]);
                auto distances = std::shared_ptr<float[]>(new float[k]);
                for (int j = 0; j < k; ++j) {
                    distances[j] = distance_func(query_vectors.data() + i * dim, dataset->GetFloat32Vectors() + qr.value()->GetIds()[j] * dim, &dim);
                    gt_distances[j] = distance_func(query_vectors.data() + i * dim, dataset->GetFloat32Vectors() + gt_vectors[i * gt_dim + j] * dim, &dim);
                }
                auto val = get_recall(distances.get(), gt_distances.get(), k, k);
                correct += val;
            }
            recall = std::max(recall, correct / static_cast<float>(num_queries));
            qps = std::max(qps, static_cast<float>(num_queries) / static_cast<float>(time_cost_strong));
        }
        logger::info("L = {}, Recall = {}, QPS = {}", L, recall, qps);
    }
}