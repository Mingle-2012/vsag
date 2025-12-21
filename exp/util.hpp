#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "logger.h"
#include "safe_thread_pool.h"
#include "simd/basic_func.h"
#include "spdlog/sinks/stdout_sinks.h"
#include "spdlog/spdlog.h"
#include "typing.h"
#include "vsag/vsag.h"

using namespace vsag;

void redirect_output(const std::string& filename) {
    namespace fs = std::filesystem;
    fs::path file_path(filename);

    if (file_path.has_parent_path()) {
        try {
            fs::create_directories(file_path.parent_path());
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Failed to create directories: " << e.what() << std::endl;
            return;
        }
    }

    const int fd = ::open(filename.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd == -1) {
        perror("open");
        return;
    }

    ::dup2(fd, STDOUT_FILENO);
    ::dup2(fd, STDERR_FILENO);
    ::close(fd);

    auto sink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("log", sink);
    spdlog::set_default_logger(logger);
}

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
    logger::info("Read {} vectors of dimension {} from file {}", N, dim, filename);
    return { std::move(data), dim, N };
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

inline void
test_search_performance(const DatasetPtr& dataset,
                        const IndexPtr& index,
                        const std::string &search_param_json,
                        const DatasetPtr& query,
                        const std::string &gt = "",
                        const std::vector<int>& search_L = {20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 500, 600, 700, 800},
                        int k = 10,
                        int round = 3) {
    logger::info("Start testing search performance");
    auto dim = dataset->GetDim();
    auto query_dim = query->GetDim();
    if (dim != query_dim) {
        logger::error("dim of dataset({}) not equal to dim of query({})", dim, query_dim);
        return;
    }

    auto num_queries = query->GetNumElements();
    int64_t gt_dim = 0, num_gt = 0;
    std::shared_ptr<float[]> gt_distances;
    auto distance_func = [](const void* query1, const void* query2, const void* qty_ptr) -> float {
        return std::sqrt(vsag::L2Sqr(query1, query2, qty_ptr));
    };
    if (gt.empty()){
        logger::warn("gt file is empty, compute the ground truth by brute-force");
        num_gt = num_queries;
        gt_dim = k;
        gt_distances = std::shared_ptr<float[]>(new float[num_gt * gt_dim]);

#pragma omp parallel for schedule(dynamic)
        for (int i = 0 ; i < num_queries; ++i) {
            std::vector<float> vec_dists;
            vec_dists.reserve(k + 1);

            for (InnerIdType j = 0; j < dataset->GetNumElements(); ++j) {
                float dist = vsag::L2Sqr(query->GetFloat32Vectors() + i * dim,
                                         dataset->GetFloat32Vectors() + j * dim,
                                         &dim);
                vec_dists.emplace_back(dist);
                std::push_heap(vec_dists.begin(), vec_dists.end());

                if (vec_dists.size() > static_cast<size_t>(k)) {
                    std::pop_heap(vec_dists.begin(), vec_dists.end());
                    vec_dists.pop_back();
                }
            }
            std::sort_heap(vec_dists.begin(), vec_dists.end());
            memcpy(gt_distances.get() + i * k, vec_dists.data(), sizeof(float) * k);
        }
        logger::info("Compute ground truth by brute-force, num_gt = {}, gt_dim = {}", num_gt, gt_dim);
    }else{
        std::vector<int> gt_vectors;
        std::tie(gt_vectors, gt_dim, num_gt) = read_vecs<int>(gt);
        if (num_queries != num_gt) {
            logger::error("num_queries({}) not equal to num_gt({})", num_queries, num_gt);
            return;
        }

        gt_distances = std::shared_ptr<float[]>(new float[num_gt * k]);
        for (int i = 0 ; i < query->GetNumElements(); ++i) {
            for (int j = 0; j < k; ++j) {
                gt_distances[i * k + j] = distance_func(query->GetFloat32Vectors() + i * dim, dataset->GetFloat32Vectors() + gt_vectors[i * gt_dim + j] * dim, &dim);
            }
        }
        logger::info("Load ground truth from file {}, num_gt = {}, gt_dim = {}", gt, num_gt, gt_dim);
    }

    for (auto L : search_L) {
        auto search_param = fmt::format(search_param_json, L, false);
        float qps = 0.0f, recall = 0.0f;
        for (int x = 0 ; x < round; ++x) {
            double time_cost_strong = 0.0;
            float correct = 0;
            for (int i = 0; i < query->GetNumElements(); ++i) {
                auto q = Dataset::Make();
                q->Dim(dim)
                    ->Float32Vectors(query->GetFloat32Vectors() + i * dim)
                    ->NumElements(1)
                    ->Owner(false);
                auto st = std::chrono::high_resolution_clock::now();
                auto qr = index->KnnSearch(q, k, search_param);
                auto ed = std::chrono::high_resolution_clock::now();
                time_cost_strong += std::chrono::duration<double>(ed - st).count();

                auto distances = std::shared_ptr<float[]>(new float[k]);
                for (int j = 0; j < k; ++j) {
                    distances[j] = distance_func(query->GetFloat32Vectors() + i * dim, dataset->GetFloat32Vectors() + qr.value()->GetIds()[j] * dim, &dim);
                }
                auto val = get_recall(distances.get(), gt_distances.get() + i * k, k, k);
                correct += val;
            }
            recall = std::max(recall, correct / static_cast<float>(num_queries));
            qps = std::max(qps, static_cast<float>(num_queries) / static_cast<float>(time_cost_strong));
        }
        logger::info("L = {}, Recall = {}, QPS = {}", L, recall, qps);
    }
}

inline void
test_search_performance_with_ids(const DatasetPtr& dataset,
                        const IndexPtr& index,
                        const std::string &search_param_json,
                        const DatasetPtr& query,
                        const std::vector<int>& search_L = {20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 500, 600, 700, 800},
                        int k = 10,
                        int round = 3) {
    logger::info("Start testing search performance");
    auto dim = dataset->GetDim();
    auto query_dim = query->GetDim();
    if (dim != query_dim) {
        logger::error("dim of dataset({}) not equal to dim of query({})", dim, query_dim);
        return;
    }

    auto num_queries = query->GetNumElements();
    int64_t gt_dim = 0, num_gt = 0;
    std::shared_ptr<std::pair<float, int>[]> gt_pair;
    num_gt = num_queries;
    gt_dim = k;
    gt_pair = std::shared_ptr<std::pair<float, int>[]>(new std::pair<float, int>[num_gt * gt_dim]);
    {
#pragma omp parallel for schedule(dynamic)
        for (int i = 0 ; i < num_queries; ++i) {
            std::vector<std::pair<float, int>> vec_dists;
            vec_dists.reserve(k + 1);

            for (InnerIdType j = 0; j < dataset->GetNumElements(); ++j) {
                float dist = vsag::L2Sqr(query->GetFloat32Vectors() + i * dim,
                                         dataset->GetFloat32Vectors() + j * dim,
                                         &dim);
                vec_dists.emplace_back(dist, dataset->GetIds()[j]);
                std::push_heap(vec_dists.begin(), vec_dists.end());

                if (vec_dists.size() > static_cast<size_t>(k)) {
                    std::pop_heap(vec_dists.begin(), vec_dists.end());
                    vec_dists.pop_back();
                }
            }
            std::sort_heap(vec_dists.begin(), vec_dists.end());
            std::move(vec_dists.begin(), vec_dists.begin() + k, gt_pair.get() + i * k);
        }
    }

    for (auto L : search_L) {
        auto search_param = fmt::format(search_param_json, L, false);
        float qps = 0.0f, recall = 0.0f;
        for (int x = 0 ; x < round; ++x) {
            std::set<int> fail_ids;
            double time_cost_strong = 0.0;
            float correct = 0;
            for (int i = 0; i < query->GetNumElements(); ++i) {
                auto q = Dataset::Make();
                q->Dim(dim)
                    ->Float32Vectors(query->GetFloat32Vectors() + i * dim)
                    ->NumElements(1)
                    ->Owner(false);
                auto st = std::chrono::high_resolution_clock::now();
                auto qr = index->KnnSearch(q, k, search_param);
                auto ed = std::chrono::high_resolution_clock::now();
                time_cost_strong += std::chrono::duration<double>(ed - st).count();

//                std::vector<std::pair<float, int>> gt_distances(gt_pair.get(), gt_pair.get() + k);
//                std::sort(gt_distances.begin(), gt_distances.end());
//                float threshold = gt_distances[k - 1].first;
                size_t count = 0;
//                for (int j = 0; j < k; ++j) {
//                    auto distance = distance_func(query->GetFloat32Vectors() + i * dim, dataset->GetFloat32Vectors() + qr.value()->GetIds()[j] * dim, &dim);
//                    if (distance <= threshold + THRESHOLD_ERROR) {
//                        ++count;
//                    } else {
//                        fail_ids.emplace_back(qr.value()->GetIds()[j]);
//                    }
//                }
                std::unordered_set<InnerIdType> gt_set, found_set;
                for (int j = 0; j < k; ++j) {
                    gt_set.insert(gt_pair[i * k + j].second);
                }
                for (int j = 0 ; j < k ; ++j){
                    if (gt_set.count(qr.value()->GetIds()[j]) > 0){
                        ++count;
                        found_set.insert(qr.value()->GetIds()[j]);
                    }
                }
                for (int j = 0; j < k; ++j) {
                    if (found_set.count(gt_pair[i * k + j].second) == 0){
                        fail_ids.insert(gt_pair[i * k + j].second);
                    }
                }

                auto val =  static_cast<float>(count) / static_cast<float>(k);
                correct += val;
            }
            recall = std::max(recall, correct / static_cast<float>(num_queries));
            qps = std::max(qps, static_cast<float>(num_queries) / static_cast<float>(time_cost_strong));

            // std::cout << "fail ids: ";
            // for (auto& id : fail_ids){
            //     std::cout << id << ",";
            // }
            // std::cout << std::endl;

        }
        logger::info("L = {}, Recall = {}, QPS = {}", L, recall, qps);
    }
}
