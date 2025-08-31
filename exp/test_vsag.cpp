#include <cmath>
#include <iostream>

#include "util.hpp"
#include "index/hnsw.h"
#include "vsag/vsag.h"

using namespace vsag;

void
test_hnsw(const std::string &base, const std::string &query, const std::string &gt) {
    auto [vectors, dim, num_vectors] = read_vecs<float>(base);

    std::vector<int64_t> ids(num_vectors);
    std::iota(ids.begin(), ids.end(), 0);

    auto dataset = Dataset::Make();
    dataset->Dim(dim)
        ->NumElements(num_vectors)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);

    auto hnsw_build_paramesters = fmt::format(R"(
        {{
            "dtype": "float32",
            "metric_type": "l2",
            "dim": {},
            "hnsw": {{
                "max_degree": 26,
                "ef_construction": 100
            }}
        }}
        )", dim);
    auto index = vsag::Factory::CreateIndex("hnsw", hnsw_build_paramesters).value();
    std::cout << "Start building HNSW index" << std::endl;
    if (auto build_result = index->Build(dataset); build_result.has_value()) {
        std::cout << "After Build(), Index HNSW contains: " << index->GetNumElements() << std::endl;
    } else {
        std::cerr << "Failed to build index: " << build_result.error().message << std::endl;
        exit(-1);
    }

    auto [query_vectors, query_dim, num_queries] = read_vecs<float>(query);
    auto [gt_vectors, gt_dim, num_gt] = read_vecs<int>(gt);

    auto search_L = {20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
    // auto search_L = {20, 200, 1000};
    auto k = 10;
    auto gt_k = std::min((int)gt_dim, k);
    for (auto L : search_L) {
        int round = 3;
        auto search_param = fmt::format(search_param_hnsw, L);
        float qps = 0.0f, recall = 0.0f;

        for (int x = 0 ; x < round; ++x) {
            double time_cost_strong = 0.0;
            int correct = 0;
            for (int i = 0; i < num_queries; ++i) {
                auto q = Dataset::Make();
                q->Dim(dim)->Float32Vectors(query_vectors.data() + i * dim)->NumElements(1)->Owner(false);
                auto st = std::chrono::high_resolution_clock::now();
                auto qr = index->KnnSearch(q, k, search_param);
                auto ed = std::chrono::high_resolution_clock::now();
                time_cost_strong += std::chrono::duration<double>(ed - st).count();

                std::set<uint64_t> gt_ids(gt_vectors.begin() + i * gt_dim, gt_vectors.begin() + i * gt_dim + gt_k);
                for (int j = 0; j < k; ++j) {
                    if (const auto& id = qr.value()->GetIds()[j]; gt_ids.find(id) != gt_ids.end()) {
                        ++correct;
                    }
                }
            }
            recall = std::max(recall, static_cast<float>(correct) / static_cast<float>(num_queries * k));
            qps = std::max(qps, static_cast<float>(num_queries) / static_cast<float>(time_cost_strong));
        }
        std::cout << "L = " << L << ", Recall = " << recall << ", QPS = " << qps << std::endl;
    }
}

void
test_hgraph(const std::string &base, const std::string &query, const std::string &gt) {
    auto [vectors, dim, num_vectors] = read_vecs<float>(base);

    std::vector<int64_t> ids(num_vectors);
    std::iota(ids.begin(), ids.end(), 0);

    auto dataset = Dataset::Make();
    dataset->Dim(dim)
        ->NumElements(num_vectors)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);

    std::string hgraph_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "max_degree": 32,
            "ef_construction": 200
        }
    }
    )";

    Options::Instance().set_num_threads_building(20);
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), vsag::Engine::CreateThreadPool(20).value());
    vsag::Engine engine(&resource);

    auto index = engine.CreateIndex("hgraph", hgraph_build_parameters).value();
    logger::debug("start building hgraph index");
    auto start = std::chrono::high_resolution_clock::now();
    index->Add(dataset);
    auto end = std::chrono::high_resolution_clock::now();
    auto build_time = std::chrono::duration<double>(end - start).count();
    logger::debug("hgraph index built in {} seconds", build_time);

    auto query_dataset = Dataset::Make();
    auto [query_vectors, query_dim, num_queries] = read_vecs<float>(query);
    query_dataset->Dim(query_dim)
        ->NumElements(num_queries)
        ->Float32Vectors(query_vectors.data())
        ->Owner(false);

    test_search_performance(dataset, index, search_param_hgraph, query_dataset, gt);
    engine.Shutdown();
}

void
test_diskann(const std::string &base, const std::string &query, const std::string &gt) {
    auto [vectors, dim, num_vectors] = read_vecs<float>(base);

    std::vector<int64_t> ids(num_vectors);
    std::iota(ids.begin(), ids.end(), 0);

    auto dataset = Dataset::Make();
    dataset->Dim(dim)
        ->NumElements(num_vectors)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);

    auto diskann_build_paramesters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "diskann": {
            "max_degree": 16,
            "ef_construction": 200,
            "pq_sample_rate": 0.5,
            "pq_dims": 9,
            "use_pq_search": true,
            "use_async_io": true,
            "use_bsa": true
        }
    }
    )";
    auto index = vsag::Factory::CreateIndex("diskann", diskann_build_paramesters).value();
    std::cout << "Start building DiskANN index" << std::endl;
    if (auto build_result = index->Build(dataset); build_result.has_value()) {
        std::cout << "After Build(), Index DiskANN contains: " << index->GetNumElements() << std::endl;
    } else {
        std::cerr << "Failed to build index: " << build_result.error().message << std::endl;
        exit(-1);
    }

    auto query_dataset = Dataset::Make();
    auto [query_vectors, query_dim, num_queries] = read_vecs<float>(query);
    query_dataset->Dim(query_dim)
        ->NumElements(num_queries)
        ->Float32Vectors(query_vectors.data())
        ->Owner(false);

    test_search_performance(dataset, index, search_param_hgraph, query_dataset, gt);
}

int
main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <base> <query> <ground_truth>" << std::endl;
        return 1;
    }

    auto base = argv[1];
    auto query = argv[2];
    auto gt = argv[3];

//    test_hnsw(base, query, gt);

    test_hgraph(base, query, gt);

//    test_diskann(base, query, gt);

    return 0;
}