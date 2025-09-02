#include <cmath>
#include <iostream>

#include "util.hpp"
#include "index/hnsw.h"
#include "vsag/vsag.h"

using namespace vsag;

void test_remove(const std::string& index_type,
            const std::string& search_param,
            const std::string& base,
            const std::string& query,
            const std::vector<std::string>& gt_files,
            int num_threads = 64) {
    auto [vectors, dim, num_vectors] = read_vecs<float>(base);
    std::string build_param = R"(
    {{
        "dtype": "float32",
        "metric_type": "l2",
        "dim": {},
        "index_param": {{
            "max_degree": 32,
            "ef_construction": 200,
            "support_remove": true
        }}
    }}
    )";
    build_param = fmt::format(build_param, dim);

    Options::Instance().set_num_threads_building(num_threads);

    std::vector<int64_t> ids(num_vectors);
    std::iota(ids.begin(), ids.end(), 0);

    auto build_num = static_cast<int64_t>((double)num_vectors * 0.9);
    size_t remain_num = num_vectors - build_num;

    auto dataset_build = Dataset::Make();
    dataset_build->Dim(dim)
        ->NumElements(build_num)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);

    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), vsag::Engine::CreateThreadPool(num_threads).value());
    vsag::Engine engine(&resource);

    auto index = engine.CreateIndex(index_type, build_param).value();
    logger::info("Start building {} index with 90% data", index_type);
    auto start = std::chrono::high_resolution_clock::now();
    if (auto build_result = index->Build(dataset_build); build_result.has_value()) {
        logger::info("After Build(), Index {} contains: {}", index_type, index->GetNumElements());
    } else {
        logger::error("Failed to build index because {}", build_result.error().message);
        return;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    logger::info("Build index time cost: {} seconds", diff.count());

    auto query_dataset = Dataset::Make();
    auto [query_vectors, query_dim, num_queries] = read_vecs<float>(query);
    query_dataset->Dim(query_dim)
        ->NumElements(num_queries)
        ->Float32Vectors(query_vectors.data())
        ->Owner(false);
    test_search_performance_with_ids(dataset_build, index, search_param, query_dataset, {20, 50, 80});

    size_t step = std::max<size_t>(1, num_vectors / 100);
    logger::info("Sliding step is set to 1% of total data, which is {} vectors", step);

    for (size_t offset = 0, gt_idx = 1; offset < remain_num; offset += step, ++gt_idx) {
        int64_t insert_num = (int64_t)std::min(step, remain_num - offset);

        auto dataset_insert = Dataset::Make();
        dataset_insert->Dim(dim)
            ->NumElements(insert_num)
            ->Ids(ids.data() + build_num + offset)
            ->Float32Vectors(vectors.data() + (build_num + offset) * dim)
            ->Owner(false);

        if (auto insert_result = index->Add(dataset_insert); insert_result.has_value()) {
            logger::info("After Add(), Index contains: {}", index->GetNumElements());
        }

        for (size_t j = 0; j < insert_num; ++j) {
            int64_t remove_id = ids[offset + j];
            if (auto remove_result = index->Remove(remove_id); remove_result.has_value() && !remove_result.value()) {
                logger::error("Failed to remove because {}", remove_result.error().message);
            }
        }
        logger::info("After Remove(), Index contains: {}", index->GetNumElements());

        auto dataset_now = Dataset::Make();
        dataset_now->Dim(dim)
            ->NumElements(index->GetNumElements())
            ->Ids(ids.data() + offset + insert_num)
            ->Float32Vectors(vectors.data() + (offset + insert_num) * dim)
            ->Owner(false);

        test_search_performance_with_ids(dataset_now, index, search_param, query_dataset, {20, 50, 80});
    }

    engine.Shutdown();

}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <base_data> <query_data> <gt_path_prefix>" << std::endl;
        return -1;
    }

    redirect_output("/root/code/algotests/vsag-test/exp/logs/sift100k_mannual_Ls.log");

    auto base = argv[1];
    auto query = argv[2];
    std::vector<std::string> gt_files = {};
    if (argc > 3) {
        for (int i = 0; i <= 10; ++i){
            std::string prefix(argv[3]);
            gt_files.emplace_back(prefix + "/gt_" + std::to_string(i) + ".ivecs");
        }
    } else {
        gt_files.resize(11, "");
    }
    test_remove("hgraph", search_param_hgraph, base, query, gt_files);

}
