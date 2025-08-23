#include <iostream>

#include "index/hnsw.h"
#include "index/hnsw_zparameters.h"
#include "index/index_common_param.h"
#include "vsag/vsag.h"
#include "spdlog/spdlog.h"

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

HnswParameters
parse_hnsw_params(IndexCommonParam index_common_param) {
    auto build_parameter_json = R"(
        {
            "max_degree": 32,
            "ef_construction": 100
        }
    )";
    nlohmann::json parsed_params = nlohmann::json::parse(build_parameter_json);
    return HnswParameters::FromJson(parsed_params, index_common_param);
}

void
test_hnsw(const std::string &base, const std::string &query, const std::string &gt) {
    auto [vectors, dim, num_vectors] = read_vecs<float>(base);
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = allocator;

    auto hnsw_obj = parse_hnsw_params(common_param);
    hnsw_obj.max_degree = 32;
    hnsw_obj.ef_construction = 100;
    auto index = std::make_shared<HNSW>(hnsw_obj, common_param);
    index->InitMemorySpace();

    std::vector<int64_t> ids(num_vectors);
    std::iota(ids.begin(), ids.end(), 0);

    auto dataset = Dataset::Make();
    dataset->Dim(dim)
        ->NumElements(num_vectors)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(true);

    index->Build(dataset);

    auto [query_vectors, query_dim, num_queries] = read_vecs<float>(query);
    auto [gt_vectors, gt_dim, num_gt] = read_vecs<int>(gt);

    auto search_L = {20, 40, 60, 80, 100, 120, 140, 160, 180, 200};
    auto k = 10;
    for (auto L : search_L) {
        JsonType search_parameters{
                {"hnsw", {{"ef_search", L}}},
            };
        int correct = 0;
        for (int i = 0; i < num_queries; ++i) {
            auto q = Dataset::Make();
            q->Dim(dim)->Float32Vectors(query_vectors.data() + i * dim)->NumElements(1);
            auto qr = index->KnnSearch(q, k, search_parameters.dump());
            std::set<uint64_t> gt_ids(gt_vectors.begin() + i * gt_dim, gt_vectors.begin() + i * gt_dim + k);

            for (int j = 0; j < k; ++j) {
                if (const auto& id = qr.value()->GetIds()[j]; gt_ids.find(id) != gt_ids.end()) {
                    correct++;
                }
            }
        }
        float recall = correct / static_cast<float>(num_queries * k);
        std::cout << "L = " << L << ", Recall = " << recall << std::endl;
    }

}

constexpr static const char* search_param_tmp = R"(
        {{
            "hgraph": {{
                "ef_search": {},
                "use_extra_info_filter": {}
            }}
        }})";

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
            "base_quantization_type": "sq8",
            "max_degree": 32,
            "ef_construction": 200
        }
    }
    )";

    auto allocator = Engine::CreateDefaultAllocator();
    Resource resource(allocator, nullptr);
    Engine engine(&resource);
    auto index = engine.CreateIndex("hgraph", hgraph_build_parameters).value();
    logger::debug("start building hgraph index");
    index->Add(dataset);
    logger::debug("hgraph index built successfully");

    logger::debug("start knn search");
    auto [query_vectors, query_dim, num_queries] = read_vecs<float>(query);
    logger::debug("query vectors read");
    auto [gt_vectors, gt_dim, num_gt] = read_vecs<int>(gt);
    logger::debug("ground truth vectors read");


    auto search_L = {20, 40, 60, 80, 100, 120, 140, 160, 180, 200};
    // auto search_L = {20, 200, 1000};
    auto k = 10;
    for (auto L : search_L) {
        int round = 3;
        auto search_param = fmt::format(search_param_tmp, L, false);
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

                std::set<uint64_t> gt_ids(gt_vectors.begin() + i * gt_dim, gt_vectors.begin() + i * gt_dim + k);
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

    engine.Shutdown();
}

int
main() {
    std::cout << "Hello, VSAG!" << std::endl;

    // auto base = "/root/mount/dataset/sift/learn.fvecs";
    // auto query = "/root/mount/dataset/sift/query.fvecs";
    // auto gt = "/root/mount/dataset/sift/groundtruth.ivecs";

    auto base = "/root/mount/dataset/siftsmall/siftsmall_base.fvecs";
    auto query = "/root/mount/dataset/siftsmall/siftsmall_query.fvecs";
    auto gt = "/root/mount/dataset/siftsmall/siftsmall_groundtruth.ivecs";

    test_hgraph(base, query, gt);

    //试试hnsw呢？重新计算一下gt呢？

    return 0;
}