#include <iostream>

#include "index/hnsw.h"
#include "index/hnsw_zparameters.h"
#include "index/index_common_param.h"
#include "vsag/vsag.h"

using namespace vsag;

std::vector<float> read_vecs(std::string filename) {
    std::ifstream is(filename, std::ios::binary);
    if (!is.good()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return {};
    }
    std::vector<float> data;
    is.seekg(0, std::ios::end);
    size_t size = is.tellg();
    is.seekg(0, std::ios::beg);
    unsigned dim;
    is.read(reinterpret_cast<char*>(&dim), sizeof(unsigned int));
    unsigned line = sizeof(float) * dim + sizeof(unsigned int);
    unsigned N = size / line;
    data.resize(N * dim);
    for (unsigned i = 0; i < N; ++i) {
        is.seekg(sizeof(unsigned int), std::ios::cur);
        is.read(reinterpret_cast<char*>(data.data() + i * dim), sizeof(float) * dim);
    }
    is.close();
    std::cout << "Read " << N << " vectors of dimension " << dim << " from file " << filename << std::endl;
    return data;
}

HnswParameters
parse_hnsw_params(IndexCommonParam index_common_param) {
    auto build_parameter_json = R"(
        {
            "max_degree": 12,
            "ef_construction": 100
        }
    )";
    nlohmann::json parsed_params = nlohmann::json::parse(build_parameter_json);
    return HnswParameters::FromJson(parsed_params, index_common_param);
}

void test_hnsw() {
    int64_t dim = 128;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = allocator;

    auto hnsw_obj = parse_hnsw_params(common_param);
    hnsw_obj.max_degree = 12;
    hnsw_obj.ef_construction = 100;
    auto index = std::make_shared<HNSW>(hnsw_obj, common_param);

    std::vector<int64_t> ids(1);
    int64_t incorrect_dim = 63;
    std::vector<float> vectors(incorrect_dim);

    auto dataset = Dataset::Make();
    dataset->Dim(incorrect_dim)
        ->NumElements(1)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);

}

int main() {
    std::cout << "Hello, VSAG!" << std::endl;

    auto base = read_vecs("/root/mount/dataset/sift/learn.fvecs");
    auto query = read_vecs("/root/mount/dataset/sift/query.fvecs");
    auto gt = read_vecs("/root/mount/dataset/sift/groundtruth.ivecs");
    if (base.empty() || query.empty() || gt.empty()) {
        std::cerr << "Failed to read dataset files." << std::endl;
        return -1;
    }

    return 0;
}