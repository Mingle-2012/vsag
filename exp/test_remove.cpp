#include <cmath>
#include <iostream>

#include "util.hpp"
#include "index/hnsw.h"
#include "vsag/vsag.h"

using namespace vsag;

void test_remove(const std::string& index_type = "hnsw",
            const std::string& build_param = "",
            const std::string& search_param = "") {
    std::string base = "/root/datasets/sift/10k/sift_base.fvecs";
    auto [vectors, dim, num_vectors] = read_vecs<float>(base);

    std::vector<int64_t> ids(num_vectors);
    std::iota(ids.begin(), ids.end(), 0);

    auto build_num = static_cast<int64_t>((double)num_vectors * 0.99);
    size_t remain_num = num_vectors - build_num;

    auto dataset_build = Dataset::Make();
    dataset_build->Dim(dim)
        ->NumElements(build_num)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);

    auto index = vsag::Factory::CreateIndex(index_type, build_param).value();
    logger::info("Start building {} index with 99% data", index_type);
    if (auto build_result = index->Build(dataset_build); build_result.has_value()) {
        std::cout << "After Build(), Index " << index_type
                  << " contains: " << index->GetNumElements() << std::endl;
    } else {
        std::cerr << "Failed to build index: "
                  << build_result.error().message << std::endl;
        exit(-1);
    }

    // TODO: test search here

    size_t step = std::max<size_t>(1, num_vectors / 1000);
    std::cout << "Sliding step = " << step << " vectors" << std::endl;

    for (size_t offset = 0; offset < remain_num; offset += step) {
        int64_t insert_num = (int64_t)std::min(step, remain_num - offset);

        auto dataset_insert = Dataset::Make();
        dataset_insert->Dim(dim)
            ->NumElements(insert_num)
            ->Ids(ids.data() + build_num + offset)
            ->Float32Vectors(vectors.data() + (build_num + offset) * dim)
            ->Owner(false);

        if (auto insert_result = index->Add(dataset_insert); insert_result.has_value()) {
            std::cout << "Inserted " << insert_num << " vectors, total = "
                      << index->GetNumElements() << std::endl;
        } else {
            std::cerr << "Insert failed: " << insert_result.error().message << std::endl;
        }

        for (size_t j = 0; j < insert_num; ++j) {
            int64_t remove_id = ids[offset + j];
            if (auto remove_result = index->Remove(remove_id); remove_result.has_value()) {
                std::cout << "Removed id " << remove_id << std::endl;
            } else {
                std::cerr << "Remove failed: " << remove_result.error().message << std::endl;
            }
        }

        // TODO: Test search after each insertion and deletion
    }


}

int main(){
    test_remove();
}