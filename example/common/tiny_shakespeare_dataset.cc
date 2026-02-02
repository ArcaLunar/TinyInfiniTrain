#include "example/common/tiny_shakespeare_dataset.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace {
using DataType = infini_train::DataType;
using TinyShakespeareType = TinyShakespeareDataset::TinyShakespeareType;
using TinyShakespeareFile = TinyShakespeareDataset::TinyShakespeareFile;

const std::unordered_map<int, TinyShakespeareType> kTypeMap = {
    {20240520, TinyShakespeareType::kUINT16}, // GPT-2
    {20240801, TinyShakespeareType::kUINT32}, // LLaMA 3
};

const std::unordered_map<TinyShakespeareType, size_t> kTypeToSize = {
    {TinyShakespeareType::kUINT16, 2},
    {TinyShakespeareType::kUINT32, 4},
};

const std::unordered_map<TinyShakespeareType, DataType> kTypeToDataType = {
    {TinyShakespeareType::kUINT16, DataType::kUINT16},
    {TinyShakespeareType::kUINT32, DataType::kINT32},
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    /* =================================== 作业 ===================================
       TODO：实现二进制数据集文件解析
       文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | DATA (tokens)                        |
    | magic(4B) | version(4B) | num_toks(4B) | reserved(1012B) | token数据           |
    ----------------------------------------------------------------------------------
       =================================== 作业 =================================== */
    TinyShakespeareFile file;

    std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open file: " << path;

    // skip 8 bytes of magic and version
    const auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
    uint32_t magic = BytesToType<uint32_t>(header, 0);
    file.type = TinyShakespeareType::kUINT16; // default to GPT-2

    // get number of tokens
    uint32_t num_toks = BytesToType<uint32_t>(header, 8);

    // read token data
    size_t type_size = kTypeToSize.at(file.type);
    size_t data_size = num_toks * type_size;

    file.dims = {static_cast<int64_t>(num_toks / sequence_length), static_cast<int64_t>(sequence_length)};
    file.tensor = infini_train::Tensor(file.dims, DataType::kINT64);
    auto dst = reinterpret_cast<int64_t *>(file.tensor.DataPtr());

    if (file.type == TinyShakespeareType::kUINT16) {
        std::vector<uint16_t> buffer(num_toks);
        ifs.read(reinterpret_cast<char *>(buffer.data()), data_size);
        for (size_t i = 0; i < num_toks; ++i) { dst[i] = static_cast<int64_t>(buffer[i]); }
    } else if (file.type == TinyShakespeareType::kUINT32) {
        std::vector<uint32_t> buffer(num_toks);
        ifs.read(reinterpret_cast<char *>(buffer.data()), data_size);
        for (size_t i = 0; i < num_toks; ++i) { dst[i] = static_cast<int64_t>(buffer[i]); }
    } else {
        LOG(FATAL) << "Unsupported TinyShakespeareType.";
    }

    return file;
}

} // namespace

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length)
    : text_file_(ReadTinyShakespeareFile(filepath, sequence_length)), sequence_length_(sequence_length),
      sequence_size_in_bytes_(sequence_length * kTypeToSize.at(text_file_.type)), num_samples_(text_file_.dims[0]) {}

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
    CHECK_LT(idx, text_file_.dims[0] - 1);
    std::vector<int64_t> dims = std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());
    // x: (seq_len), y: (seq_len) -> stack -> (bs, seq_len) (bs, seq_len)
    return {std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_, dims),
            std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_ + sizeof(int64_t),
                                                   dims)};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }
