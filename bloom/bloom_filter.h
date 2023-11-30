#ifndef BLOOM_FILTER_H
#define BLOOM_FILTER_H

#include <openssl/md5.h>

#include <cstddef>
#include <exception>
#include <memory>
#include <stdexcept>

class BloomFilter {
   public:
    BloomFilter(size_t hash_func_count = 4)
        : hash_function_count(hash_func_count),
          object_count(0),
          md5_hash_result_buffer(
              std::make_unique<unsigned char[]>(md5_result_size_bytes)) {
        if (hash_func_count == 0) {
            throw std::invalid_argument(
                "BloomFilter cannot be initialized with hash_func_count equal "
                "to 0");
        }
        if (md5_result_size_bytes <
            hash_function_count * bytes_per_hash_function) {
            throw std::invalid_argument(
                "BloomFilter cannot be initialized with hash_func_count too "
                "large");
        }
    }

   private:
    static constexpr size_t md5_result_size_bytes = 16;
    static constexpr size_t bloom_filter_store_size = 65536;
    static constexpr size_t bytes_per_hash_function = 2;

    const size_t hash_function_count;
    size_t object_count;
    const std::unique_ptr<unsigned char[]> md5_hash_result_buffer;
};
#endif
