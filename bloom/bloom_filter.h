#ifndef BLOOM_FILTER_H
#define BLOOM_FILTER_H

#include <openssl/md5.h>

#include <bitset>
#include <cstddef>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

class BloomFilter {
   public:
    BloomFilter(size_t hash_func_count = 4)
        : hash_function_count(hash_func_count),
          _object_count(0),
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
    void clear() {
        _bloom_filter_store.reset();
        _object_count = 0;
    }
    size_t object_count() const { return _object_count; }
    size_t empty() const { return object_count() == 0; }

   private:
    // size of the md5 hash result fixed at 16 bytes
    static constexpr size_t md5_result_size_bytes = 16;
    // size of the bloom filter state (2^16)
    static constexpr size_t bloom_filter_store_size = 65536;
    // set to 2 so all bloom filter can be indexed (2^16 values)
    static constexpr size_t bytes_per_hash_function = 2;

    void hash(const std::string& value) const {
        const unsigned char* const md5_input_value =
            reinterpret_cast<const unsigned char*>(value.data());
        const size_t md5_input_length = value.length();
        MD5(md5_input_value, md5_input_length, md5_hash_result_buffer.get());
    }

    // number of hash functions to use when hashing objects
    const size_t hash_function_count;
    std::bitset<bloom_filter_store_size> _bloom_filter_store;
    size_t _object_count;
    const std::unique_ptr<unsigned char[]> md5_hash_result_buffer;
};
#endif
