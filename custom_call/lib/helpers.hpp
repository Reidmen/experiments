#ifndef _CUSTOM_CALL_HELPERS_H_
#define _CUSTOM_CALL_HELPERS_H_

#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
namespace custom_helpers {
template <class To, class From>
typename std::enable_if<sizeof(To) == sizeof(From) &&
                            std::is_trivially_copyable<To>::value,
                        To>::type
bit_cast(const From &src) noexcept {
  static_assert(std::is_trivially_constructible<To>::value,
                "This implementation requires destination type to be trivially "
                "destructible.");
  To destination;
  memcpy(&destination, &src, sizeof(To));
  return destination;
}

template <typename T> std::string PackDescriptorsAsString(const T &descriptor) {
  return std::string(bit_cast<char const *>(&descriptor), sizeof(T));
}

template <typename T>
const T *UnpackDescriptor(const char *opaque, std::size_t opaque_length) {
  if (opaque_length != sizeof(T)) {
    throw std::runtime_error("Invalid opaque object size");
  }
  return bit_cast<const T *>(opaque);
}
} // namespace custom_helpers

#endif
