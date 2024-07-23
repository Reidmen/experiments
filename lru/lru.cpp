#include "lru.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <new>
#include <string>

LRU::LRU() {}

LRU::~LRU() {}

bool LRU::empty() { return std::empty(lookup_map); }

unsigned int LRU::size() { return std::size(lookup_map); }

bool LRU::get(const std::string &key, std::string &result) {
  const std::string *value_ptr = get_by_pointer(key);
  if (nullptr == value_ptr) {
    // key not found in cache
    return false;
  }

  result = *value_ptr;

  return true;
}

LRU::lru_list_type::iterator LRU::most_recent_list_node() {
  return recently_used_list.begin();
}

const std::string *LRU::get_by_pointer(const std::string &key) {
  lru_map_type::iterator map_element_iterator = lookup_map.find(key);
  if (lookup_map.cend() == map_element_iterator) {
    // key not found
    return nullptr;
  }
  try {
    visit_list_node(map_element_iterator->second);
  } catch (const std::bad_alloc &) {
    cleanup();
    return nullptr;
  }

  // update the lookup_map to the point of newest node
  map_element_iterator->second = most_recent_list_node();
  // return a pointer to the value of the newest node
  return &most_recent_list_node()->second;
}

void LRU::put(const std::string &key, const std::string &value) {
  lru_map_type::iterator map_element_iterator = lookup_map.find(key);
  if (lookup_map.end() == map_element_iterator) {
    // element has not been seen before, add it to the list
    try {
      recently_used_list.emplace_front(key, value);
    } catch (const std::bad_alloc &) {
      cleanup();
      return;
    }
    // also add element and the list to the lookup_map
    try {
      lookup_map.emplace(key, recently_used_list.begin());
    } catch (const std::bad_alloc &) {
      recently_used_list.erase(recently_used_list.begin());
      cleanup();
      return;
    }
  } else {
    // element has been seen before, move to the corresponding list node
    lru_list_type::iterator list_node_iterator = map_element_iterator->second;

    try {
      visit_list_node_new_value(list_node_iterator, value);
    } catch (const std::bad_alloc &) {
      cleanup();
      return;
    }

    // within the lookup_map, update the list node iterator
    map_element_iterator->second = recently_used_list.begin();
  }
}

void LRU::visit_list_node(lru_list_type::iterator &node_iterator) {
  const std::string &key_str = node_iterator->first;
  const std::string &value_str = node_iterator->second;

  // create new node at front
  recently_used_list.emplace_front(
      lru_node_type(std::move(key_str), std::move(value_str)));
  recently_used_list.erase(node_iterator);
}

void LRU::visit_list_node_new_value(lru_list_type::iterator &node_iterator,
                                    const std::string &value) {
  const std::string &key_str = node_iterator->first;
  recently_used_list.emplace_front(
      lru_node_type(std::move(key_str), std::move(value)));

  recently_used_list.erase(node_iterator);
}

void LRU::delete_last() {
  const lru_node_type &last_node = recently_used_list.back();
  const std::string &key = last_node.first;

  lookup_map.erase(key);
  recently_used_list.pop_back();
}

void LRU::cleanup() {
  if (std::empty())
    return;

  const size_t amount_to_remove =
      std::max(static_cast<size_t>(size() * 0.1), 1u);

  for (size_t i = 0; i < amount_to_remove; i++) {
    delete_last();
  }
}
