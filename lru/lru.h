#ifndef LRU_H
#define LRU_H
// https://github.com/daankolthof/LRU-memory-cache/blob/master/LRU-memory-cache/LRU.h
#include <list>
#include <string>
#include <unordered_map>
#include <vector>
// Base for Least-Recent-Used cache data structure
class LRU {
   public:
    using lru_node_type = std::pair<std::string, std::string>;
    using lru_list_type = std::list<lru_node_type>;
    using lru_map_type =
        std::unordered_map<std::string, lru_list_type::iterator>;

    LRU();

    // delete copy constructor as they will cause large memory copies
    LRU(const LRU&) = delete;
    LRU& operator=(LRU&) = delete;

    // move constructors, cache is moved to another LRU, not copied
    LRU(LRU&&) = default;
    LRU& operator=(LRU&&) = default;

    ~LRU();
    // query for an element in the cache by key
    // returns true if found and copy the string in the results field
    bool get(const std::string& key, std::string& result);
    // query for an element in the cache by key, returns
    // the pointer to the value string, else returns nullptr
    const std::string* get_by_pointer(const std::string& key);
    // insert a key and associated value pair into cache,
    // replacing previous valu  // insert a key and associated value pair into
    // cache, replacing previous valuee
    void put(const std::string& key, const std::string& value);

    bool empty() const { return lookup_map.empty(); };
    size_t size() const { return lookup_map.size(); };
    const lru_list_type& get_recent_used_list() const {
        return recently_used_list;
    }
    const lru_map_type& get_lookup_map() const { return lookup_map; }

   private:
    lru_list_type recently_used_list;
    lru_map_type lookup_map;
    // set a node as most recently used
    void visit_list_node(lru_list_type::iterator& node_iterator);
    // set a node as the most recently used and assigns a new value
    void visit_list_node_new_value(lru_list_type::iterator& node_iterator,
                                   const std::string& value);
    // delete least recently used element
    void delete_last();
    // remove 10% least recently used elements, freeing memory
    void cleanup();
    // return the most recent node
    lru_list_type::iterator most_recent_list_node();
};
#endif
