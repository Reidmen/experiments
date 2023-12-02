// check cpputest -> https://cpputest.github.io/manual.html#assertions
#include <CppUTest/TestHarness.h>
#include <CppUTest/UtestMacros.h>

#include "../lru.h"
void check_not_nullptr(const std::string *ptr_to_check) {
    CHECK_TRUE(ptr_to_check != nullptr);
}

void check_nullptr(const std::string *ptr_to_check) {
    CHECK_TRUE(ptr_to_check == nullptr);
}

void check_is_empty(LRU lru) {
    CHECK_TRUE(lru.empty());
    CHECK_EQUAL(0u, lru.size());
}

TEST_GROUP(UnitTest){};

TEST(UnitTest, PutTest) {
    LRU lru;

    lru.put("firstname", "Daan");
    lru.put("lastname", "Kolthof");

    CHECK_FALSE(lru.empty());
    CHECK_EQUAL(2u, lru.size());
}

TEST(UnitTest, PointerTest) {
    LRU lru;

    lru.put("firstname", "Daan");
    lru.put("lastname", "Kolthof");

    check_not_nullptr(lru.get_by_pointer("firstname"));
    check_not_nullptr(lru.get_by_pointer("lastname"));
    check_nullptr(lru.get_by_pointer("address"));

    const std::string *firstname_str_ptr = lru.get_by_pointer("firstname");
    check_not_nullptr(firstname_str_ptr);
    CHECK_EQUAL(std::string("Daan"), *firstname_str_ptr);

    const std::string *lastname_str_ptr = lru.get_by_pointer("lastname");
    check_not_nullptr(lastname_str_ptr);
    CHECK_EQUAL(std::string("Kolthof"), *lastname_str_ptr);

    const std::string *address_str_ptr = lru.get_by_pointer("address");
    check_nullptr(address_str_ptr);
}

TEST(UnitTest, GetByReferenceTest) {
    LRU lru;

    lru.put("firstname", "Daan");
    lru.put("lastname", "Kolthof");

    std::string firstname_str;
    const bool firstname_get_result = lru.get("firstname", firstname_str);
    CHECK_TRUE(firstname_get_result);
    CHECK_EQUAL(std::string("Daan"), firstname_str);

    std::string lastname_str;
    const bool lastname_get_result = lru.get("lastname", lastname_str);
    CHECK_TRUE(lastname_get_result);
    CHECK_EQUAL(std::string("Kolthof"), lastname_str);

    std::string address_str;
    const bool address_get_result = lru.get("address", address_str);
    CHECK_FALSE(address_get_result);
    CHECK_TRUE(address_str.empty());
}

TEST(UnitTest, PutOverwriteTest) {
    LRU lru;

    lru.put("firstname", "Bob");
    lru.put("lastname", "Smith");

    lru.put("firstname", "Oliver");

    CHECK_FALSE(lru.empty());
    CHECK_EQUAL(2u, lru.size());

    std::string firstname_str;
    const bool firstname_get_result = lru.get("firstname", firstname_str);
    CHECK_TRUE(firstname_get_result);
    CHECK_EQUAL(std::string("Oliver"), firstname_str);
}

TEST(UnitTest, OrderTest) {
    LRU lru;

    lru.put("key1", "value1");
    lru.put("key2", "value2");
    lru.put("key3", "value3");

    const LRU::lru_list_type &lru_list = lru.get_recent_used_list();
    LRU::lru_list_type::const_iterator iter = lru_list.begin();

    CHECK_EQUAL(std::string("key3"), iter->first);
    CHECK_EQUAL(std::string("value3"), iter->second);
    ++iter;
    CHECK_EQUAL(std::string("key2"), iter->first);
    CHECK_EQUAL(std::string("value2"), iter->second);
    ++iter;
    CHECK_EQUAL(std::string("key1"), iter->first);
    CHECK_EQUAL(std::string("value1"), iter->second);

    CHECK_TRUE(lru_list.end() == ++iter);

    // Key2 is now most recently used.
    lru.get_by_pointer("key2");

    iter = lru_list.begin();

    CHECK_EQUAL(std::string("key2"), iter->first);
    CHECK_EQUAL(std::string("value2"), iter->second);
    ++iter;
    CHECK_EQUAL(std::string("key3"), iter->first);
    CHECK_EQUAL(std::string("value3"), iter->second);
    ++iter;
    CHECK_EQUAL(std::string("key1"), iter->first);
    CHECK_EQUAL(std::string("value1"), iter->second);

    CHECK_TRUE(lru_list.end() == ++iter);

    // Key1 is now most recently used.
    lru.put("key1", "value1");

    iter = lru_list.begin();

    CHECK_EQUAL(std::string("key1"), iter->first);
    CHECK_EQUAL(std::string("value1"), iter->second);
    ++iter;
    CHECK_EQUAL(std::string("key2"), iter->first);
    CHECK_EQUAL(std::string("value2"), iter->second);
    ++iter;
    CHECK_EQUAL(std::string("key3"), iter->first);
    CHECK_EQUAL(std::string("value3"), iter->second);

    CHECK_TRUE(lru_list.end() == ++iter);
};  // namespace LRUTest

