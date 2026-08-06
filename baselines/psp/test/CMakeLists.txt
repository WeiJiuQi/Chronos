set(CMAKE_CXX_STANDARD 17)

add_executable(test_mips_index test_mips_index.cpp)
target_link_libraries(test_mips_index ${PROJECT_NAME})

add_executable(test_mips_search test_mips_search.cpp)
target_link_libraries(test_mips_search ${PROJECT_NAME})





