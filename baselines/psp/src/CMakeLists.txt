set(CMAKE_CXX_STANDARD 17)

list(
    APPEND CPP_SOURCES
    index.cpp
    index_random.cpp
    index_mips.cpp
    util.cpp
)

add_library(${PROJECT_NAME} ${CPP_SOURCES})
add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})