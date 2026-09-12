cmake_minimum_required(VERSION 3.20)
project(microduck_control LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization and warning flags
if(MSVC)
    add_compile_options(/O2 /W4 /permissive-)
else()
    add_compile_options(
        -O3
        -march=native
        -Wall
        -Wextra
        -Wpedantic
        -fno-omit-frame-pointer
    )
endif()

find_package(Threads REQUIRED)

# Link librt on Linux if present
if(UNIX AND NOT APPLE)
    find_library(LIBRT rt)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ------------------------------------------------------------------------------
# 1. Real-Time Motor Daemon (robotd)
# ------------------------------------------------------------------------------
add_executable(robotd
    src/robotd_main.cpp
)

target_link_libraries(robotd PRIVATE
    Threads::Threads
    $<$<BOOL:${LIBRT}>:${LIBRT}>
)

# ------------------------------------------------------------------------------
# 2. Behavior Tree IPC Bridge Node
# ------------------------------------------------------------------------------
add_executable(bt_bridge_node
    src/bt_bridge_node.cpp
)

target_link_libraries(bt_bridge_node PRIVATE
    Threads::Threads
    $<$<BOOL:${LIBRT}>:${LIBRT}>
)

# ------------------------------------------------------------------------------
# Install Rules
# ------------------------------------------------------------------------------
install(TARGETS robotd bt_bridge_node
    RUNTIME DESTINATION bin
)
