cmake_minimum_required(VERSION 3.20)
project(hf_microstructure CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Compiler flags ───────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(
  -Wall -Wextra -Wpedantic
  $<$<CONFIG:Release>:-O3>
  $<$<CONFIG:Release>:-march=native>
  $<$<CONFIG:Release>:-DNDEBUG>
  $<$<CONFIG:Debug>:-O0>
  $<$<CONFIG:Debug>:-g>
)
# Note: ASan/UBSan omitted on macOS due to AppleClang compatibility

# ─── Dependencies (Catch2 v3, nanobench) ─────────────────────────────────────
include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.3
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

FetchContent_Declare(
  nanobench
  GIT_REPOSITORY https://github.com/martinus/nanobench.git
  GIT_TAG        v4.3.11
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nanobench)

# ─── Library: hf_lib ─────────────────────────────────────────────────────────
file(GLOB LIB_SOURCES cpp/src/lob_parser.cpp
                      cpp/src/feature_engine.cpp
                      cpp/src/hawkes_mle.cpp
                      cpp/src/hjb_solver.cpp)

add_library(hf_lib STATIC ${LIB_SOURCES})
target_include_directories(hf_lib PUBLIC cpp/include)

# ─── Demo executable: hf_demo ────────────────────────────────────────────────
add_executable(hf_demo cpp/src/main.cpp)
target_link_libraries(hf_demo PRIVATE hf_lib)

# ─── Test executable: hf_tests ───────────────────────────────────────────────
file(GLOB TEST_SOURCES cpp/tests/test_feature_engine.cpp
                       cpp/tests/test_hawkes_mle.cpp
                       cpp/tests/test_hjb_solver.cpp)

add_executable(hf_tests ${TEST_SOURCES})
target_link_libraries(hf_tests PRIVATE hf_lib Catch2::Catch2WithMain nanobench::nanobench)

include(CTest)
include(Catch)
catch_discover_tests(hf_tests)
