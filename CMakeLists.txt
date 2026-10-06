cmake_minimum_required(VERSION 3.20)
project(strata VERSION 0.1.0 LANGUAGES CXX)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# pip/scikit-build builds produce packages that may run on other machines (cloud, Docker),
# so they default to portable code; direct CMake builds tune for the host CPU.
if(DEFINED SKBUILD)
  set(_strata_native_default OFF)
else()
  set(_strata_native_default ON)
endif()
option(STRATA_NATIVE "Tune for the host CPU (-march=native on x86-64)" ${_strata_native_default})
option(STRATA_AVX2 "When STRATA_NATIVE is OFF on x86-64, still build the AVX2+FMA kernels (portable to any AVX2 CPU)" ON)
option(STRATA_BUILD_TESTS "Build C++ unit tests" ON)
option(STRATA_BUILD_SERVER "Build the HTTP server" ON)
option(STRATA_BUILD_BENCH "Build C++ micro-benchmarks" ON)
option(STRATA_BUILD_PYTHON "Build the Python extension" OFF)
set(STRATA_SANITIZE "" CACHE STRING "Sanitizer to enable: address, thread or undefined")

find_package(Threads REQUIRED)
include(FetchContent)
# Some third-party CMakeLists still declare cmake_minimum_required < 3.5, which CMake 4 rejects.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)

add_library(strata STATIC
  src/distance.cpp
  src/io.cpp
  src/hnsw.cpp
  src/flat.cpp
  src/metadata.cpp
  src/wal.cpp
  src/collection.cpp)
target_include_directories(strata PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata PUBLIC nlohmann_json::nlohmann_json Threads::Threads)
set_target_properties(strata PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(strata PRIVATE -Wall -Wextra -Wpedantic)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  if(STRATA_NATIVE)
    target_compile_options(strata PUBLIC -march=native)
  elseif(STRATA_AVX2)
    # Docker images / wheels: the build machine may not be the machine that runs the code.
    target_compile_options(strata PUBLIC -mavx2 -mfma)
  endif()
endif()

if(STRATA_SANITIZE)
  set(_san -fsanitize=${STRATA_SANITIZE} -fno-omit-frame-pointer -g)
  target_compile_options(strata PUBLIC ${_san})
  target_link_options(strata PUBLIC -fsanitize=${STRATA_SANITIZE})
endif()

if(STRATA_BUILD_TESTS)
  FetchContent_Declare(doctest
    URL https://raw.githubusercontent.com/doctest/doctest/v2.4.11/doctest/doctest.h
    DOWNLOAD_NO_EXTRACT TRUE)
  FetchContent_MakeAvailable(doctest)
  enable_testing()
  add_executable(strata_tests
    tests/cpp/test_main.cpp
    tests/cpp/test_distance.cpp
    tests/cpp/test_hnsw.cpp
    tests/cpp/test_persistence.cpp
    tests/cpp/test_quantization.cpp
    tests/cpp/test_collection.cpp)
  target_include_directories(strata_tests PRIVATE ${doctest_SOURCE_DIR} tests/cpp)
  target_link_libraries(strata_tests PRIVATE strata)
  add_test(NAME strata_tests COMMAND strata_tests)
endif()

if(STRATA_BUILD_SERVER)
  FetchContent_Declare(httplib
    URL https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.18.3/httplib.h
    DOWNLOAD_NO_EXTRACT TRUE)
  FetchContent_MakeAvailable(httplib)
  add_executable(strata-server server/main.cpp)
  target_include_directories(strata-server PRIVATE ${httplib_SOURCE_DIR})
  target_link_libraries(strata-server PRIVATE strata)
endif()

if(STRATA_BUILD_BENCH)
  add_executable(micro_distance bench/micro_distance.cpp)
  target_link_libraries(micro_distance PRIVATE strata)
endif()

if(STRATA_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_strata python/bindings.cpp)
  target_link_libraries(_strata PRIVATE strata)
  target_compile_definitions(_strata PRIVATE STRATA_VERSION="${PROJECT_VERSION}")
  install(TARGETS _strata DESTINATION strata)
endif()
