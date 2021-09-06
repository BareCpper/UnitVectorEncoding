
cmake_minimum_required (VERSION 3.15) #FetchContent

project(UnitVectorEncoding)


include(FetchContent)
FetchContent_Declare(
  cmakefetchcontent_googletest
  GIT_REPOSITORY https://github.com/CMakeFetchContent/googletest.git
)
FetchContent_MakeAvailable(cmakefetchcontent_googletest)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  CMAKE_ARGS
      -DCMAKE_BUILD_TYPE:STRING=Release
      -DBENCHMARK_ENABLE_TESTING=OFF
      -DBENCHMARK_ENABLE_GTEST_TESTS=OFF
#  BUILD_COMMAND ${CMAKE_COMMAND} --build . --config Release
  # TODO Living at trunk!
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "") 
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE INTERNAL "") 
#set(CMAKE_BUILD_TYPE Release CACHE INTERNAL "<some description>")
FetchContent_MakeAvailable(googlebenchmark)

# Add source to this project'sexecutable.
add_executable (UnitVectorEncoding 
    "test.cpp" )
target_compile_options(UnitVectorEncoding
    PUBLIC 
        -ffast-math -mfma
)
target_compile_features( UnitVectorEncoding 
    PUBLIC
        cxx_std_17 
)
target_link_libraries(UnitVectorEncoding
    PUBLIC
        GTest::Main
        benchmark::benchmark
)