
cmake_minimum_required (VERSION 3.15) #FetchContent

project(UnitVectorEncoding)


include(FetchContent)
FetchContent_Declare(
  cmakefetchcontent_googletest
  GIT_REPOSITORY https://github.com/CMakeFetchContent/googletest.git
)
FetchContent_MakeAvailable(cmakefetchcontent_googletest)

FetchContent_Declare(
  fetchcontent_googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  # TODO Living at trunk!
)
FetchContent_MakeAvailable(fetchcontent_googlebenchmark)

# Add source to this project'sexecutable.
add_executable (UnitVectorEncoding 
    "test.cpp" )

target_link_libraries(UnitVectorEncoding
    PUBLIC
        GTest::Main
        benchmark::benchmark
)