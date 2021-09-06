
#if __has_include("benchmark/benchmark.h")
#define BENCHMARK_ENABLED 1
#include <benchmark/benchmark.h>
#endif

#if __has_include("gtest/gtest.h")
#define TEST_ENABLED 1
#include <gtest/gtest.h>
#endif

#include <cstddef> //< std::size
#include <cmath>
#include <random>

//PAPER http://jcgt.org/published/0003/02/01/paper.pdf

#if __has_attribute(vector_size)
//https://gcc.gnu.org/onlinedocs/gcc/Vector-Extensions.html
using v2sf = float __attribute__ ((vector_size (8)));
using v4sf = float __attribute__ ((vector_size (16)));
#endif

union float2
{
    v2sf v;
    struct{
        float x, y;
    };

    inline constexpr operator bool () const
    {  return (x != 0) && (y != 0); }
};

inline bool operator < ( const float2 a, const float b)
{  return  (a.x < 0.0F) && (a.y < 0.0F); }

inline float2 operator >= ( const float2 a, const float2 b)
{  return {a.v >= b.v}; }

inline float2& operator += ( float2& a, const float2 b)
{ a.x += b.x, a.y += b.y; return a; }

inline float2& operator += ( float2& a, const float b)
{ return a += float2{b,b}; }

union float3
{
    struct{
     float x, y, z;
    };
    float2 xy;
};

union float4
{
    v4sf v;
    struct{
        float x, y, z, w;
    };
    float3 xyz;
    float2 xy;
};

// -O3 -ffast-math -mfma4
inline float4 normalize(float4 n)
{
    const float4 sqr = {n.v * n.v};
    const float sum = sqr.x + sqr.y + sqr.z;// + sqr.w;
    const float invMag = 1.0F / std::sqrt(sum);  
    return float4{ n.x * invMag, n.y * invMag, n.z * invMag, n.w * invMag } ;
}
inline v4sf normalize(v4sf n)
{
    v4sf sqr = {n * n};
    float sum = sqr[0] + sqr[1] + sqr[2] + sqr[3]; //<@todo Use W or not?
    float invMag = 1.0F / std::sqrt(sum);  
    return n * invMag;
}

float2 signNotZero(float2 v) {
    return float2{(v.x >= 0.0F) ? +1.0F : -1.0F
                , (v.y >= 0.0F) ? +1.0F : -1.0F};
}

float3 octDecodeFloat2( float2 f )
{
    float z = 1.0F - std::abs(f.x) - std::abs(f.y);
    const float t = std::max(-z, 0.0F);
    float2 xy = {f.x, f.y};
    //xy = z>0 ? xy : (1-abs)
#if 0
    xy.v += ((xy.v < 0.0F) ? t : -t);   /// @note Uinsg vector extension!
#else
    xy.x += ((xy.x < 0.0F) ? t : -t);
    xy.y += ((xy.y < 0.0F) ? t : -t);
#endif
   // xy += signNotZero(xy) * -t;   
    return normalize( float4{ xy.x, xy.y, z, 0} ).xyz;
}

float3 octDecodeVec( float2 f )
{
    float z = 1.0F - std::abs(f.x) - std::abs(f.y);
    const float t = std::max(-z, 0.0F);
    v2sf quad = (f.v < 0.0F) ? t : -t;
    v4sf norm = normalize( v4sf{ f.v[0] + quad[0], f.v[1] + quad[1], z, 0} );
    return float3{ norm[0], norm[1], norm[2] };
}

#if 0// JCGT http://jcgt.org/published/0003/02/01/paper.pdf
    // Returns ±1
    vec2 signNotZero(vec2 v) {
        return vec2((v.x >= 0.0) ? +1.0 : -1.0, (v.y >= 0.0) ? +1.0 : -1.0);
    }
    // Assume normalized input. Output is on [-1, 1] for each component.
    vec2 float32x3_to_oct(in vec3 v) {
        // Project the sphere onto the octahedron, and then onto the xy plane
        vec2 p = v.xy * (1.0 / (abs(v.x) + abs(v.y) + abs(v.z)));
        // Reflect the folds of the lower hemisphere over the diagonals
        return (v.z <= 0.0) ? ((1.0 - abs(p.yx)) * signNotZero(p)) : p;
    }
    vec3 oct_to_float32x3(vec2 e) {
        vec3 v = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
        if (v.z < 0) v.xy = (1.0 - abs(v.yx)) * signNotZero(v.xy);
        return normalize(v);
    }
#endif

float3 octDecodeJCGT( float2 e )
{
    float3 v = {e.x, e.y, 1.0F - abs(e.x) - abs(e.y)};

#if 0

    const float t = std::max(-v.z, 0.0F);
    //xy = z>0 ? xy : (1-abs)
    v.x += ((v.x < 0.0F) ? t : -t);
    v.y += ((v.y < 0.0F) ? t : -t);
#else
    /// TODO: Should the yx be swapped here or not!?
    //if (v.z < 0) v.xy = (1.0 - abs(v.yx)) * signNotZero(v.xy);
    if (v.z < 0)
    {
        auto sig = signNotZero(v.xy);
         v.xy = float2{ 
               (1.0F - abs(v.y)) * ((v.x >= 0.0F) ? +1.0F : -1.0F)
             , (1.0F - abs(v.x)) * ((v.y >= 0.0F) ? +1.0F : -1.0F) };
    }
#endif
    return normalize( float4{ v.x, v.y, v.z, 0} ).xyz;
}

#if BENCHMARK_ENABLED
const int seed = 0;
std::uniform_real_distribution<float> dist(-0.5F, 0.5F);

static void BenchOctDecodeFloat2(benchmark::State& state) {
    std::mt19937 gen(seed);
    
  // Code inside this loop is measured repeatedly
  for (auto _ : state) {
    state.PauseTiming();
    float2 f = { dist(gen), dist(gen) };  
    state.ResumeTiming();
    // Make sure the variable is not optimized away by compiler
    benchmark::DoNotOptimize(octDecodeFloat2(f));
  }
}

static void BenchOctDecodeVec(benchmark::State& state) {
    std::mt19937 gen(seed);
    
  // Code inside this loop is measured repeatedly
  for (auto _ : state) {
    state.PauseTiming();
    float2 f = { dist(gen), dist(gen) };  
    state.ResumeTiming();
    // Make sure the variable is not optimized away by compiler
    benchmark::DoNotOptimize(octDecodeVec(f));
  }
}

static void BenchOctDecodeJCGT(benchmark::State& state) {
    std::mt19937 gen(seed);

  // Code inside this loop is measured repeatedly
  for (auto _ : state) {      
    state.PauseTiming();
    float2 f = {  dist(gen), dist(gen) };  
    state.ResumeTiming();
    // Make sure the variable is not optimized away by compiler
    benchmark::DoNotOptimize(octDecodeJCGT(f));
  }
}
// Register the function as a benchmark
BENCHMARK(BenchOctDecodeJCGT)->MinTime(3);
BENCHMARK(BenchOctDecodeFloat2)->MinTime(3);
BENCHMARK(BenchOctDecodeVec)->MinTime(3);

TEST(MyTest, Benchmarks)
{
    ::benchmark::RunSpecifiedBenchmarks();
}
#endif //< BENCHMARK_ENABLED

struct { float2 enc; float3 dec; } encDecTests[] =
{
    //NOTE: @corners = -z_axis
    //NOTE: @center = +z_axis
      {{-1.0,-1.0}, {0, 0, -1.0}}
    , {{-0.5,-1.0}, {0, -0.707106781, -0.707106781}}
    , {{0.0,-1.0}, {0, -1, 0}}
    , {{0.5,-1.0}, {0, -0.707106781, -0.707106781}}
    , {{1.0,-1.0}, {0, 0, -1}}

    ,  {{-1.0,-0.5}, {-0.707106781, 0, -0.707106781}}
    , {{-0.5,-0.5}, {-0.707106781, -0.707106781, 0}}
    , {{0.0,-0.5}, {0, -0.707106781, 0.707106781}}
    , {{0.5,-0.5}, {0.707106781, -0.707106781, 0}}
    , {{1.0,-0.5}, {0.707106781, 0, -0.707106781}}

    , {{-1.0,0.0}, {-1, 0, 0}}
    , {{-0.5,0.0}, {-0.707106781, 0, 0.707106781}}
    , {{0.0,0.0}, {0, 0, 1}}
    , {{0.5,0.0}, {0.707106781, 0, 0.707106781}}
    , {{1.0,0.0}, {1, 0, 0}}

   // , {{0.25,0.25}, {1, 0, 0}}

    , {{-1.0,0.5}, {-0.707106781, 0, -0.707106781}}
    , {{0.5,0.5}, {0.707106781, 0.707106781, 0}}
    , {{0.0,0.5}, {0, 0.707106781, 0.707106781}}
    , {{0.5,0.5}, {0.707106781, 0.707106781, 0}}
    , {{1.0,0.5}, {0.707106781, 0, -0.707106781}}

    , {{-1.0,1.0}, {0, 0, -1}}
    , {{-0.5,1.0}, {0, 0.707106781, -0.707106781}}
    , {{0.0,1.0}, {0, 1, 0}}
    , {{0.5,1.0}, {0, 0.707106781, -0.707106781}}
    , {{1.0,1.0}, {0, 0, -1}}
};

#if TEST_ENABLED
class DecodeJCGT : public testing::TestWithParam<size_t> {};
TEST_P(DecodeJCGT, Correct)
{
    auto encDec = encDecTests[GetParam()];
    auto res = octDecodeJCGT( encDec.enc );
    EXPECT_NEAR( res.x, encDec.dec.x, 0.001 );
    EXPECT_NEAR( res.y, encDec.dec.y, 0.001 );
    EXPECT_NEAR( res.z, encDec.dec.z, 0.001);
}


class DecodeFloat2 : public testing::TestWithParam<size_t> {};
TEST_P(DecodeFloat2, Correct)
{
    auto encDec = encDecTests[GetParam()];
    auto res = octDecodeFloat2( encDec.enc );
    EXPECT_NEAR( res.x, encDec.dec.x, 0.001 );
    EXPECT_NEAR( res.y, encDec.dec.y, 0.001 );
    EXPECT_NEAR( res.z, encDec.dec.z, 0.001);
}

class DecodeVec : public testing::TestWithParam<size_t> {};
TEST_P(DecodeVec, Correct)
{
    auto encDec = encDecTests[GetParam()];
    auto res = octDecodeVec( encDec.enc );
    EXPECT_NEAR( res.x, encDec.dec.x, 0.001 );
    EXPECT_NEAR( res.y, encDec.dec.y, 0.001 );
    EXPECT_NEAR( res.z, encDec.dec.z, 0.001);
}

INSTANTIATE_TEST_SUITE_P(Decodes, DecodeJCGT, testing::Range(size_t(0), std::size(encDecTests) ) );
INSTANTIATE_TEST_SUITE_P(Decodes, DecodeFloat2, testing::Range(size_t(0), std::size(encDecTests) ) );
INSTANTIATE_TEST_SUITE_P(Decodes, DecodeVec, testing::Range(size_t(0), std::size(encDecTests) ) );

#endif //TEST_ENABLED

int main(int argc, char **argv) {

    int retval = -1;
  #if TEST_ENABLED
    ::testing::InitGoogleTest(&argc, argv);
    retval = RUN_ALL_TESTS();
  #elif BENCHMARK_ENABLED
    retval = ::benchmark::RunSpecifiedBenchmarks();
  #endif
    return retval;
}
//int main(int,char**);
//BENCHMARK_MAIN();