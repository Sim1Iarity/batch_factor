#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <queue>
#include <vector>
#include <chrono>
#include <fstream>
#include <map>
#include <sstream>
#include <unistd.h>
#include <bitset>
#include <inttypes.h>
#if __has_include(<primesieve.h>)
#include <primesieve.h>
#endif
typedef int64_t ll;
typedef uint64_t ull;
static inline __attribute__((always_inline)) void mark(ull* s, int o) { s[o >> 6] |= 1ull << (o & 63); }
static inline __attribute__((always_inline)) void unmark(ull* s, int o) { s[o >> 6] &= ~(1ull << (o & 63)); }
static inline __attribute__((always_inline)) bool test(ull* s, int o) { return (s[o >> 6] & (1ull << (o & 63))); }
#if __has_include(<primecount.h>)
#include <primecount.h>
#else
ull piSieve(const ull N, const int nthreads = 8) {
    const double n = N;
    if (n <= 1) return 0LL;
    if (n == 2) return 1LL;
    const int lim = std::sqrt(n);
    int vsz = (lim + 1) >> 1;
    std::vector<int> smalls(vsz);
#pragma GCC ivdep
    for (int cx = 0; cx < vsz; ++cx) smalls[cx] = cx;
    std::vector<libdivide::branchfree_divider<ull>> roughs_div(vsz);
    std::vector<int> roughs_val(vsz);
#pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int cx = 1; cx < vsz; ++cx) roughs_div[cx] = roughs_val[cx] = cx << 1 | 1;
    std::vector<ull> larges(vsz);
    larges[0] = (N - 1) >> 1;
#pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int cx = 1; cx < vsz; ++cx) larges[cx] = (N / roughs_div[cx] - 1) >> 1;
    std::vector<uint64_t> skips_mask((vsz + 63) / 64 + 1, 0);
    int pCnt = 0;
    for (int p = 3; p <= lim; p += 2) {
        if (test(skips_mask.data(), p >> 1)) continue;
        int p2 = p * p;
        if ((ull)p2 * p2 > N) break;
        mark(skips_mask.data(), p >> 1);
        for (int cx = p2 >> 1; cx <= lim >> 1; cx += p) mark(skips_mask.data(), cx);
        const ull local_n = N / p;
        larges[0] = larges[0] - larges[smalls[p >> 1] - pCnt] + pCnt;
        int ns = 1, target_limit = lim / p;
        int idx = std::upper_bound(roughs_val.begin() + 1, roughs_val.begin() + vsz, target_limit) - roughs_val.begin();
        for (int cx = 1; cx < idx; ++cx) {
            auto cur = roughs_val[cx];
            if (__builtin_expect(test(skips_mask.data(), cur >> 1), true)) continue;
            ull d = (ull)cur * p;
            larges[ns] = larges[cx] - larges[smalls[d >> 1] - pCnt] + pCnt;
            memcpy(roughs_div.data() + ns, roughs_div.data() + cx, sizeof(decltype(roughs_div[0])));
            roughs_val[ns++] = cur;
        }
        for (int cx = idx; cx < vsz; ++cx) {
            auto cur = roughs_val[cx];
            if (__builtin_expect(test(skips_mask.data(), cur >> 1), true)) continue;
            larges[ns] = larges[cx] - smalls[(local_n / roughs_div[cx] - 1) >> 1] + pCnt;
            memcpy(roughs_div.data() + ns, roughs_div.data() + cx, sizeof(decltype(roughs_div[0])));
            roughs_val[ns++] = cur;
        }
        vsz = ns;
        for (int cx = (lim - 1) >> 1, cy = ((lim / p) - 1) | 1; cy >= p; cy -= 2) {
            int cur = smalls[cy >> 1] - pCnt;
#pragma GCC ivdep
            for (int cz = (cy * p) >> 1; cz <= cx; --cx) smalls[cx] -= cur;
        }
        ++pCnt;
    }
    larges[0] += (ull)(vsz + ((pCnt - 1) << 1)) * (vsz - 1) >> 1;
    for (int cx = 1; cx < vsz; ++cx) larges[0] -= larges[cx];
    int low = 1, high = vsz, split_cx = vsz;
    while (low < high) {
        int mid = low + (high - low) / 2;
        auto q = roughs_div[mid];
        ull m = N / q;
        int e = smalls[(m / q - 1) >> 1] - pCnt;
        if (e >= mid + 1) low = mid + 1;
        else {
            split_cx = mid;
            high = mid;
        }
    }
    ull larges_0_delta = 0;
#pragma omp parallel for reduction(+: larges_0_delta) schedule(dynamic) num_threads(nthreads)
    for (int cx = 1; cx < split_cx; ++cx) {
        auto q = roughs_div[cx];
        ull m = N / q, t = 0;
        int e = smalls[(m / q - 1) >> 1] - pCnt;
#pragma omp simd
        for (int cy = cx + 1; cy <= e; ++cy) t += smalls[(m / roughs_div[cy] - 1) >> 1];
        larges_0_delta += t - (ull)(e - cx) * (pCnt + cx - 1);
    }
    return larges[0] + larges_0_delta + 1;
}
#endif
#if !__has_include(<gmp.h>)
typedef struct
{
  long _mp_alloc;
  long _mp_size;
  unsigned long *_mp_d;
} __mpz_struct;
typedef __mpz_struct mpz_t[1];
typedef const __mpz_struct *mpz_srcptr;
typedef __mpz_struct *mpz_ptr;
extern "C" {
__declspec(__dllimport__) void __gmpz_init (mpz_ptr) __nothrow;
__declspec(__dllimport__) void __gmpz_set_ui (mpz_ptr, unsigned long int);
__declspec(__dllimport__) void __gmpz_mul (mpz_ptr, mpz_srcptr, mpz_srcptr);
__declspec(__dllimport__) void __gmpz_mul_ui (mpz_ptr, mpz_srcptr, unsigned long int);
__declspec(__dllimport__) void __gmpz_clear (mpz_ptr);
__declspec(__dllimport__) size_t __gmpz_sizeinbase (mpz_srcptr, int) __nothrow __attribute__ ((__pure__));
__declspec(__dllimport__) void __gmpz_realloc2 (mpz_ptr, unsigned long);
}
#else
#include <gmp.h>
#endif
#ifdef __linux__
std::map<std::string, std::string> proc_fields(pid_t pid, std::string const& endpoint) {
    std::map<std::string, std::string> D;
    const std::string str = std::string("/proc/") + std::to_string(pid) + "/" + endpoint;
    std::ifstream f(str);
    for(std::string key; f >> key; ) {
        std::string value;
        key.erase(key.end() - 1);
        std::getline(f, value);
        D[key] = value;
    }
    return D;
}
size_t Memusage(void) {
    size_t mem;
    if (std::istringstream(proc_fields(getpid(), "status")["VmSize"]) >> mem) return mem;
    else return -1;
}
size_t PeakMemusage(void) {
    size_t mem;
    if (std::istringstream(proc_fields(getpid(), "status")["VmPeak"]) >> mem) return mem;
    else return -1;
}
#elif defined(_WIN32)
#include <windows.h>
#include <psapi.h>
size_t Memusage(void) {
    HANDLE hProcess = GetCurrentProcess();
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(hProcess, &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / 1024;
    }
    else return -1;
}
size_t PeakMemusage(void) {
    HANDLE hProcess = GetCurrentProcess();
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(hProcess, &pmc, sizeof(pmc))) {
        return pmc.PeakWorkingSetSize / 1024;
    }
    else return -1;
}
#endif
#define TIMING(str, cpu, wct, ...) fprintf(stderr, str " took %.3lfms cpu (%.6lfms wct); memory %.3lfMB, peak %.3lfMB\n",\
    __VA_ARGS__, (clock() - cpu) * 1000.0 / CLOCKS_PER_SEC,\
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - wct).count() / 1e6,\
    Memusage() / 1024.0, PeakMemusage() / 1024.0);\
cpu = clock();\
wct = std::chrono::steady_clock::now()
int64_t *all;
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max_primes ((160000))
#define sieve_span ((1 << 22))
#define sieve_words ((sieve_span >> 7))
#define wheel_size ((3 * 5 * 7 * 11 * 13))
int primes[max_primes], mcnt;
ull sieve[sieve_words];
ull pattern[wheel_size];
ull pcnt;
void update_sieve(ll base) {
    ll o = base % wheel_size;
    o = (o + ((o * 85) & 127) * wheel_size) >> 7;
    for (int i = 0, k; i < sieve_words; i += k, o = 0) {
        k = min(wheel_size - o, sieve_words - i);
        memcpy(sieve + i, pattern + o, sizeof(*pattern) * k);
    }
    if (base == 0) {
        sieve[0] |= 1;
        sieve[0] &= ~46ull;
    }
    for (int i = 0; i < mcnt; ++i) {
        ll j = primes[i] * primes[i];
        if (j > base + sieve_span - 1) break;
        if (j > base) j = (j - base) >> 1;
        else {
            j = primes[i] - base % primes[i];
            if (!(j & 1)) j += primes[i];
            j >>= 1;
        }
        while (j < sieve_span >> 1) {
            mark(sieve, j);
            j += primes[i];
        }
    }
}
void pre_sieve() {
    for (int i = 0; i < (1048576 >> 7); ++i) sieve[i] = 0;
    for (int i = 3; i < 1024; i += 2) {
        if (!test(sieve, i >> 1)) {
            for (int j = (i * i) >> 1; j < 1048576; j += i) mark(sieve, j);
        }
    }
    mcnt = 0;
    for (int i = 8; i < 1048576; ++i) {
        if (!test(sieve, i)) {
            primes[mcnt] = (i << 1) + 1;
            ++mcnt;
        }
    }
    for (int i = 0; i < wheel_size; ++i) pattern[i] = 0;
    for (int i = 1; i < wheel_size * 64; i += 3) mark(pattern, i);
    for (int i = 2; i < wheel_size * 64; i += 5) mark(pattern, i);
    for (int i = 3; i < wheel_size * 64; i += 7) mark(pattern, i);
    for (int i = 5; i < wheel_size * 64; i += 11) mark(pattern, i);
}
void segment_sieve(ll base, ll lim) {
    update_sieve(base);
    for (ll i = 0, p, u; i < sieve_words; ++i) {
        ull o = ~sieve[i];
        while (o) {
            p = __builtin_ctzll(o);
            u = base + (i << 7) + (p << 1) | 1;
            if (u >= lim) break;
            all[pcnt++] = u;
            o -= o & ((~o) + 1);
        }
    }
}
void fast_sieve(ll lim, ll start = 0) {
    pre_sieve();
    all[pcnt++] = 2;
    for (ll base = start; base < lim; base += sieve_span) {
        fprintf(stderr, "sieving primes <= %" PRIi64 "\r", base);
        segment_sieve(base, lim);
    }
}
void multiply(mpz_ptr ret, size_t left, size_t right) {
    if (right - left < 50) {
        __gmpz_set_ui(ret, 1);
        for (size_t i = left; i < right; i++) __gmpz_mul_ui(ret, ret, all[i]);
        return;
    }
    size_t mid = left + 1.05 * ((right - left) >> 1);
    mpz_t a;
    __gmpz_init(a);
    multiply(ret, left, mid);
    multiply(a, mid, right);
    __gmpz_mul(ret, ret, a);
    __gmpz_clear(a);
}
mpz_t rets[128];
size_t lefts[128], rights[128];
int main(int argc, char* argv[]) {
    int64_t n = 0, start = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--start") == 0) {
            i++;
            sscanf(argv[i], "%" PRIi64, &start);
            continue;
        }
        sscanf(argv[i], "%" PRIi64, &n);
    }
    if (n <= 0) {
        printf("Error: n must be positive\n");
        exit(1);
    }
    if (n > (1ll << 42)) {
        printf("Warning: %" PRIi64 " truncated to %lld due to implementation\n", n, 1ll << 42);
        n = 1ll << 42;
    }
    clock_t t_cpu = clock();
    auto t_wct = std::chrono::steady_clock::now();
#if !__has_include(<primesieve.h>)
#if __has_include(<primecount.h>)
    primecount_set_num_threads(8);
    int64_t pi = primecount_pi(n);
#else
    int64_t pi = piSieve(n);
#endif
    all = NULL;
    pcnt = 0;
    TIMING("pi(%" PRIi64 ")", t_cpu, t_wct, n);
    fprintf(stderr, "Needs to allocate %" PRIi64 " values\n", pi);
    all = (int64_t*)malloc(pi * sizeof(int64_t));
    if (all == NULL) {
        fprintf(stderr, "Error: not enough memory\n");
        exit(-1);
    }
    TIMING("Allocating %" PRIi64 " elements", t_cpu, t_wct, n);
    fast_sieve(n, start);
    TIMING("sieving primes <= %" PRIi64 " (Total %" PRIi64 ")", t_cpu, t_wct, n, pcnt);
#else
    pcnt = 0;
    all = (int64_t*)primesieve_generate_primes(start, n, &pcnt, INT64_PRIMES);
    if (!all) {
        fprintf(stderr, "Warning: primesieve allocation failed, using internal sieve\n");
        all = NULL;
        pcnt = 0;
#if __has_include(<primecount.h>)
        primecount_set_num_threads(8);
        int64_t pi = primecount_pi(n);
        int64_t pi_start = start ? primecount_pi(start - 1) : 0;
#else
        int64_t pi = piSieve(n);
        int64_t pi_start = start ? piSieve(start - 1) : 0;
#endif
        TIMING("pi(%" PRIi64 ") and pi(%" PRIi64 ")", t_cpu, t_wct, n, start);
        pi -= pi_start;
        fprintf(stderr, "Needs to allocate %" PRIi64 " values\n", pi);
        all = (int64_t*)malloc(pi * sizeof(int64_t));
        if (all == NULL) {
            fprintf(stderr, "Error: not enough memory\n");
            exit(-1);
        }
        TIMING("Allocating %" PRIi64 " elements", t_cpu, t_wct, n);
        fast_sieve(n, start);
    }
    TIMING("sieving primes <= %" PRIi64 " (Total %" PRIu64 ")", t_cpu, t_wct, n, pcnt);
#endif
    for (size_t i = 0; i < 128; i++) {
        lefts[i] = pcnt * i / 128;
        rights[i] = pcnt * (i + 1) / 128;
        __gmpz_init(rets[i]);
    }
    unsigned done = 0;
#pragma omp parallel for num_threads(8)
    for (unsigned i = 0; i < 128; i++) {
        multiply(rets[i], lefts[i], rights[i]);
        fprintf(stderr, "%u\r", ++done);
    }
    TIMING("%s", t_cpu, t_wct, "128 fragments");
    free(all);
    TIMING("%s", t_cpu, t_wct, "freeing the prime array");
    for (unsigned level = 1, j = 64; j >= 2; j >>= 1, level++) {
        done = 0;
#pragma omp parallel for num_threads(8)
        for (unsigned i = 0; i < j; i++) {
            __gmpz_mul(rets[i], rets[i + j], rets[i]);
            __gmpz_clear(rets[i + j]);
            fprintf(stderr, "%u\r", ++done);
        }
        TIMING("level %u (%u fragments)", t_cpu, t_wct, level, j);
    }
    __gmpz_mul(rets[0], rets[0], rets[1]);
    TIMING("%s", t_cpu, t_wct, "final multiplication");
    if (__gmpz_sizeinbase(rets[0], 2) >= (1ll << 36)) {
        fprintf(stderr, "warning: primorial has %llu bits\n", (unsigned long long)__gmpz_sizeinbase(rets[0], 2));
    }
    __gmpz_clear(rets[1]);
    FILE *f = fopen("prim.dat", "wb");
    __gmpz_out_raw(f, rets[0]);
    fclose(f);
    __gmpz_clear(rets[0]);
}