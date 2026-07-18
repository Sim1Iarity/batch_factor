#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <quadmath.h>
#include <numeric>
#include <map>
#include <sstream>
#include <fstream>
#include <string.h>
#include <vector>
#include <utility>
#include <inttypes.h>
#include <libdivide.h>
#include <algorithm>
#pragma GCC target("avx2")
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) < (b)) ? (b) : (a))
#define max_primes ((160000))
#define sieve_span ((1 << 22))
#define sieve_words ((sieve_span >> 7))
#define wheel_size ((3 * 5 * 7 * 11))
template <typename T>
T initialNthPrimeApprox(T x) {
    if (x < 1) return 0;
    if (x >= 1 && x < 2) return 2;
    if (x >= 2 && x < 3) return 3;
    T logx = logl(x);
    T loglogx = logl(logx);
    T t = logx + (loglogx / 2);
    if (x > 1600) t += (loglogx / 2) - 1 + (loglogx - 2) / logx;
    if (x > 1200000) t -= (loglogx * loglogx - 6 * loglogx + 11) / (2 * logx * logx);
    return x * t;
}
template <typename T>
T Li(T x) {
    if (x <= 1) return 0;
    T sum = 0, inner_sum = 0, factorial = 1, p = -1, q = 0, power2 = 1, logx = logl(x);
    int k = 0;
    for (int n = 1; n < 200; n++) {
        p *= -logx;
        factorial *= n;
        q = factorial * power2;
        power2 *= 2;
        for (; k <= (n - 1) / 2; k++) inner_sum += T(1.0) / (2 * k + 1);
        auto old_sum = sum;
        sum += (p / q) * inner_sum;
        if (fabsl(sum - old_sum) <= std::numeric_limits<T>::epsilon()) break;
    }
    return logl(logx) + sqrtl(x) * sum - 0.46794811521595992423807679911221070548L;
}
template <typename T>
T Li_inverse(T x) {
    if (x < 1) return 0;
    T t = initialNthPrimeApprox(x);
    T old_term = std::numeric_limits<T>::infinity();
    for (int i = 0; i < 10; i++) {
        T delta = Li(t) - x;
        T term = delta * logl(t) / (1 + delta / (2 * t));
        if (std::abs(term) >= std::abs(old_term)) break;
        t -= term;
        old_term = term;
    }
    return t;
}
extern "C" {
typedef int64_t ll;
typedef uint64_t ull;
static inline __attribute__((always_inline)) void mark(ull* s, int o) { s[o >> 6] |= 1ull << (o & 63); }
static inline __attribute__((always_inline)) void unmark(ull* s, int o) { s[o >> 6] &= ~(1ull << (o & 63)); }
static inline __attribute__((always_inline)) bool test(ull* s, int o) { return (s[o >> 6] & (1ull << (o & 63))); }
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
int primes[max_primes], mcnt;
ull sieve[sieve_words];
ull pattern[wheel_size];
ll pcnt;
int64_t *all;
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
std::vector<int64_t> small_prime_lookup;
const uint64_t THRESHOLD_2_24 = 1 << 24;
void initialize_small_lookup() {
    all = (int64_t*)malloc(2000000 * sizeof(int64_t)); 
    pcnt = 0;
    pre_sieve();
    all[pcnt++] = 2;
    for (ull base = 0; base < THRESHOLD_2_24; base += sieve_span) {
        segment_sieve(base, THRESHOLD_2_24);
    }
    small_prime_lookup.resize(pcnt);
    memcpy(small_prime_lookup.data(), all, pcnt * sizeof(int64_t));
    free(all);
    all = NULL;
    pcnt = 0;
}
uint64_t micro_window_sieve(uint64_t start, uint64_t end, uint64_t md, int64_t prime_delta) {
    if (start % 2 == 0) start++;
    if (end % 2 == 0) end--;
    size_t range_len = (end - start) / 2 + 1;
    std::vector<bool> segment(range_len, true);
    uint64_t max_base_prime = (uint64_t)sqrt((double)end);
    int64_t* saved_all = all; 
    ll saved_pcnt = pcnt;
    for (ull i = 1; i < small_prime_lookup.size(); ++i) {
        uint64_t p = small_prime_lookup[i];
        if (p * p > end) break;
        uint64_t first_multiple = ((start + p - 1) / p) * p;
        if (first_multiple % 2 == 0) first_multiple += p;
        if (first_multiple < p * p) first_multiple = p * p;
        for (uint64_t m = first_multiple; m <= end; m += 2 * p) {
            segment[(m - start) / 2] = false;
        }
    }
    if (small_prime_lookup.back() < max_base_prime) {
        all = (int64_t*)malloc(((max_base_prime / 2) + 100) * sizeof(int64_t));
        pcnt = 0;
        for (ll base = THRESHOLD_2_24; base < (ll)max_base_prime + 100; base += sieve_span) {
            segment_sieve(base, max_base_prime + 100);
        }
        for (ll i = 0; i < pcnt; ++i) {
            uint64_t p = all[i];
            if (p * p > end) break;
            uint64_t first_multiple = ((start + p - 1) / p) * p;
            if (first_multiple % 2 == 0) first_multiple += p;
            if (first_multiple < p * p) first_multiple = p * p;
            for (uint64_t m = first_multiple; m <= end; m += 2 * p) {
                segment[(m - start) / 2] = false;
            }
        }
    }
    std::vector<uint64_t> survivors;
    for (size_t i = 0; i < range_len; ++i) {
        if (segment[i]) survivors.push_back(start + 2 * i);
    }
    free(all);
    all = saved_all;
    pcnt = saved_pcnt;
    int64_t md_idx = std::lower_bound(survivors.begin(), survivors.end(), md) - survivors.begin();
    int64_t target_idx = md_idx - prime_delta - 1;
    if (target_idx >= 0 && target_idx < (int64_t)survivors.size()) {
        return survivors[target_idx];
    }
    return 0;
}
uint64_t find_nth_prime(uint64_t n) {
    if (n < small_prime_lookup.size()) {
        return small_prime_lookup[n - 1];
    }
    uint64_t lb = Li_inverse<long double>(n);
    uint64_t ub = lb + 0.5 * Li_inverse(sqrtl(n));
    double log_n = log10((double)n);
    uint64_t md = (uint64_t)((double)lb * (1.0 / (log_n + 1.0)) + (double)ub * (log_n / (log_n + 1.0)));
    uint64_t k = (uint64_t)piSieve(md);
    int64_t prime_delta = (int64_t)k - (int64_t)n;
    double local_density = log((double)md);
    int64_t raw_offset = (int64_t)((double)prime_delta * local_density);
    uint64_t es = md - raw_offset;
    return micro_window_sieve(min(es - 50000, md), max(es + 50000, md), md, prime_delta);
}
int main(int argc, char* argv[]) {
    bool compute_pi = false;
    int nthreads = 8;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--pi") == 0) compute_pi = true;
        else if (strcmp(argv[i], "-t") == 0) {
            i++;
            sscanf(argv[i], "%d", &nthreads);
        }
    }
    initialize_small_lookup();
    uint64_t target = 1000000000000ULL;
    scanf("%" PRIu64, &target);
    if (compute_pi) printf("%" PRIu64 "\n", piSieve(target));
    else printf("%" PRIu64 "\n", find_nth_prime(target));
    return 0;
}
}