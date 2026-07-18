#include <pari/pari.h>
#include <string.h>
#include <vector>
#include <gmpxx.h>
#include <fstream>
#include <ecm.h>
#include <signal.h>
#include <chrono>
#include "gw_utility.c"
#define PARALLEL_FOR _Pragma("omp parallel for num_threads(8)")
size_t lefts[129], rights[129];
size_t *indexs;
std::vector<mpz_class> nums, gcds;
std::vector<std::string> strs;
static int prime_arr[128];
static void Z_get_mpz (mpz_ptr z, GEN x) {
    const long l = lgefint (x) - 2;
    int i;
    _mpz_realloc (z, l);
    z->_mp_size = (signe (x) > 0 ? l : -l);
    for (i = 0; i < l; i++)
        (z->_mp_d) [i] = *int_W (x, i);
}
void cm_pari_eval_int (mpz_ptr n, char *e) {
    pari_sp av;
    GEN z;
    av = avma;
    z = gp_read_str (e);
    Z_get_mpz (n, z);
    avma = av;
}
void pari_eval_crash(int) {
    fprintf(stderr, "\033[1;31mError: input has non-integer values\nCheck the input\033[0m\n");
    _exit(1);
}
bool cmp_ulul(size_t i, size_t j) { return nums[i] < nums[j]; }
const char* stat[] = {"composite", "prp", "prime"};
#define ECM_LEVEL(lvlstr, B1, ncurves) \
size_t res = 0;\
PARALLEL_FOR for (size_t i = 0; i < nums.size(); i++) { \
    if (gcds[i] == 1) for (int t = 0; t < (ncurves); t++) { \
        ecm_params param; \
        ecm_init(param); \
        param->method = ECM_ECM; \
        if (!gw_prp(nums[i].get_mpz_t()) && ecm_factor(gcds[i].get_mpz_t(), nums[i].get_mpz_t(), (B1), param) != ECM_NO_FACTOR_FOUND) { \
            gmp_fprintf(stderr, "%s has a %s factor of %d digits: %Zd\n", strs[i].c_str(), stat[gw_prp(gcds[i].get_mpz_t())], \
                mpz_sizeinbase(gcds[i].get_mpz_t(), 10), gcds[i].get_mpz_t()); \
            nfac++; \
            break; \
        } \
        ecm_clear(param); \
    } \
    fprintf(stderr, lvlstr " %lu/%lu cracked %lu nums\t\t\t\t\r", ++res, nums.size(), nfac); \
}
void perform_ecm(double tlevel, int pm1B1) {
    size_t nfac = 0;
    if (pm1B1) {
        size_t res = 0;
        PARALLEL_FOR for (size_t i = 0; i < nums.size(); i++) {
            ecm_params param;
            ecm_init(param);
            param->method = ECM_ECM;
            if (!gw_prp(nums[i].get_mpz_t()) && ecm_factor(gcds[i].get_mpz_t(), nums[i].get_mpz_t(), pm1B1, param) != ECM_NO_FACTOR_FOUND) {
                gmp_fprintf(stderr, "%s has a %s factor of %d digits: %Zd\n", strs[i].c_str(), stat[gw_prp(gcds[i].get_mpz_t())],
                    mpz_sizeinbase(gcds[i].get_mpz_t(), 10), gcds[i].get_mpz_t());
                nfac++;
            }
            ecm_clear(param);
            fprintf(stderr, "P-1 %lu/%lu cracked %lu nums\t\t\t\t\r", ++res, nums.size(), nfac);
        }
    }
    if (tlevel >= 10 && tlevel <= 15) {
        ECM_LEVEL("T1", 2000, (30 * (tlevel - 10) / 5));
    }
    if (tlevel > 15 && tlevel <= 20) {
        ECM_LEVEL("T2", 11000, (74 * (tlevel - 15) / 5));
    }
    if (tlevel > 20 && tlevel <= 25) {
        ECM_LEVEL("T3", 50000, (214 * (tlevel - 20) / 5));
    }
    if (tlevel > 25 && tlevel <= 30) {
        ECM_LEVEL("T4", 250000, (430 * (tlevel - 25) / 5));
    }
}
int main(int argc, char* argv[]) {
    bool sort = false, pure_num = false, out_num = false;
    int prove = 0, pm1B1 = 0;
    double tlevel = 0;
    std::ios::sync_with_stdio(false);
    std::ifstream inf;
    std::ofstream outf;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0) {
            i++;
            inf = std::ifstream(argv[i], std::ios_base::in | std::ios::binary);
            continue;
        }
        if (strcmp(argv[i], "-o") == 0) {
            i++;
            outf = std::ofstream(argv[i], std::ios_base::app | std::ios::binary);
            continue;
        }
        if (strcmp(argv[i], "-s") == 0) {
            sort = true;
            continue;
        }
        if (strcmp(argv[i], "-p") == 0) {
            prove = 1;
            continue;
        }
        if (strcmp(argv[i], "--ecm") == 0) {
            prove = -1;
            i++;
            sscanf(argv[i], "%lf", &tlevel);
            continue;
        }
        if (strcmp(argv[i], "--pm1") == 0) {
            prove = -1;
            i++;
            sscanf(argv[i], "%d", &pm1B1);
            continue;
        }
        if (strcmp(argv[i], "--pure-num") == 0) {
            pure_num = true;
            continue;
        }
        if (strcmp(argv[i], "--out-num") == 0) {
            out_num = true;
            continue;
        }
    }
    pari_init_opts (1ul<<23, 0, INIT_JMPm | INIT_DFTm | INIT_noIMTm);
    paristack_setsize (1ul<<23, 1ul<<31);
    if (sizeof (long int) * CHAR_BIT >= 64)
        sd_threadsizemax ("1G", d_SILENT);
    pari_mt_init ();
    auto t = std::chrono::steady_clock::now();
    size_t done = 0;
    signal(SIGFPE, pari_eval_crash);
    while (!inf.eof()) {
        std::string tmp;
        std::getline(inf, tmp);
        if (tmp.empty()) continue;
        nums.push_back(mpz_class{0});
        gcds.push_back(mpz_class{1});
        if (pure_num) mpz_set_str(nums.back().get_mpz_t(), tmp.data(), 10);
        else cm_pari_eval_int(nums.back().get_mpz_t(), tmp.data());
        if (out_num) {
            strs.push_back("");
            strs.back().resize(mpz_sizeinbase(nums.back().get_mpz_t(), 10) + 2);
            mpz_get_str(strs.back().data(), 10, nums.back().get_mpz_t());
            while (strs.back().back() == '\0') strs.back().pop_back();
        }
        else strs.push_back(tmp);
        if (++done % 1000 == 0) fprintf(stderr, "%lu\r", done);
    }
    signal(SIGFPE, NULL);
    indexs = (size_t*)malloc(nums.size() * sizeof(size_t));
    for (size_t i = 0; i < nums.size(); i++) indexs[i] = i;
    fprintf(stderr, "evaluation of %lu numbers took %.6lf ms\n", nums.size(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t).count() / 1e6);
    t = std::chrono::steady_clock::now();
    pari_close ();
    if (!sort) goto output_entry;
    for (size_t i = 0; i <= 128; i++) {
        lefts[i] = nums.size() * i / 128;
        rights[i] = nums.size() * (i + 1) / 128;
    }
    PARALLEL_FOR for (size_t i = 0; i < 128; i++) {
        std::sort(indexs + lefts[i], indexs + rights[i], cmp_ulul);
        fprintf(stderr, "%lu/128\r", ++done);
    }
    fprintf(stderr, "128 fragments took %.6lf ms\n", std::chrono::duration_cast<std::chrono::nanoseconds>
        (std::chrono::steady_clock::now() - t).count() / 1e6);
    t = std::chrono::steady_clock::now();
    for (size_t level = 1, j = 2; j <= 128; j <<= 1, level++) {
        done = 0;
        size_t jend = 128 / j;
        PARALLEL_FOR for (size_t i = 0; i < jend; i++) {
            std::inplace_merge(indexs + lefts[i << level], indexs + lefts[(i << level) | (1 << (level - 1))], indexs + lefts[(i + 1) << level], cmp_ulul);
        }
        fprintf(stderr, "level %lu (%lu fragments) took %.6lf ms\n", level, jend, std::chrono::duration_cast<std::chrono::nanoseconds>
            (std::chrono::steady_clock::now() - t).count() / 1e6);
        t = std::chrono::steady_clock::now();
    }
output_entry:
    perform_ecm(tlevel, pm1B1);
    size_t nprp = 0;
    for (size_t i = 0; i < nums.size() >> 7; i++) {
        if (!prove) {
            for (size_t j = 0; j < 128; j++) outf << strs[indexs[i << 7 | j]] << '\n';
        }
        else if (prove == 1) {
            memset(prime_arr, 0, sizeof(prime_arr));
            const bool output = mpz_sizeinbase(nums[indexs[i << 7]].get_mpz_t(), 2) > 10000;
            size_t done = 0;
            PARALLEL_FOR for (size_t j = 0; j < 128; j++) {
                prime_arr[j] = gw_prp(nums[indexs[i << 7 | j]].get_mpz_t());
                if (prime_arr[j]) ++nprp;
                if (output) fprintf(stderr, "tested %lu numbers, found %lu prps\r", i << 7 | (++done), nprp);
            }
            for (size_t j = 0; j < 128; j++) if (prime_arr[j]) outf << strs[indexs[i << 7 | j]] << '\n';
            fprintf(stderr, "tested %lu numbers, found %lu prps\r", (i + 1) << 7, nprp);
        }
        else {
            memset(prime_arr, 0, sizeof(prime_arr));
            const bool output = mpz_sizeinbase(nums[indexs[i << 7]].get_mpz_t(), 2) > 10000;
            size_t done = 0;
            PARALLEL_FOR for (size_t j = 0; j < 128; j++) {
                prime_arr[j] = gw_prp(nums[indexs[i << 7 | j]].get_mpz_t());
                if (prime_arr[j]) ++nprp;
                if (output) fprintf(stderr, "tested %lu numbers, found %lu prps\r", i << 7 | (++done), nprp);
            }
            for (size_t j = 0; j < 128; j++) {
                if (gcds[indexs[i << 7 | j]] == 1) outf << strs[indexs[i << 7 | j]] << '\n';
                else outf << strs[indexs[i << 7 | j]] << "/" << gcds[indexs[i << 7 | j]] << '\n';
            }
            fprintf(stderr, "tested %lu numbers, found %lu composites\r", (i + 1) << 7, ((i + 1) << 7) - nprp);
        }
        outf.flush();
    }
    size_t i = nums.size() >> 7;
    if (prove == 0) {
        for (size_t j = 0; j < nums.size() % 128; j++) outf << strs[indexs[i << 7 | j]] << '\n';
    }
    else if (prove == 1) {
        memset(prime_arr, 0, sizeof(prime_arr));
        PARALLEL_FOR for (size_t j = 0; j < nums.size() % 128; j++) {
            prime_arr[j] = gw_prp(nums[indexs[i << 7 | j]].get_mpz_t());
        }
        for (size_t j = 0; j < nums.size() % 128; j++) if (prime_arr[j]) outf << strs[indexs[i << 7 | j]] << '\n';
    }
    else {
        memset(prime_arr, 0, sizeof(prime_arr));
        PARALLEL_FOR for (size_t j = 0; j < nums.size() % 128; j++) {
            prime_arr[j] = gw_prp(nums[indexs[i << 7 | j]].get_mpz_t());
        }
        for (size_t j = 0; j < nums.size() % 128; j++) {
            if (gcds[indexs[i << 7 | j]] == 1) outf << strs[indexs[i << 7 | j]] << '\n';
            else outf << strs[indexs[i << 7 | j]] << "/" << gcds[indexs[i << 7 | j]] << '\n';
        }
    }
    return 0;
}