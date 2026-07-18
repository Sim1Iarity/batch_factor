#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sstream>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>
#include <chrono>
#include <ecm.h>
#include "gw_utility.c"
#define PARALLEL_FOR _Pragma("omp parallel for num_threads(8)")
extern "C" {
std::map<std::string, std::string> proc_fields(pid_t pid, std::string const & endpoint) {
    std::map<std::string, std::string> D;
    const std::string str = std::string("/proc/") + std::to_string(pid) + "/" + endpoint;
    std::ifstream f(str);
    for(std::string key ; f >> key ; ) {
        std::string value;
        key.erase(key.end() - 1);
        std::getline(f, value);
        D[key] = value;
    }
    return D;
}
std::map<unsigned long, int> mobius_table;
static int mobius(unsigned long n) {
    if (n == 0) return 0;
    if (mobius_table.count(n)) return mobius_table[n];
    unsigned long orig_n = n;
    unsigned long count = 0;
    if ((n & 1) == 0) {
        n >>= 1;
        count++;
        if ((n & 1) == 0) return 0;
    }
    for (unsigned long p = 3; p * p <= n; p += 2) {
        if (n % p == 0) {
            count++;
            n /= p;
            if (n % p == 0) return mobius_table[orig_n] = 0;
        }
    }
    if (n > 1) count++;
    return mobius_table[orig_n] = ((count & 1) ? -1 : 1);
}
static void get_divisors(unsigned long n, std::vector<unsigned long> &divs) {
    divs.clear();
    for (unsigned long d = 1; d * d <= n; d++) {
        if (n % d == 0) {
            divs.push_back(d);
            if (d != n / d) divs.push_back(n / d);
        }
    }
    std::sort(divs.begin(), divs.end());
}
static void cyclotomic_eval(mpz_t res, uint64_t n, uint64_t x) {
    if (n <= 0) {
        mpz_set_ui(res, 0);
        return;
    }
    if (n == 1) {
        mpz_set_ui(res, x - 1);
        return;
    }
    if (x == 1) {
        if (n == 1) mpz_set_ui(res, 0);
        else mpz_set_ui(res, 1);
        return;
    }
    std::vector<unsigned long> divs;
    get_divisors(n, divs);
    mpz_t numerator, denominator, term;
    mpz_inits(numerator, denominator, term, NULL);
    mpz_set_ui(numerator, 1);
    mpz_set_ui(denominator, 1);
    for (unsigned long d : divs) {
        int mu = mobius(n / d);
        if (mu == 0) continue;
        mpz_ui_pow_ui(term, x, d);
        mpz_sub_ui(term, term, 1);
        if (mu > 0) mpz_mul(numerator, numerator, term);
        else mpz_mul(denominator, denominator, term);
    }
    if (mpz_cmp_ui(denominator, 1) == 0) mpz_set(res, numerator);
    else mpz_divexact(res, numerator, denominator);
    mpz_clears(numerator, denominator, term, NULL);
}
void reduce_cyclotomic_index(uint64_t n, uint64_t &R, uint64_t &power_multiplier) {
    R = 1;
    uint64_t temp = n;
    if (temp % 2 == 0) {
        R *= 2;
        while (temp % 2 == 0) temp /= 2;
    }
    for (uint64_t p = 3; p * p <= temp; p += 2) {
        if (temp % p == 0) {
            R *= p;
            while (temp % p == 0) temp /= p;
        }
    }
    if (temp > 1) R *= temp;
    power_multiplier = n / R;
}
std::string get_cyclotomic_string(uint64_t n, uint64_t x) {
    if (n <= 0) return "0";
    if (n == 1) return std::to_string(x) + "-1";
    if (n == 2) return std::to_string(x) + "+1";
    if (x == 1) return "1";
    uint64_t R, power_multiplier;
    reduce_cyclotomic_index(n, R, power_multiplier);
    if (R == 1) return std::to_string(x) + "-1";
    if (R == 2) return std::to_string(x) + "^" + std::to_string(power_multiplier) + "+1";
    std::vector<unsigned long> divs;
    get_divisors(R, divs);
    std::map<uint64_t, bool> num_exponents;
    std::map<uint64_t, bool> den_exponents;
    for (unsigned long d : divs) {
        int mu = mobius(R / d);
        if (mu == 0) continue;
        uint64_t final_exponent = d * power_multiplier;
        if (mu > 0) num_exponents[final_exponent] = true;
        else den_exponents[final_exponent] = true;
    }
    std::vector<std::string> simplified_multipliers;
    for (auto it = num_exponents.begin(); it != num_exponents.end(); ) {
        uint64_t exp_2d = it->first;
        if (exp_2d % 2 == 0) {
            uint64_t exp_d = exp_2d / 2;
            if (den_exponents.count(exp_d)) {
                simplified_multipliers.push_back(std::to_string(x) + "^" + std::to_string(exp_d) + "+1");
                den_exponents.erase(exp_d);
                it = num_exponents.erase(it);
                continue;
            }
        }
        ++it;
    }
    std::vector<std::string> simplified_divisors;
    for (auto it = den_exponents.begin(); it != den_exponents.end(); ) {
        uint64_t exp_2d = it->first;
        if (exp_2d % 2 == 0) {
            uint64_t exp_d = exp_2d / 2;
            if (num_exponents.count(exp_d)) {
                simplified_divisors.push_back(std::to_string(x) + "^" + std::to_string(exp_d) + "+1");
                num_exponents.erase(exp_d);
                it = den_exponents.erase(it);
                continue;
            }
        }
        ++it;
    }
    std::vector<std::string> numerators, denominators;
    for (auto const& [exp, val] : num_exponents) numerators.push_back(std::to_string(x) + "^" + std::to_string(exp) + "-1");
    for (auto const& [exp, val] : den_exponents) denominators.push_back(std::to_string(x) + "^" + std::to_string(exp) + "-1");
    std::string ret = "";
    bool first = true;
    if (!numerators.empty()) {
        ret += "(" + numerators[0] + ")";
        for (size_t i = 1; i < numerators.size(); i++) ret += "*(" + numerators[i] + ")";
        first = false;
    }
    for (const std::string& term : simplified_multipliers) {
        if (first) {
            ret += "(" + term + ")";
            first = false;
        }
        else ret += "*(" + term + ")";
    }
    if (ret.empty() && (!denominators.empty() || !simplified_divisors.empty())) ret = "1";
    for (const std::string& term : denominators) ret += "/(" + term + ")";
    for (const std::string& term : simplified_divisors) ret += "/(" + term + ")";
    return ret;
}
size_t Memusage (void) {
    size_t mem;
    if (std::istringstream(proc_fields(getpid(), "status")["VmSize"]) >> mem) return mem;
    else return -1;
}
size_t PeakMemusage (void) {
    size_t mem;
    if (std::istringstream(proc_fields(getpid(), "status")["VmPeak"]) >> mem) return mem;
    else return -1;
}
#define TIMING(str, cpu, wct, ...) fprintf(stderr, str " took %.3lfms cpu (%.6lfms wct); memory %.3lfMB, peak %.3lfMB\n",\
    __VA_ARGS__, (clock() - cpu) * 1000.0 / CLOCKS_PER_SEC,\
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - wct).count() / 1e6,\
    Memusage() / 1024.0, PeakMemusage() / 1024.0);\
cpu = clock();\
wct = std::chrono::steady_clock::now()
static void tree_gcd(mpz_t *gcd, mpz_srcptr n, mpz_t *m, int no_m, clock_t t_cpu,
    std::chrono::steady_clock::time_point t_wct) {
    mpz_t **tree;
    int *width;
    int levels;
    int i, j;
    for (i = no_m, levels = 1; i > 1; i = (i + 1) / 2, levels++);
    tree = (mpz_t **)malloc(levels * sizeof(mpz_t *));
    width = (int *)malloc(levels * sizeof(int));
    width[0] = no_m;
    tree[1] = (mpz_t *)malloc(((no_m + 1) / 2) * sizeof(mpz_t));
    width[1] = (no_m + 1) / 2;
    clock_t original_t_cpu = t_cpu;
    std::chrono::steady_clock::time_point original_t_wct = t_wct;
    PARALLEL_FOR for (j = 0; j < no_m / 2; j++) {
        mpz_init(tree[1][j]);
        mpz_mul(tree[1][j], m[2 * j], m[2 * j + 1]);
    }
    TIMING("\t\tmul level 1 (%u fragments)", t_cpu, t_wct, no_m / 2);
    if (no_m % 2 != 0) mpz_init_set(tree[1][no_m / 2], m[no_m - 1]);
    for (i = 2; i < levels; i++) {
        width[i] = (width[i - 1] + 1) / 2;
        tree[i] = (mpz_t *)malloc(width[i] * sizeof(mpz_t));
        PARALLEL_FOR for (j = 0; j < width[i - 1] / 2; j++) {
            mpz_init(tree[i][j]);
            mpz_mul(tree[i][j], tree[i - 1][2 * j], tree[i - 1][2 * j + 1]);
        }
        j = width[i - 1] / 2;
        if (width[i - 1] % 2 != 0) {
            mpz_init_set(tree[i][j], tree[i - 1][2 * j]);
        }
        TIMING("\t\tmul level %d (%u fragments)", t_cpu,  t_wct, i, width[i]);
    }
    TIMING("\t%s", original_t_cpu, original_t_wct, "mul");
    mpz_mod(tree[levels - 1][0], n, tree[levels - 1][0]);
    TIMING("\t%s", t_cpu, t_wct, "main mod");
    for (i = levels - 2; i >= 1; i--) {
        PARALLEL_FOR for (j = 0; j < (width[i] / 2) * 2; j++) {
            mpz_mod(tree[i][j], tree[i + 1][j / 2], tree[i][j]);
        }
        j = (width[i] / 2) * 2;
        if (width[i] % 2 != 0) mpz_set(tree[i][j], tree[i + 1][j / 2]);
        TIMING("\t\tmod level %d (%u fragments)", t_cpu, t_wct, i, width[i]);
    }
    TIMING("\t%s", original_t_cpu, original_t_wct, "mod");
    PARALLEL_FOR for (j = 0; j < no_m; j++) {
        mpz_mod(gcd[j], tree[1][j / 2], m[j]);
        mpz_gcd(gcd[j], gcd[j], m[j]);
    }
    TIMING("\t%s", t_cpu, t_wct, "gcd");
    for (i = 1; i < levels; i++) {
        for (j = 0; j < width[i]; j++) mpz_clear(tree[i][j]);
        free(tree[i]);
    }
    free(tree);
    free(width);
}
#define EVALUATE_INIT(V) mpz_sub(var, var, V);\
length = mpz_get_ui(var) + 1;\
nums = (mpz_t*)malloc(length * sizeof(mpz_t));\
strs = (char**)malloc(length * sizeof(char*));\
for (size_t i = 0; i < length; i++) strs[i] = (char*)malloc(4096);\
TIMING("\t%s", t_cpu, t_wct, "malloc")
#define EVALUATE_INIT_INT(V) var_int -= V;\
length = var_int + 1;\
nums = (mpz_t*)malloc(length * sizeof(mpz_t));\
strs = (char**)malloc(length * sizeof(char*));\
for (size_t i = 0; i < length; i++) strs[i] = (char*)malloc(4096);\
TIMING("\t%s", t_cpu, t_wct, "malloc")
#define EVALUATE_CANONICALIZE for (size_t i = 0; i < length; i++) {\
    if (strs[i][0] == '1' && strs[i][1] == '*') strcpy(strs[i], strs[i] + 2);\
}
size_t evaluate(mpz_t*& nums, char**& strs, char* input, clock_t t_cpu,
    std::chrono::steady_clock::time_point t_wct) {
    mpz_t k, b, n, c, var, num;
    mpz_inits(k, b, n, c, var, num, NULL);
    size_t length = 0, k_int, b_int, var_int;
    if (gmp_sscanf(input, "(%Zd-%Zd)*%Zd^%Zd%Zd", &k, &var, &b, &n, &c) == 5 && mpz_cmp_ui(c, 0) != 0) {
        EVALUATE_INIT(k);
        mpz_pow_ui(num, b, mpz_get_ui(n));
        mpz_init(nums[0]);
        mpz_mul(nums[0], num, k);
        for (size_t i = 1; i < length; i++) {
            mpz_init(nums[i]);
            mpz_add(nums[i], nums[i - 1], num);
        }
        PARALLEL_FOR for (size_t i = 0; i < length; i++) mpz_add(nums[i], nums[i], c);
        if (mpz_sgn(c) < 0) {
            mpz_neg(c, c);
            mpz_set(num, k);
            for (size_t i = 0; i < length; i++) {
                gmp_sprintf(strs[i], "%Zd*%Zd^%Zd-%Zd", num, b, n, c);
                mpz_add_ui(num, num, 1);
            }
        }
        else {
            mpz_set(num, k);
            for (size_t i = 0; i < length; i++) {
                gmp_sprintf(strs[i], "%Zd*%Zd^%Zd+%Zd", num, b, n, c);
                mpz_add_ui(num, num, 1);
            }
        }
        EVALUATE_CANONICALIZE;
        goto evaluate_cleanup;
    }
    if (gmp_sscanf(input, "%Zd*(%Zd-%Zd)^%Zd%Zd", &k, &b, &var, &n, &c) == 5 && mpz_cmp_ui(c, 0) != 0) {
        EVALUATE_INIT(b);
        for (size_t i = 0; i < length; i++) {
            mpz_init(nums[i]);
            mpz_pow_ui(nums[i], b, mpz_get_ui(n));
            mpz_add_ui(b, b, 1);
        }
        PARALLEL_FOR for (size_t i = 0; i < length; i++) {
            mpz_mul(nums[i], nums[i], k);
            mpz_add(nums[i], nums[i], c);
        }
        mpz_sub_ui(b, b, length);
        if (mpz_sgn(c) < 0) {
            mpz_neg(c, c);
            mpz_set(num, b);
            for (size_t i = 0; i < length; i++) {
                gmp_sprintf(strs[i], "%Zd*%Zd^%Zd-%Zd", k, num, n, c);
                mpz_add_ui(num, num, 1);
            }
        }
        else {
            mpz_set(num, b);
            for (size_t i = 0; i < length; i++) {
                gmp_sprintf(strs[i], "%Zd*%Zd^%Zd+%Zd", k, num, n, c);
                mpz_add_ui(num, num, 1);
            }
        }
        EVALUATE_CANONICALIZE;
        goto evaluate_cleanup;
    }
    if (gmp_sscanf(input, "%Zd*%Zd^(%Zd-%Zd)%Zd", &k, &b, &n, &var, &c) == 5 && mpz_cmp_ui(c, 0) != 0) {
        EVALUATE_INIT(n);
        mpz_pow_ui(num, b, mpz_get_ui(n));
        mpz_mul(num, num, k);
        mpz_init_set(nums[0], num);
        for (size_t i = 1; i < length; i++) {
            mpz_mul(num, num, b);
            mpz_init_set(nums[i], num);
        }
        PARALLEL_FOR for (size_t i = 0; i < length; i++) mpz_add(nums[i], nums[i], c);
        if (mpz_sgn(c) < 0) {
            mpz_neg(c, c);
            mpz_set(num, n);
            for (size_t i = 0; i < length; i++) {
                gmp_sprintf(strs[i], "%Zd*%Zd^%Zd-%Zd", k, b, num, c);
                mpz_add_ui(num, num, 1);
            }
        }
        else {
            mpz_set(num, n);
            for (size_t i = 0; i < length; i++) {
                gmp_sprintf(strs[i], "%Zd*%Zd^%Zd+%Zd", k, b, num, c);
                mpz_add_ui(num, num, 1);
            }
        }
        EVALUATE_CANONICALIZE;
        goto evaluate_cleanup;
    }
    if (gmp_sscanf(input, "%Zd*%Zd^%Zd+(%Zd-%Zd)", &k, &b, &n, &c, &var) == 5 && mpz_cmp_ui(c, 0) != 0) {
        EVALUATE_INIT(c);
        mpz_pow_ui(num, b, mpz_get_ui(n));
        mpz_mul(num, num, k);
        mpz_add(num, num, c);
        mpz_init_set(nums[0], num);
        for (size_t i = 1; i < length; i++) {
            mpz_add_ui(num, num, 1);
            mpz_init_set(nums[i], num);
        }
        mpz_set(num, c);
        for (size_t i = 0; i < length; i++) {
            gmp_sprintf(strs[i], "%Zd*%Zd^%Zd+%Zd", k, b, n, num);
            mpz_add_ui(num, num, 1);
        }
        EVALUATE_CANONICALIZE;
        goto evaluate_cleanup;
    }
    if (gmp_sscanf(input, "%Zd*%Zd^%Zd-(%Zd-%Zd)", &k, &b, &n, &var, &c) == 5 && mpz_cmp_ui(c, 0) != 0) {
        mpz_neg(c, c); mpz_neg(var, var);
        EVALUATE_INIT(c);
        mpz_pow_ui(num, b, mpz_get_ui(n));
        mpz_mul(num, num, k);
        mpz_add(num, num, c);
        mpz_init_set(nums[0], num);
        for (size_t i = 1; i < length; i++) {
            mpz_add_ui(num, num, 1);
            mpz_init_set(nums[i], num);
        }
        mpz_set(num, c);
        for (size_t i = 0; i < length; i++) {
            gmp_sprintf(strs[i], "%Zd*%Zd^%Zd%Zd", k, b, n, num);
            mpz_add_ui(num, num, 1);
        }
        EVALUATE_CANONICALIZE;
        goto evaluate_cleanup;
    }
    if (sscanf(input, "Phi(%lu-%lu,%lu)", &k_int, &var_int, &b_int) == 3) {
        EVALUATE_INIT_INT(k_int);
        PARALLEL_FOR for (size_t i = 0; i < length; i++) {
            mpz_init(nums[i]);
            cyclotomic_eval(nums[i], k_int + i, b_int);
            std::string tmp = get_cyclotomic_string(k_int + i, b_int);
            strcpy(strs[i], tmp.c_str());
        }
        goto evaluate_cleanup;
    }
    if (sscanf(input, "Phi(%lu,%lu-%lu)", &k_int, &b_int, &var_int) == 3) {
        EVALUATE_INIT_INT(b_int);
        PARALLEL_FOR for (size_t i = 0; i < length; i++) {
            mpz_init(nums[i]);
            cyclotomic_eval(nums[i], k_int, b_int + i);
            std::string tmp = get_cyclotomic_string(k_int, b_int + i);
            strcpy(strs[i], tmp.c_str());
        }
        goto evaluate_cleanup;
    }
evaluate_cleanup:
    mpz_clears(k, b, n, c, var, NULL);
    return length;
}
int main(int argc, char* argv[]) {
    mpz_t* nums;
    size_t length;
    if (argc == 1) {
        printf("Usage: %s [-o file] expression\n", argv[0]);
        exit(0);
    }
    FILE *outf = stdout, *stdout = stdin;
    bool check_primality = false;
    int pm1_curves = 0, ecm_curves = 0;
    char* arg = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0) {
            i++;
            outf = fopen(argv[i], "wb");
            continue;
        }
        if (strcmp(argv[i], "-p") == 0) {
            check_primality = true;
            i++;
            primf = fopen(argv[i], "wb");
            continue;
        }
        if (strcmp(argv[i], "--pm1") == 0) {
            i++;
            sscanf(argv[i], "%d", &pm1_curves);
            continue;
        }
        if (strcmp(argv[i], "--ecm") == 0) {
            i++;
            sscanf(argv[i], "%d", &ecm_curves);
            continue;
        }
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [-o outfile] [--pm1 B1pm1] [--ecm B1ecm] [-p primfile] expression\n", argv[0]);
            printf("Options:\n");
            printf("\t-o outfile  output the results to file (default stdout)\n");
            printf("\t-p primfile output the expression of probable primes to primfile (default stdout)\n");
            printf("\t--pm1 B1pm1 perform P-1 factorization with B1=B1pm1 on the numbers\n");
            printf("\t--ecm B1ecm perform ECM factorization with B1=B1ecm on the numbers\n");
            printf("\nCurrently supported expression types:\n");
            printf("\t(a-b)*c^d+-e\n");
            printf("\ta*(b-c)^d+-e\n");
            printf("\ta*b^c+-(d-e)\n");
            printf("\tPhi(a-b,c)\n");
            printf("\tPhi(a,b-c)\n");
            printf("Where Phi(n,x) is the n-th cyclotomic polynomial evaluated at x.\n")
            exit(1);
        }
        if (arg != NULL) {
            puts("Error: multiple expressions");
            printf("Usage: %s [-o outfile] [--pm1 B1pm1] [--ecm B1ecm] [-p primfile] expression\n", argv[0]);
            exit(1);
        }
        arg = argv[i];
    }
    if (arg == NULL) {
        printf("Usage: %s [-o outfile] [--pm1 B1pm1] [--ecm B1ecm] [-p primfile] expression\n", argv[0]);
        exit(1);
    }
    FILE *f = fopen(getenv("PRIM_GIANT"), "rb");
    if (f == NULL) {
        printf("Error: No primorial file found\n");
        printf("Use the \"primorial\" utility to generate the primorial file\n");
        printf("and set PRIM_GIANT to the absolute path of that file.\n");
        exit(1);
    }
    char** strs;
    clock_t t_cpu = clock();
    std::chrono::steady_clock::time_point t_wct = std::chrono::steady_clock::now();
    const clock_t original_t_cpu = t_cpu;
    const std::chrono::steady_clock::time_point original_t_wct = t_wct;
    length = evaluate(nums, strs, arg, t_cpu, t_wct);
    mpz_t* denoms = (mpz_t*)malloc(length * sizeof(mpz_t));
    for (size_t i = 0; i < length; i++) mpz_init_set_ui(denoms[i], 1ul);
    mpz_t* gcds = (mpz_t*)malloc(length * sizeof(mpz_t));
    for (size_t i = 0; i < length; i++) mpz_init_set_ui(gcds[i], 1ul);
    TIMING("%s", t_cpu, t_wct, "evaluation");
    mpz_t n;
    mpz_init(n);
    size_t res = mpz_inp_raw(n, f);
    fclose(f);
    TIMING("loading primorial (%lu chars)", t_cpu, t_wct, res);
    tree_gcd(gcds, n, nums, length, t_cpu, t_wct);
    TIMING("%s", t_cpu, t_wct, "tree gcd");
    res = 0;
    PARALLEL_FOR for (size_t i = 0; i < length; i++) {
        while (mpz_cmp_ui(gcds[i], 1) != 0) {
            mpz_mul(denoms[i], denoms[i], gcds[i]);
            mpz_div(nums[i], nums[i], gcds[i]);
            mpz_gcd(gcds[i], nums[i], gcds[i]);
        }
        fprintf(stderr, "%lu/%lu\r", ++res, length);
    }
    TIMING("%s", t_cpu, t_wct, "trial division");
    if (pm1_curves) {
        double B1 = pm1_curves;
        res = 0;
        PARALLEL_FOR for (size_t i = 0; i < length; i++) {
            ecm_params param;
            ecm_init(param);
            param->method = ECM_PM1;
            if (ecm_factor(gcds[i], nums[i], B1, param) != ECM_NO_FACTOR_FOUND) {
                while (mpz_cmp_ui(gcds[i], 1) != 0) {
                    mpz_mul(denoms[i], denoms[i], gcds[i]);
                    mpz_div(nums[i], nums[i], gcds[i]);
                    mpz_gcd(gcds[i], nums[i], gcds[i]);
                }
            }
            ecm_clear(param);
            fprintf(stderr, "%lu/%lu\r", ++res, length);
        }
        TIMING("%s", t_cpu, t_wct, "P-1");
    }
    if (ecm_curves) {
        double B1 = ecm_curves;
        res = 0;
        PARALLEL_FOR for (size_t i = 0; i < length; i++) {
            for (size_t t = 0; t < 5; t++) {
                ecm_params param;
                ecm_init(param);
                param->method = ECM_ECM;
                if (ecm_factor(gcds[i], nums[i], B1, param) != ECM_NO_FACTOR_FOUND) {
                    while (mpz_cmp_ui(gcds[i], 1) != 0) {
                        mpz_mul(denoms[i], denoms[i], gcds[i]);
                        mpz_div(nums[i], nums[i], gcds[i]);
                        mpz_gcd(gcds[i], nums[i], gcds[i]);
                    }
                }
                ecm_clear(param);
            }
            fprintf(stderr, "%lu/%lu\r", ++res, length);
        }
        TIMING("%s", t_cpu, t_wct, "ECM");
    }
    for (size_t i = 0; i < length; i++) {
        if (__gmpz_cmp_ui(denoms[i], 1) == 0) fputs(strs[i], outf);
        else gmp_fprintf(outf, "(%s)/%Zd", strs[i], denoms[i]);
        fputc('\n', outf);
    }
    fclose(outf);
    if (check_primality) {
        char* primality = (char*)malloc(length);
        size_t cnt = 0;
        PARALLEL_FOR for (size_t i = 0; i < length; i++) {
            primality[i] = gw_prp(nums[i]);
            fprintf(stderr, "%lu/%lu\r", ++cnt, length);
        }
        TIMING("%s", t_cpu, t_wct, "primality");
        for (size_t i = 0; i < length; i++) if (primality[i]) {
            if (__gmpz_cmp_ui(denoms[i], 1) == 0) fputs(strs[i], primf);
            else gmp_fprintf(primf, "(%s)/%Zd", strs[i], denoms[i]);
            fputc('\n', primf);
            fflush(primf);
        }
        free(primality);
    }
    for (size_t i = 0; i < length; i++) {
        mpz_clears(nums[i], denoms[i], gcds[i], NULL);
        free(strs[i]);
    }
    free(nums);
    free(denoms);
    free(gcds);
    free(strs);
    fprintf(stderr, "Total: CPU %.3lfms, WCT %.6lfms; memory %.3lfMB, peak %.3lfMB\n", (clock() - original_t_cpu) * 1000.0 / CLOCKS_PER_SEC,
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - original_t_wct).count() / 1e6,
        Memusage() / 1024.0, PeakMemusage() / 1024.0);
}
}