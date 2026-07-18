#include <fstream>
#include <pari/pari.h>
#include <stdio.h>
#include <gmpxx.h>
#include <stdlib.h>
#include <vector>
#include <thread>
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
std::vector<mpz_class> nums;
void work(size_t id) {
    std::string cmd = std::string("ecpp -n ") + nums[id % 8].get_str() + " -f cert" + std::to_string(id) + ".out -t";
    system(cmd.c_str());
    cmd = std::string("echo 'write(\"cert") + std::to_string(id) + ".pari\",primecertexport(read(\"cert" + std::to_string(id) + ".out\"),1));quit;' | gp -q -s 1073748124";
    system(cmd.c_str());
    cmd = std::string("rm -f cert") + std::to_string(id) + ".out*";
    system(cmd.c_str());
}
int main(int argc, char* argv[]) {
    std::ios::sync_with_stdio(false);
    std::ifstream inf;
    size_t done = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0) {
            i++;
            inf = std::ifstream(argv[i], std::ios_base::in | std::ios::binary);
            continue;
        }
        if (strcmp(argv[i], "-d") == 0) {
            i++;
            sscanf(argv[i], "%lu", &done);
            continue;
        }
    }
    pari_init_opts (1ul<<23, 0, INIT_JMPm | INIT_DFTm | INIT_noIMTm);
    paristack_setsize (1ul<<23, 1ul<<31);
    if (sizeof (long int) * CHAR_BIT >= 64)
        sd_threadsizemax ("1G", d_SILENT);
    pari_mt_init ();
    while (!inf.eof()) {
        std::string tmp;
        std::getline(inf, tmp);
        nums.push_back(mpz_class{0});
        cm_pari_eval_int(nums.back().get_mpz_t(), tmp.data());
        if (nums.size() == 8) {
            std::vector<std::thread> vec;
            for (size_t i = 0; i < 8; i++) vec.emplace_back(work, done + i);
            for (size_t i = 0; i < 8; i++) vec[i].join();
            done += 8;
            nums.clear();
        }
    }
    std::vector<std::thread> vec;
    for (size_t i = 0; i < nums.size(); i++) vec.emplace_back(work, done + i);
    for (size_t i = 0; i < nums.size(); i++) vec[i].join();
}