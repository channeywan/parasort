// 调度预热、排序计时和输出校验。
// 对应 DESIGN.md 第 4：统一运行框架 节。
// 实验配置与校验均沿用公共框架，算法默认参数保持不变。
#include "common.hpp"
#include "cli.hpp"
#include "datagen.hpp"
#include "verify.hpp"
#include <iostream>
namespace parasort {
int run_cpu(int argc, char **argv, const std::string &impl, const std::string &variant,
            const Sorter &sorter) {
    try {
        Config c = parse_cli(argc, argv, variant);
        if (c.help) {
            std::cout << help_text();
            return 0;
        }
        if (impl == "seq")
            c.threads = 1;
        auto a = buffer(c.n), out = buffer(c.n);
        bool serial = c.variant == "serial_init";
        first_touch(out.get(), c.n, c.threads, serial);
        for (int rep = -1; rep < c.repeat; ++rep) {
            generate(a.get(), c.n, 0, c, serial);
            Fingerprint before;
            std::vector<Key> exact;
            if (c.verify) {
                before = fingerprint(a.get(), c.n, c.threads);
                if (c.n <= 100000) {
                    exact.assign(a.get(), a.get() + c.n);
                    std::sort(exact.begin(), exact.end());
                }
            }
            Timer timer;
            SortResult r = sorter(a.get(), out.get(), c.n, c);
            double ms = timer.ms();
            if (!r.data)
                throw std::runtime_error("sorter did not return output");
            int verified =
                c.verify ? int(verify_result(r.data, c.n, before, exact, c.threads)) : -1;
            if (verified == 0)
                throw std::runtime_error("verification FAILED");
            if (rep >= 0 && output_fingerprint_requested())
                append_output_fingerprint(r, fingerprint(r.data, c.n, c.threads));
            if (rep >= 0)
                emit_csv(c, impl, r, rep, ms, verified);
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
} // namespace parasort
