// STREAM Copy/Triad provides measured bandwidth for the E14 experiment.
// Parallel first-touch follows the same static ownership as both kernels.
// This supports DESIGN.md's performance-analysis chapter and IMPROVEMENTS.md T5.
#include "cli.hpp"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace {
constexpr double scalar = 3.0;
constexpr int warmups = 1;
constexpr const char *header = "threads,bind,kernel,best_GBps,median_GBps,n,repeats,verified,host,"
                               "places,warmups,timestamp,bind_requested";

struct Options {
    uint64_t n = uint64_t(1) << 28;
    int threads = omp_get_max_threads(), repeats = 5;
    std::string csv = "results/E14.csv";
    bool help = false;
};

int positive(const std::string &text, const char *name) {
    uint64_t value = parasort::parse_size(text);
    if (!value || value > uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument(std::string(name) + " must be a positive int");
    return int(value);
}

Options parse(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string argument = argv[i];
        auto value = [&]() {
            if (i + 1 == argc)
                throw std::invalid_argument("missing value for " + argument);
            return std::string(argv[++i]);
        };
        if (argument == "--n")
            options.n = parasort::parse_size(value());
        else if (argument == "--threads")
            options.threads = positive(value(), "--threads");
        else if (argument == "--repeat")
            options.repeats = positive(value(), "--repeat");
        else if (argument == "--csv")
            options.csv = value();
        else if (argument == "--help" || argument == "-h")
            options.help = true;
        else
            throw std::invalid_argument("unknown option: " + argument);
    }
    if (!options.n)
        throw std::invalid_argument("--n must be positive");
    if (options.n > std::numeric_limits<size_t>::max() / (3 * sizeof(double)))
        throw std::length_error("three STREAM arrays exceed address space");
    return options;
}

std::string binding() {
    omp_proc_bind_t policy = omp_get_proc_bind();
    if (policy == omp_proc_bind_close)
        return "close";
    if (policy == omp_proc_bind_spread)
        return "spread";
    if (policy == omp_proc_bind_false)
        return "false";
    if (policy == omp_proc_bind_true)
        return "true";
    if (policy == omp_proc_bind_master)
        return "master";
    return "unknown";
}

std::string environment(const char *name) {
    const char *value = std::getenv(name);
    return value ? value : "unset";
}

void require_team(int actual, int requested) {
    if (actual != requested)
        throw std::runtime_error("OpenMP created " + std::to_string(actual) +
                                 " threads; requested " + std::to_string(requested));
}

void initialize(double *a, double *b, double *c, const Options &options) {
    int actual = 0;
#pragma omp parallel num_threads(options.threads)
    {
#pragma omp single nowait
        actual = omp_get_num_threads();
#pragma omp for schedule(static)
        for (uint64_t i = 0; i < options.n; ++i) {
            a[i] = 1.0;
            b[i] = 2.0;
            c[i] = 0.0;
        }
    }
    require_team(actual, options.threads);
}

double measure(double *a, const double *b, double *c, const Options &options, bool triad) {
    int actual = 0;
    // Wall time includes team launch and completion, but no validation or initialization.
    double start = omp_get_wtime();
#pragma omp parallel num_threads(options.threads)
    {
#pragma omp single nowait
        actual = omp_get_num_threads();
        if (triad) {
#pragma omp for schedule(static)
            for (uint64_t i = 0; i < options.n; ++i)
                a[i] = b[i] + scalar * c[i];
        } else {
#pragma omp for schedule(static)
            for (uint64_t i = 0; i < options.n; ++i)
                c[i] = a[i];
        }
    }
    double elapsed = omp_get_wtime() - start;
    require_team(actual, options.threads);
    if (!(elapsed > 0) || !std::isfinite(elapsed))
        throw std::runtime_error("STREAM kernel has invalid elapsed time");
    return elapsed;
}

void verify(const double *values, double expected, const Options &options, const char *kernel) {
    int bad = 0;
    double tolerance = 1e-12 * std::max(1.0, std::abs(expected));
#pragma omp parallel for num_threads(options.threads) schedule(static) reduction(| : bad)
    for (uint64_t i = 0; i < options.n; ++i)
        bad |= !std::isfinite(values[i]) || std::abs(values[i] - expected) > tolerance;
    if (bad)
        throw std::runtime_error(std::string(kernel) + " output verification FAILED");
}

std::string quote(const std::string &value) {
    std::string result = "\"";
    for (char ch : value) {
        result += ch;
        if (ch == '\"')
            result += '\"';
    }
    return result + '\"';
}

std::string timestamp() {
    std::time_t now = std::time(nullptr);
    std::tm utc{};
    gmtime_r(&now, &utc);
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

std::string row(const Options &options, const char *kernel, const std::vector<double> &elapsed,
                double bytes, const std::string &host, const std::string &when) {
    std::vector<double> rates;
    rates.reserve(elapsed.size());
    for (double seconds : elapsed)
        rates.push_back(bytes / seconds / 1e9);
    std::sort(rates.begin(), rates.end());
    size_t middle = rates.size() / 2;
    double median = rates.size() % 2 ? rates[middle] : (rates[middle - 1] + rates[middle]) / 2;
    std::ostringstream output;
    output << std::setprecision(12) << options.threads << ',' << binding() << ',' << kernel << ','
           << rates.back() << ',' << median << ',' << options.n << ',' << options.repeats << ",1,"
           << quote(host) << ',' << quote(environment("OMP_PLACES")) << ',' << warmups << ','
           << quote(when) << ',' << quote(environment("OMP_PROC_BIND"));
    return output.str();
}

void emit(const Options &options, const std::string &records) {
    if (!options.csv.empty()) {
        std::filesystem::path path(options.csv);
        if (path.has_parent_path())
            std::filesystem::create_directories(path.parent_path());
        bool empty = !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
        if (!empty) {
            std::ifstream existing(path);
            std::string first;
            if (!std::getline(existing, first) || first != header)
                throw std::runtime_error("existing STREAM CSV has a different header: " +
                                         options.csv);
        }
        std::ofstream output(path, std::ios::app);
        if (!output)
            throw std::runtime_error("cannot open STREAM CSV: " + options.csv);
        if (empty)
            output << header << '\n';
        output << records;
        output.flush();
        if (!output)
            throw std::runtime_error("STREAM CSV write failed: " + options.csv);
    }
    std::cout << header << '\n' << records;
}
} // namespace

int main(int argc, char **argv) {
    try {
        Options options = parse(argc, argv);
        if (options.help) {
            std::cout << "STREAM Copy/Triad --n <N|2^k> --threads T --repeat R --csv FILE\n"
                      << "Defaults: n=2^28 doubles per array, threads=OMP maximum, repeat=5, "
                      << "csv=results/E14.csv. One unrecorded warmup; every output is checked.\n"
                      << "Set OMP_PROC_BIND=close|spread and OMP_PLACES before launch.\n"
                      << "Copy counts 16n bytes; Triad a=b+3*c counts 24n bytes (decimal GB/s).\n";
            return 0;
        }
        omp_set_dynamic(0);
        // No value initialization: the first writes happen in the owning OpenMP threads.
        std::unique_ptr<double[]> a(new double[options.n]), b(new double[options.n]),
            c(new double[options.n]);
        initialize(a.get(), b.get(), c.get(), options);
        std::vector<double> copy, triad;
        copy.reserve(options.repeats);
        triad.reserve(options.repeats);
        double expected_a = 1.0;
        for (int64_t repetition = -warmups; repetition < options.repeats; ++repetition) {
            double copy_seconds = measure(a.get(), b.get(), c.get(), options, false);
            verify(c.get(), expected_a, options, "Copy");
            expected_a = 2.0 + scalar * expected_a;
            if (!std::isfinite(expected_a))
                throw std::runtime_error("Triad expectation overflows; reduce --repeat");
            double triad_seconds = measure(a.get(), b.get(), c.get(), options, true);
            verify(a.get(), expected_a, options, "Triad");
            if (repetition >= 0) {
                copy.push_back(copy_seconds);
                triad.push_back(triad_seconds);
            }
        }
        char host[256] = {};
        if (gethostname(host, sizeof(host) - 1) != 0)
            throw std::runtime_error("cannot read hostname");
        std::string when = timestamp();
        emit(options,
             row(options, "copy", copy, double(options.n) * 2 * sizeof(double), host, when) + '\n' +
                 row(options, "triad", triad, double(options.n) * 3 * sizeof(double), host, when) +
                 '\n');
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
