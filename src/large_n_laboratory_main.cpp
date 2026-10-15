#include "nbody/large_n_laboratory.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

#ifndef NBODY_BUILD_CONFIGURATION
#define NBODY_BUILD_CONFIGURATION "unknown"
#endif
#ifndef NBODY_CXX_COMPILER_ID
#define NBODY_CXX_COMPILER_ID "unknown"
#endif
#ifndef NBODY_CXX_COMPILER_VERSION
#define NBODY_CXX_COMPILER_VERSION "unknown"
#endif

struct Options {
    std::string output_path;
    nbody::LargeNExperimentConfiguration experiment;
    bool bodies_set{false};
    bool seed_set{false};
};

std::size_t parse_size(const std::string& text, const char* label) {
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    std::size_t consumed = 0;
    const unsigned long long raw = std::stoull(text, &consumed, 10);
    if (consumed != text.size() || raw > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<std::size_t>(raw);
}

std::uint64_t parse_u64(const std::string& text, const char* label) {
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    std::size_t consumed = 0;
    const unsigned long long raw = std::stoull(text, &consumed, 0);
    if (consumed != text.size()) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<std::uint64_t>(raw);
}

int parse_int(const std::string& text, const char* label) {
    std::size_t consumed = 0;
    const long raw = std::stol(text, &consumed, 10);
    if (consumed != text.size() || raw < std::numeric_limits<int>::min() ||
        raw > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<int>(raw);
}

double parse_double(const std::string& text, const char* label) {
    std::size_t consumed = 0;
    const double raw = std::stod(text, &consumed);
    if (consumed != text.size()) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return raw;
}

bool parse_bool(const std::string& text, const char* label) {
    if (text == "true" || text == "1") {
        return true;
    }
    if (text == "false" || text == "0") {
        return false;
    }
    throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
}

nbody::LargeNDistribution parse_distribution(const std::string& text) {
    if (text == "uniform") return nbody::LargeNDistribution::uniform;
    if (text == "clustered") return nbody::LargeNDistribution::clustered;
    if (text == "planar") return nbody::LargeNDistribution::planar;
    throw std::invalid_argument("unknown distribution: " + text);
}

nbody::LargeNBackend parse_backend(const std::string& text) {
    if (text == "sequential") return nbody::LargeNBackend::sequential;
    if (text == "openmp") return nbody::LargeNBackend::openmp;
    if (text == "cuda") return nbody::LargeNBackend::cuda;
    throw std::invalid_argument("unknown backend: " + text);
}

nbody::LargeNSolver parse_solver(const std::string& text) {
    if (text == "direct") return nbody::LargeNSolver::direct;
    if (text == "barnes-hut") return nbody::LargeNSolver::barnes_hut;
    throw std::invalid_argument("unknown solver: " + text);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            std::cout
                << "Usage: " << argv[0] << " --output PATH --bodies N --seed U64 "
                   "--distribution uniform|clustered|planar "
                   "--backend sequential|openmp|cuda --solver direct|barnes-hut [options]\n"
                   "Supported requested sizes include 256, 1024, 4096, 8192, 16384, "
                   "32768, 50000, and 100000; admission limits still apply.\n"
                   "Options: --theta X --threads N --dt S --steps N --warmups N "
                   "--repetitions N --direct-reference true|false "
                   "--max-direct-reference-bodies N --memory-budget-bytes N "
                   "--max-direct-interactions N --wall-time-budget-s S "
                   "--maximum-nodes N --maximum-traversal-work N "
                   "--solver-memory-budget-bytes N\n";
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + argument);
        }
        const std::string value = argv[++index];
        if (argument == "--output") {
            options.output_path = value;
        } else if (argument == "--bodies") {
            options.experiment.generation.body_count = parse_size(value, "body count");
            options.bodies_set = true;
        } else if (argument == "--seed") {
            options.experiment.generation.seed = parse_u64(value, "seed");
            options.seed_set = true;
        } else if (argument == "--distribution") {
            options.experiment.generation.distribution = parse_distribution(value);
        } else if (argument == "--backend") {
            options.experiment.backend = parse_backend(value);
        } else if (argument == "--solver") {
            options.experiment.solver = parse_solver(value);
        } else if (argument == "--theta") {
            options.experiment.barnes_hut.theta = parse_double(value, "theta");
        } else if (argument == "--threads") {
            options.experiment.requested_threads = parse_int(value, "thread count");
        } else if (argument == "--dt") {
            options.experiment.timestep_s = parse_double(value, "timestep");
        } else if (argument == "--steps") {
            options.experiment.steps = parse_size(value, "step count");
        } else if (argument == "--warmups") {
            options.experiment.warmups = parse_size(value, "warmup count");
        } else if (argument == "--repetitions") {
            options.experiment.repetitions = parse_size(value, "repetition count");
        } else if (argument == "--direct-reference") {
            options.experiment.request_direct_reference =
                parse_bool(value, "direct-reference flag");
        } else if (argument == "--max-direct-reference-bodies") {
            options.experiment.resources.maximum_direct_reference_bodies =
                parse_size(value, "direct-reference body limit");
        } else if (argument == "--memory-budget-bytes") {
            options.experiment.resources.memory_budget_bytes =
                parse_size(value, "memory budget");
        } else if (argument == "--max-direct-interactions") {
            options.experiment.resources.maximum_direct_interactions =
                parse_u64(value, "direct interaction budget");
        } else if (argument == "--wall-time-budget-s") {
            options.experiment.resources.wall_time_budget_seconds =
                parse_double(value, "wall-time budget");
        } else if (argument == "--maximum-nodes") {
            options.experiment.barnes_hut.maximum_nodes =
                parse_size(value, "node budget");
        } else if (argument == "--maximum-traversal-work") {
            options.experiment.barnes_hut.maximum_traversal_work =
                parse_size(value, "traversal-work budget");
        } else if (argument == "--solver-memory-budget-bytes") {
            options.experiment.barnes_hut.memory_budget_bytes =
                parse_size(value, "solver memory budget");
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.output_path.empty() || !options.bodies_set || !options.seed_set) {
        throw std::invalid_argument("--output, --bodies, and --seed are required");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto result = nbody::run_large_n_experiment(options.experiment);
        std::ofstream output(options.output_path, std::ios::binary);
        if (!output) {
            throw std::runtime_error("could not open output: " + options.output_path);
        }
        output << "# study=large_n_laboratory_l1\n"
               << "# stage5_evidence=false\n"
               << "# build_configuration=" << NBODY_BUILD_CONFIGURATION << "\n"
               << "# cxx_compiler_id=" << NBODY_CXX_COMPILER_ID << "\n"
               << "# cxx_compiler_version=" << NBODY_CXX_COMPILER_VERSION << "\n"
               << "# timing_scope=generation_separate_force_complete_step_total\n"
               << nbody::large_n_csv_header() << '\n';
        for (const auto& row : nbody::serialize_large_n_csv_rows(result)) {
            output << row << '\n';
        }
        std::cout << result.status << ": " << result.message << '\n';
        return result.status == "failed" ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "large-N laboratory error: " << error.what() << '\n';
        return 1;
    }
}
