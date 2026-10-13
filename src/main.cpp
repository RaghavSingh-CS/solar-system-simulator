#include "nbody/barnes_hut.hpp"
#include "nbody/diagnostics.hpp"
#include "nbody/scenarios.hpp"

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::size_t steps{5000};
    std::size_t sample_every{50};
    double dt_s{0.0};
    std::string output_path{"orbit.csv"};
    std::string backend{"sequential"};
    std::string solver{"direct"};
    nbody::BarnesHutConfiguration barnes_hut;
    bool barnes_hut_option_seen{false};
};

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program
        << " [--backend sequential] [--solver direct|barnes-hut] [options]\n"
        << "Options: --steps N --dt SECONDS --sample-every N --output PATH\n"
        << "Barnes-Hut: --theta X --leaf-capacity N --max-depth N\n"
        << "            --minimum-cell-size METRES --max-nodes N\n"
        << "            --max-traversal-work N --memory-budget-bytes N\n"
        << "Runs the deterministic analytic circular two-body scenario.\n"
        << "The backend and force solver are separate choices. Stage 1 supports\n"
        << "Barnes-Hut only with the sequential backend; direct remains the default.\n"
        << "If --dt is omitted, the timestep is one orbital period / --steps.\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value for " + argument);
        }
        const std::string value = argv[++i];
        if (argument == "--steps") {
            options.steps = std::stoull(value);
        } else if (argument == "--dt") {
            options.dt_s = std::stod(value);
        } else if (argument == "--sample-every") {
            options.sample_every = std::stoull(value);
        } else if (argument == "--output") {
            options.output_path = value;
        } else if (argument == "--backend") {
            options.backend = value;
        } else if (argument == "--solver") {
            options.solver = value;
        } else if (argument == "--theta") {
            options.barnes_hut_option_seen = true;
            options.barnes_hut.theta = std::stod(value);
        } else if (argument == "--leaf-capacity") {
            options.barnes_hut_option_seen = true;
            options.barnes_hut.leaf_capacity = std::stoull(value);
        } else if (argument == "--max-depth") {
            options.barnes_hut_option_seen = true;
            options.barnes_hut.maximum_depth = std::stoull(value);
        } else if (argument == "--minimum-cell-size") {
            options.barnes_hut_option_seen = true;
            options.barnes_hut.minimum_cell_size_m = std::stod(value);
        } else if (argument == "--max-nodes") {
            options.barnes_hut_option_seen = true;
            options.barnes_hut.maximum_nodes = std::stoull(value);
        } else if (argument == "--max-traversal-work") {
            options.barnes_hut_option_seen = true;
            options.barnes_hut.maximum_traversal_work = std::stoull(value);
        } else if (argument == "--memory-budget-bytes") {
            options.barnes_hut_option_seen = true;
            options.barnes_hut.memory_budget_bytes = std::stoull(value);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.steps == 0 || options.sample_every == 0) {
        throw std::invalid_argument("steps and sample-every must be positive");
    }
    if (options.backend != "sequential") {
        throw std::invalid_argument(
            "headless backend must be sequential; no backend fallback is performed");
    }
    if (options.solver != "direct" && options.solver != "barnes-hut") {
        throw std::invalid_argument("solver must be direct or barnes-hut");
    }
    if (options.solver == "direct" && options.barnes_hut_option_seen) {
        throw std::invalid_argument(
            "Barnes-Hut options require --solver barnes-hut; no option is silently ignored");
    }
    return options;
}

void write_state(std::ostream& output, const nbody::SimulationState& state) {
    for (std::size_t i = 0; i < state.bodies.size(); ++i) {
        const auto& body = state.bodies[i];
        output << state.time_s << ',' << i << ',' << body.name << ',' << body.mass_kg << ','
               << body.position_m.x << ',' << body.position_m.y << ',' << body.position_m.z << ','
               << body.velocity_m_s.x << ',' << body.velocity_m_s.y << ',' << body.velocity_m_s.z
               << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto scenario = nbody::make_analytic_circular_two_body();
        auto state = scenario.initial_state;
        const double dt_s = options.dt_s > 0.0
            ? options.dt_s
            : scenario.orbital_period_s / static_cast<double>(options.steps);

        std::unique_ptr<nbody::ForceBackend> owned_force_backend;
        const nbody::ForceBackend* force_backend = &nbody::sequential_force_backend();
        if (options.solver == "barnes-hut") {
            owned_force_backend =
                std::make_unique<nbody::SequentialBarnesHutForceBackend>(options.barnes_hut);
            force_backend = owned_force_backend.get();
        }

        std::ofstream output(options.output_path);
        if (!output) {
            throw std::runtime_error("could not open output file: " + options.output_path);
        }
        output << std::setprecision(17)
               << "# scenario=analytic_circular_two_body\n"
               << "# provenance=analytic_fixture_not_ephemeris\n"
               << "# frame=barycentric_inertial_cartesian\n"
               << "# units=kg,m,s\n"
               << "# integrator=velocity_verlet\n"
               << "# backend=" << options.backend << '\n'
               << "# force_solver=" << options.solver << '\n'
               << "# gravitational_constant_m3_kg-1_s-2=" << nbody::gravitational_constant << '\n'
               << "# dt_s=" << dt_s << '\n';
        if (options.solver == "barnes-hut") {
            output << "# barnes_hut_theta=" << options.barnes_hut.theta << '\n'
                   << "# barnes_hut_leaf_capacity=" << options.barnes_hut.leaf_capacity << '\n'
                   << "# barnes_hut_maximum_depth=" << options.barnes_hut.maximum_depth << '\n'
                   << "# barnes_hut_minimum_cell_size_m="
                   << options.barnes_hut.minimum_cell_size_m << '\n'
                   << "# barnes_hut_maximum_nodes=" << options.barnes_hut.maximum_nodes << '\n'
                   << "# barnes_hut_maximum_traversal_work="
                   << options.barnes_hut.maximum_traversal_work << '\n'
                   << "# barnes_hut_memory_budget_bytes="
                   << options.barnes_hut.memory_budget_bytes << '\n';
        }
        output << "time_s,body_index,name,mass_kg,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s\n";

        const auto initial_diagnostics = nbody::compute_diagnostics(state);
        write_state(output, state);
        for (std::size_t step = 1; step <= options.steps; ++step) {
            nbody::velocity_verlet_step(state, dt_s, *force_backend);
            if (step % options.sample_every == 0 || step == options.steps) {
                write_state(output, state);
            }
        }

        const auto final_diagnostics = nbody::compute_diagnostics(state);
        const double relative_energy_drift =
            std::abs((final_diagnostics.total_energy_j() - initial_diagnostics.total_energy_j()) /
                     initial_diagnostics.total_energy_j());
        std::cout << std::setprecision(10)
                  << "wrote " << options.output_path << '\n'
                  << "simulated_time_s=" << state.time_s << '\n'
                  << "relative_energy_drift=" << relative_energy_drift << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
