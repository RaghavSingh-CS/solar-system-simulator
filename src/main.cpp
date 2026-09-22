#include "nbody/diagnostics.hpp"
#include "nbody/scenarios.hpp"

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::size_t steps{5000};
    std::size_t sample_every{50};
    double dt_s{0.0};
    std::string output_path{"orbit.csv"};
};

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [--steps N] [--dt SECONDS] [--sample-every N] [--output PATH]\n"
        << "Runs the deterministic analytic circular two-body scenario.\n"
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
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.steps == 0 || options.sample_every == 0) {
        throw std::invalid_argument("steps and sample-every must be positive");
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
               << "# gravitational_constant_m3_kg-1_s-2=" << nbody::gravitational_constant << '\n'
               << "# dt_s=" << dt_s << '\n'
               << "time_s,body_index,name,mass_kg,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s\n";

        const auto initial_diagnostics = nbody::compute_diagnostics(state);
        write_state(output, state);
        for (std::size_t step = 1; step <= options.steps; ++step) {
            nbody::velocity_verlet_step(state, dt_s);
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
