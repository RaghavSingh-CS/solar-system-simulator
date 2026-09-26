#include "nbody/diagnostics.hpp"
#include "nbody/scenarios.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#ifndef NBODY_BUILD_CONFIGURATION
#define NBODY_BUILD_CONFIGURATION "unknown"
#endif

namespace {

constexpr double energy_drift_bound = 5.0e-6;
constexpr double angular_momentum_drift_bound = 5.0e-10;
constexpr std::uint64_t seeded_scenario_seed = 0x5eedULL;

struct Options {
    std::string scenario;
    double timestep_multiplier{1.0};
    double duration_multiplier{1.0};
};

struct ScenarioDefinition {
    std::string name;
    std::string seed;
    nbody::SimulationState initial_state;
    double base_timestep_s{0.0};
    std::size_t baseline_steps{0};
};

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program
        << " --scenario analytic_two_body|seeded_32_body"
           " --dt-multiplier X --duration-multiplier X\n";
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
        if (argument == "--scenario") {
            options.scenario = value;
        } else if (argument == "--dt-multiplier") {
            options.timestep_multiplier = std::stod(value);
        } else if (argument == "--duration-multiplier") {
            options.duration_multiplier = std::stod(value);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.scenario.empty() || !(options.timestep_multiplier > 0.0) ||
        !std::isfinite(options.timestep_multiplier) ||
        !(options.duration_multiplier > 0.0) ||
        !std::isfinite(options.duration_multiplier)) {
        throw std::invalid_argument(
            "scenario and finite positive timestep/duration multipliers are required");
    }
    return options;
}

ScenarioDefinition make_scenario(const std::string& name) {
    if (name == "analytic_two_body") {
        const auto scenario = nbody::make_analytic_circular_two_body();
        constexpr std::size_t steps_per_orbit = 4000;
        return {
            name,
            "analytic",
            scenario.initial_state,
            scenario.orbital_period_s / static_cast<double>(steps_per_orbit),
            steps_per_orbit,
        };
    }
    if (name == "seeded_32_body") {
        // 256 steps is close to the existing 250-step validation horizon and is divisible
        // by every timestep multiplier in the study's 0.25x--4x sweep.
        constexpr std::size_t baseline_steps = 256;
        return {
            name,
            std::to_string(seeded_scenario_seed),
            nbody::make_seeded_synthetic_system(32, seeded_scenario_seed),
            10.0,
            baseline_steps,
        };
    }
    throw std::invalid_argument("unsupported scenario: " + name);
}

double relative_energy_drift(double current, double initial) {
    return std::abs(current - initial) /
        std::max({std::abs(current), std::abs(initial), 1.0});
}

double relative_angular_momentum_drift(
    const nbody::Vec3& current,
    const nbody::Vec3& initial) {
    return nbody::norm(current - initial) / std::max(nbody::norm(initial), 1.0);
}

double state_checksum(const nbody::SimulationState& state) {
    double checksum = state.time_s;
    for (std::size_t i = 0; i < state.bodies.size(); ++i) {
        const double weight = static_cast<double>(i + 1);
        checksum += weight *
            (state.bodies[i].position_m.x + 3.0 * state.bodies[i].position_m.y +
             5.0 * state.bodies[i].position_m.z + state.bodies[i].velocity_m_s.x +
             7.0 * state.bodies[i].velocity_m_s.y +
             11.0 * state.bodies[i].velocity_m_s.z);
    }
    return checksum;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const ScenarioDefinition scenario = make_scenario(options.scenario);
        const double timestep_s = scenario.base_timestep_s * options.timestep_multiplier;
        const double baseline_duration_s =
            scenario.base_timestep_s * static_cast<double>(scenario.baseline_steps);
        const double requested_steps =
            static_cast<double>(scenario.baseline_steps) * options.duration_multiplier /
            options.timestep_multiplier;
        const double rounded_steps = std::round(requested_steps);
        if (std::abs(requested_steps - rounded_steps) >
            1.0e-12 * std::max(requested_steps, 1.0) ||
            rounded_steps < 1.0 ||
            rounded_steps > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
            throw std::invalid_argument(
                "multipliers do not produce a positive integral step count");
        }
        const auto steps = static_cast<std::size_t>(rounded_steps);

        auto state = scenario.initial_state;
        const auto initial = nbody::compute_diagnostics(state);
        const double initial_energy = initial.total_energy_j();
        double maximum_energy_drift = 0.0;
        double maximum_angular_drift = 0.0;

        const auto started = std::chrono::steady_clock::now();
        for (std::size_t step = 0; step < steps; ++step) {
            nbody::velocity_verlet_step(
                state, timestep_s, nbody::sequential_force_backend());
            const auto current = nbody::compute_diagnostics(state);
            const double energy_drift =
                relative_energy_drift(current.total_energy_j(), initial_energy);
            const double angular_drift = relative_angular_momentum_drift(
                current.angular_momentum_kg_m2_s,
                initial.angular_momentum_kg_m2_s);
            if (!std::isfinite(energy_drift) || !std::isfinite(angular_drift)) {
                throw std::runtime_error("non-finite conservation diagnostic");
            }
            maximum_energy_drift = std::max(maximum_energy_drift, energy_drift);
            maximum_angular_drift = std::max(maximum_angular_drift, angular_drift);
        }
        const double wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();

        const auto final = nbody::compute_diagnostics(state);
        const double final_energy_drift =
            relative_energy_drift(final.total_energy_j(), initial_energy);
        const double final_angular_drift = relative_angular_momentum_drift(
            final.angular_momentum_kg_m2_s,
            initial.angular_momentum_kg_m2_s);
        const bool energy_pass = maximum_energy_drift <= energy_drift_bound;
        const bool angular_pass = maximum_angular_drift <= angular_momentum_drift_bound;

        std::cout
            << "scenario,seed,backend,integrator,precision,base_dt_s,dt_s,dt_multiplier,"
               "baseline_duration_s,duration_multiplier,steps,simulated_duration_s,"
               "requested_duration_s,time_accumulation_error_s,diagnostic_samples,"
               "initial_energy_j,final_energy_j,"
               "maximum_relative_energy_drift,final_relative_energy_drift,"
               "initial_angular_x,initial_angular_y,initial_angular_z,"
               "initial_angular_magnitude,final_angular_x,final_angular_y,final_angular_z,"
               "final_angular_magnitude,maximum_relative_angular_momentum_drift,"
               "final_relative_angular_momentum_drift,energy_drift_bound,"
               "angular_momentum_drift_bound,energy_status,angular_momentum_status,"
               "overall_status,experiment_wall_ms,final_state_checksum,"
               "build_configuration\n";
        std::cout << std::setprecision(17)
                  << scenario.name << ',' << scenario.seed
                  << ",sequential,velocity_verlet,binary64,"
                  << scenario.base_timestep_s << ',' << timestep_s << ','
                  << options.timestep_multiplier << ',' << baseline_duration_s << ','
                  << options.duration_multiplier << ',' << steps << ',' << state.time_s << ','
                  << (static_cast<double>(steps) * timestep_s) << ','
                  << (state.time_s - static_cast<double>(steps) * timestep_s) << ','
                  << (steps + 1) << ',' << initial_energy << ',' << final.total_energy_j() << ','
                  << maximum_energy_drift << ',' << final_energy_drift << ','
                  << initial.angular_momentum_kg_m2_s.x << ','
                  << initial.angular_momentum_kg_m2_s.y << ','
                  << initial.angular_momentum_kg_m2_s.z << ','
                  << nbody::norm(initial.angular_momentum_kg_m2_s) << ','
                  << final.angular_momentum_kg_m2_s.x << ','
                  << final.angular_momentum_kg_m2_s.y << ','
                  << final.angular_momentum_kg_m2_s.z << ','
                  << nbody::norm(final.angular_momentum_kg_m2_s) << ','
                  << maximum_angular_drift << ',' << final_angular_drift << ','
                  << energy_drift_bound << ',' << angular_momentum_drift_bound << ','
                  << (energy_pass ? "pass" : "fail") << ','
                  << (angular_pass ? "pass" : "fail") << ','
                  << (energy_pass && angular_pass ? "pass" : "fail") << ','
                  << wall_ms << ',' << state_checksum(state) << ','
                  << NBODY_BUILD_CONFIGURATION << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
