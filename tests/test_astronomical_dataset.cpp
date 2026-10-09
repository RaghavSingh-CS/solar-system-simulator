#include "nbody/astronomy/solar_system_60.hpp"

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

double distance(const nbody::Vec3& left, const nbody::Vec3& right) {
    const double dx = left.x - right.x;
    const double dy = left.y - right.y;
    const double dz = left.z - right.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

bool finite_vector(const nbody::Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

const nbody::astronomy::AstronomicalBodyRecord& body(
    const nbody::astronomy::AstronomicalCatalog& catalog,
    const std::string& stable_id) {
    const auto* found = catalog.find(stable_id);
    require(found != nullptr, "missing catalog body " + stable_id);
    return *found;
}

void require_relative_range(
    const nbody::astronomy::AstronomicalCatalog& catalog,
    const std::string& child_id,
    const std::string& parent_id,
    double minimum_distance_m,
    double maximum_distance_m,
    double minimum_speed_m_s,
    double maximum_speed_m_s) {
    const auto& child = body(catalog, child_id);
    const auto& parent = body(catalog, parent_id);
    const double separation = distance(child.position_m(), parent.position_m());
    const double speed = distance(child.velocity_m_s(), parent.velocity_m_s());
    require(separation >= minimum_distance_m && separation <= maximum_distance_m,
            child_id + " parent-relative distance is outside the scientific sanity range");
    require(speed >= minimum_speed_m_s && speed <= maximum_speed_m_s,
            child_id + " parent-relative speed is outside the scientific sanity range");
}

void test_identity_order_and_contract() {
    const auto catalog = nbody::astronomy::make_solar_system_60_catalog();
    constexpr std::array<const char*, 60> expected_ids{{
        "sol:sun", "sol:mercury", "sol:venus", "sol:earth", "sol:mars",
        "sol:jupiter", "sol:saturn", "sol:uranus", "sol:neptune", "sol:pluto",
        "sol:ceres", "sol:moon", "sol:phobos", "sol:deimos", "sol:io",
        "sol:europa", "sol:ganymede", "sol:callisto", "sol:amalthea", "sol:thebe",
        "sol:himalia", "sol:metis", "sol:titan", "sol:rhea", "sol:iapetus",
        "sol:dione", "sol:tethys", "sol:enceladus", "sol:mimas", "sol:hyperion",
        "sol:phoebe", "sol:janus", "sol:titania", "sol:oberon", "sol:ariel",
        "sol:umbriel", "sol:miranda", "sol:triton", "sol:proteus", "sol:charon",
        "sol:vesta", "sol:pallas", "sol:hygiea", "sol:juno", "sol:psyche",
        "sol:eros", "sol:bennu", "sol:ryugu", "sol:davida", "sol:interamnia",
        "sol:67p", "sol:epimetheus", "sol:helene", "sol:atlas", "sol:prometheus",
        "sol:pandora", "sol:nix", "sol:hydra", "sol:thalassa", "sol:eunomia",
    }};
    require(catalog.catalog_id() == nbody::astronomy::solar_system_60_dataset_id,
            "dataset ID differs from the frozen identity");
    require(std::string(nbody::astronomy::solar_system_60_dataset_version) == "1.0.0",
            "dataset version differs from the frozen identity");
    require(catalog.records().size() == expected_ids.size(),
            "catalog does not contain exactly 60 records");
    const auto& contract = catalog.coordinate_contract();
    require(contract.epoch_label == "J2000.0 (JD 2451545.0 TDB)" &&
                contract.epoch_julian_date == 2451545.0 &&
                contract.time_scale == nbody::astronomy::EphemerisTimeScale::tdb &&
                contract.reference_frame == nbody::astronomy::EphemerisReferenceFrame::icrf &&
                contract.origin == nbody::astronomy::EphemerisOrigin::solar_system_barycenter &&
                contract.state_vector_kind == nbody::astronomy::StateVectorKind::geometric,
            "catalog coordinate contract differs from the frozen J2000 contract");

    std::unordered_set<std::string> unique_ids;
    for (std::size_t index = 0; index < catalog.records().size(); ++index) {
        const auto& record = catalog.records()[index];
        require(record.deterministic_order() == index &&
                    record.stable_id() == expected_ids[index],
                "catalog deterministic roster order differs");
        require(unique_ids.insert(record.stable_id()).second, "duplicate stable ID");
        require(record.gravitational_parameter_m3_s2() > 0.0 &&
                    std::isfinite(record.gravitational_parameter_m3_s2()) &&
                    record.mean_radius_m() > 0.0 && std::isfinite(record.mean_radius_m()),
                "catalog contains an invalid physical parameter");
        require(finite_vector(record.position_m()) && finite_vector(record.velocity_m_s()),
                "catalog contains a non-finite state");
        const double relative_gm_error = std::abs(
            record.mass_kg() * nbody::gravitational_constant -
            record.gravitational_parameter_m3_s2()) /
            record.gravitational_parameter_m3_s2();
        require(relative_gm_error <= 2.0e-15,
                "catalog mass was not reconstructed from GM using the project constant");
        require(!record.provenance().model_limitations.empty(),
                "catalog record lacks scientific limitations");
    }
}

void test_parent_relationships_and_planet_only_gm() {
    const auto catalog = nbody::astronomy::make_solar_system_60_catalog();
    require(*body(catalog, "sol:moon").parent_stable_id() == "sol:earth",
            "Moon parent differs");
    require(*body(catalog, "sol:io").parent_stable_id() == "sol:jupiter",
            "Io parent differs");
    require(*body(catalog, "sol:titan").parent_stable_id() == "sol:saturn",
            "Titan parent differs");
    require(*body(catalog, "sol:charon").parent_stable_id() == "sol:pluto",
            "Charon parent differs");
    require(*body(catalog, "sol:nix").parent_stable_id() == "sol:pluto",
            "Nix parent differs");
    const double jupiter_gm = body(catalog, "sol:jupiter").gravitational_parameter_m3_s2();
    require(std::abs(jupiter_gm - 1.266865319003704e17) / jupiter_gm <= 1.0e-15,
            "Jupiter does not use the body-only GM from the JPL kernel");
}

void test_representative_relative_states() {
    const auto catalog = nbody::astronomy::make_solar_system_60_catalog();
    require_relative_range(catalog, "sol:earth", "sol:sun",
                           1.45e11, 1.53e11, 2.8e4, 3.2e4);
    require_relative_range(catalog, "sol:moon", "sol:earth",
                           3.4e8, 4.2e8, 8.0e2, 1.2e3);
    require_relative_range(catalog, "sol:io", "sol:jupiter",
                           4.0e8, 4.5e8, 1.6e4, 1.9e4);
    require_relative_range(catalog, "sol:titan", "sol:saturn",
                           1.1e9, 1.4e9, 4.5e3, 6.5e3);
    require_relative_range(catalog, "sol:charon", "sol:pluto",
                           1.8e7, 2.1e7, 1.8e2, 2.7e2);
}

}  // namespace

int main() {
    try {
        test_identity_order_and_contract();
        test_parent_relationships_and_planet_only_gm();
        test_representative_relative_states();
        std::cout << "all astronomical dataset tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "astronomical dataset test failure: " << error.what() << '\n';
        return 1;
    }
}
