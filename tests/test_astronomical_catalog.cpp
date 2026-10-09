#include "nbody/astronomy/catalog.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using nbody::astronomy::AstronomicalBodyClass;
using nbody::astronomy::AstronomicalBodyDefinition;
using nbody::astronomy::AstronomicalCatalog;
using nbody::astronomy::CatalogDefinition;
using nbody::astronomy::CatalogIdentifier;
using nbody::astronomy::CoordinateContract;
using nbody::astronomy::ProvenanceRecord;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Action>
void require_rejected(Action action, const std::string& message) {
    bool rejected = false;
    try {
        action();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, message);
}

CoordinateContract fixture_contract() {
    return {
        "fixture epoch (not astronomical data)",
        2451545.0,
        nbody::astronomy::EphemerisTimeScale::tdb,
        nbody::astronomy::EphemerisReferenceFrame::icrf,
        nbody::astronomy::EphemerisOrigin::solar_system_barycenter,
        nbody::astronomy::StateVectorKind::geometric,
        nbody::astronomy::PositionUnit::metre,
        nbody::astronomy::VelocityUnit::metre_per_second,
    };
}

ProvenanceRecord fixture_provenance(const std::string& record_id) {
    return {
        "unit-test fixture",
        "https://example.invalid/catalog-fixture",
        record_id,
        "2000-01-01T00:00:00Z",
        std::string(64, 'a'),
    };
}

AstronomicalBodyDefinition fixture_body(
    std::size_t order,
    std::string stable_id,
    AstronomicalBodyClass body_class,
    std::optional<std::string> parent,
    CoordinateContract contract = fixture_contract()) {
    const std::string display_name = "Fixture " + stable_id;
    const auto state_source = fixture_provenance("state-" + stable_id);
    const auto gm_source = fixture_provenance("gm-" + stable_id);
    const auto radius_source = fixture_provenance("radius-" + stable_id);
    return {
        order,
        stable_id,
        display_name,
        body_class,
        std::move(parent),
        {{"fixture-authority", stable_id}},
        std::move(contract),
        1.0e10 + static_cast<double>(order),
        1.0e5 + static_cast<double>(order),
        {static_cast<double>(order), 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {state_source, gm_source, radius_source, "Synthetic unit-test fixture only."},
    };
}

CatalogDefinition two_body_definition() {
    const auto contract = fixture_contract();
    CatalogDefinition definition;
    definition.catalog_id = "fixture:catalog";
    definition.coordinate_contract = contract;
    // Deliberately reverse ingestion order. Deterministic order is explicit.
    definition.bodies.push_back(fixture_body(
        1, "fixture:planet", AstronomicalBodyClass::planet, "fixture:star", contract));
    definition.bodies.push_back(fixture_body(
        0, "fixture:star", AstronomicalBodyClass::star, std::nullopt, contract));
    return definition;
}

void test_mass_conversion() {
    constexpr double gm = 3.98600435507e14;
    const double mass = nbody::astronomy::mass_from_gravitational_parameter(gm);
    require(std::isfinite(mass) && mass > 0.0, "GM conversion did not produce a mass");
    require(std::abs(mass * nbody::gravitational_constant - gm) / gm <= 1.0e-15,
            "GM conversion does not use the project gravitational constant");
    require_rejected(
        [] { (void)nbody::astronomy::mass_from_gravitational_parameter(0.0); },
        "zero GM was admitted");
    require_rejected(
        [] {
            (void)nbody::astronomy::mass_from_gravitational_parameter(
                1.0, std::numeric_limits<double>::quiet_NaN());
        },
        "non-finite G was admitted");
}

void test_immutable_records_and_deterministic_order() {
    const AstronomicalCatalog catalog(two_body_definition(), {});
    static_assert(std::is_same_v<
        decltype(std::declval<const AstronomicalCatalog&>().records()),
        const std::vector<nbody::astronomy::AstronomicalBodyRecord>&>);
    require(catalog.records().size() == 2, "catalog body count differs");
    require(catalog.records()[0].stable_id() == "fixture:star" &&
                catalog.records()[1].stable_id() == "fixture:planet",
            "catalog did not apply deterministic order");
    require(catalog.find("fixture:planet") == &catalog.records()[1] &&
                catalog.find("fixture:missing") == nullptr,
            "stable-ID lookup is incorrect");
    const auto& planet = catalog.records()[1];
    require(planet.parent_stable_id() && *planet.parent_stable_id() == "fixture:star",
            "parent relationship was not retained");
    require(std::abs(
                planet.mass_kg() * nbody::gravitational_constant -
                planet.gravitational_parameter_m3_s2()) /
                planet.gravitational_parameter_m3_s2() <= 1.0e-15,
            "admitted record mass was not derived from GM");
}

void test_coordinate_and_provenance_admission() {
    auto mismatched = two_body_definition();
    mismatched.bodies[0].coordinate_contract.epoch_julian_date += 1.0;
    require_rejected(
        [&] { (void)AstronomicalCatalog(mismatched, {}); },
        "mixed catalog epochs were admitted");

    auto apparent = two_body_definition();
    apparent.bodies[0].coordinate_contract.state_vector_kind =
        static_cast<nbody::astronomy::StateVectorKind>(99);
    require_rejected(
        [&] { (void)AstronomicalCatalog(apparent, {}); },
        "unsupported state-vector kind was admitted");

    auto bad_hash = two_body_definition();
    bad_hash.bodies[0].provenance.gravitational_parameter.sha256 = "not-a-sha256";
    require_rejected(
        [&] { (void)AstronomicalCatalog(bad_hash, {}); },
        "incomplete GM provenance was admitted");

    auto missing_limitations = two_body_definition();
    missing_limitations.bodies[0].provenance.model_limitations.clear();
    require_rejected(
        [&] { (void)AstronomicalCatalog(missing_limitations, {}); },
        "body without model limitations was admitted");

    auto unsupported_class = two_body_definition();
    unsupported_class.bodies[0].body_class = static_cast<AstronomicalBodyClass>(99);
    require_rejected(
        [&] { (void)AstronomicalCatalog(unsupported_class, {}); },
        "unsupported body class was admitted");
}

void test_identity_and_parent_admission() {
    auto duplicate_id = two_body_definition();
    duplicate_id.bodies[0].stable_id = duplicate_id.bodies[1].stable_id;
    require_rejected(
        [&] { (void)AstronomicalCatalog(duplicate_id, {}); },
        "duplicate stable ID was admitted");

    auto duplicate_external = two_body_definition();
    duplicate_external.bodies[0].identifiers = duplicate_external.bodies[1].identifiers;
    require_rejected(
        [&] { (void)AstronomicalCatalog(duplicate_external, {}); },
        "duplicate external identifier was admitted");

    auto unknown_parent = two_body_definition();
    unknown_parent.bodies[0].parent_stable_id = "fixture:missing";
    require_rejected(
        [&] { (void)AstronomicalCatalog(unknown_parent, {}); },
        "unknown parent was admitted");

    auto child_before_parent = two_body_definition();
    child_before_parent.bodies[0].deterministic_order = 0;
    child_before_parent.bodies[1].deterministic_order = 1;
    require_rejected(
        [&] { (void)AstronomicalCatalog(child_before_parent, {}); },
        "child preceding its parent was admitted");
}

void test_physical_parameter_admission() {
    auto zero_gm = two_body_definition();
    zero_gm.bodies[0].gravitational_parameter_m3_s2 = 0.0;
    require_rejected(
        [&] { (void)AstronomicalCatalog(zero_gm, {}); },
        "zero-GM body was admitted as gravitating");

    auto zero_radius = two_body_definition();
    zero_radius.bodies[0].mean_radius_m = 0.0;
    require_rejected(
        [&] { (void)AstronomicalCatalog(zero_radius, {}); },
        "body without a sourced positive radius was admitted");

    auto non_finite_state = two_body_definition();
    non_finite_state.bodies[0].position_m.x =
        std::numeric_limits<double>::infinity();
    require_rejected(
        [&] { (void)AstronomicalCatalog(non_finite_state, {}); },
        "non-finite state vector was admitted");
}

void test_primary_catalog_policy() {
    const auto contract = fixture_contract();
    CatalogDefinition definition;
    definition.catalog_id = "fixture:primary";
    definition.coordinate_contract = contract;
    definition.bodies.push_back(fixture_body(
        0, "fixture:star", AstronomicalBodyClass::star, std::nullopt, contract));
    for (std::size_t index = 1;
         index < nbody::astronomy::primary_extended_catalog_minimum_body_count;
         ++index) {
        definition.bodies.push_back(fixture_body(
            index,
            "fixture:asteroid-" + std::to_string(index),
            AstronomicalBodyClass::asteroid,
            "fixture:star",
            contract));
    }
    const AstronomicalCatalog admitted =
        AstronomicalCatalog::admit_primary_solar_system(definition);
    require(admitted.records().size() == 60,
            "primary policy did not admit the 60-record fixture");

    definition.bodies.pop_back();
    require_rejected(
        [&] {
            (void)AstronomicalCatalog::admit_primary_solar_system(definition);
        },
        "primary policy admitted fewer than 60 bodies");
}

}  // namespace

int main() {
    try {
        test_mass_conversion();
        test_immutable_records_and_deterministic_order();
        test_coordinate_and_provenance_admission();
        test_identity_and_parent_admission();
        test_physical_parameter_admission();
        test_primary_catalog_policy();
        std::cout << "all astronomical catalog foundation tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "astronomical catalog foundation test failure: " << error.what() << '\n';
        return 1;
    }
}
