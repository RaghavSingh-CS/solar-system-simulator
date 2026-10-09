#pragma once

#include "nbody/simulation.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace nbody::astronomy {

inline constexpr std::size_t primary_extended_catalog_minimum_body_count = 60;

enum class AstronomicalBodyClass {
    star,
    planet,
    dwarf_planet,
    natural_satellite,
    asteroid,
    comet,
    trans_neptunian_object,
};

enum class EphemerisTimeScale {
    tdb,
};

enum class EphemerisReferenceFrame {
    icrf,
};

enum class EphemerisOrigin {
    solar_system_barycenter,
};

enum class StateVectorKind {
    geometric,
};

enum class PositionUnit {
    metre,
};

enum class VelocityUnit {
    metre_per_second,
};

struct CoordinateContract {
    std::string epoch_label;
    double epoch_julian_date{0.0};
    EphemerisTimeScale time_scale{EphemerisTimeScale::tdb};
    EphemerisReferenceFrame reference_frame{EphemerisReferenceFrame::icrf};
    EphemerisOrigin origin{EphemerisOrigin::solar_system_barycenter};
    StateVectorKind state_vector_kind{StateVectorKind::geometric};
    PositionUnit position_unit{PositionUnit::metre};
    VelocityUnit velocity_unit{VelocityUnit::metre_per_second};
};

struct CatalogIdentifier {
    std::string authority;
    std::string value;
};

struct ProvenanceRecord {
    std::string source_name;
    std::string source_uri;
    std::string record_id;
    std::string retrieved_utc;
    std::string sha256;
};

struct BodyProvenance {
    ProvenanceRecord state_vector;
    ProvenanceRecord gravitational_parameter;
    ProvenanceRecord mean_radius;
    std::string model_limitations;
};

// Mutable only as an ingestion value. AstronomicalCatalog validates, orders,
// converts, and stores it as an immutable AstronomicalBodyRecord.
struct AstronomicalBodyDefinition {
    std::size_t deterministic_order{0};
    std::string stable_id;
    std::string display_name;
    AstronomicalBodyClass body_class{AstronomicalBodyClass::asteroid};
    std::optional<std::string> parent_stable_id;
    std::vector<CatalogIdentifier> identifiers;
    CoordinateContract coordinate_contract;
    double gravitational_parameter_m3_s2{0.0};
    double mean_radius_m{0.0};
    Vec3 position_m;
    Vec3 velocity_m_s;
    BodyProvenance provenance;
};

struct CatalogDefinition {
    std::string catalog_id;
    CoordinateContract coordinate_contract;
    std::vector<AstronomicalBodyDefinition> bodies;
};

struct CatalogAdmissionPolicy {
    std::size_t minimum_body_count{1};
    bool require_single_parentless_star{false};
    bool require_parent_for_non_star{false};
    bool require_parent_before_child{true};
    bool enforce_solar_parent_classes{false};
};

CatalogAdmissionPolicy primary_solar_system_catalog_policy() noexcept;

double mass_from_gravitational_parameter(
    double gravitational_parameter_m3_s2,
    double gravitational_constant_value = gravitational_constant);

class AstronomicalBodyRecord {
public:
    std::size_t deterministic_order() const noexcept;
    const std::string& stable_id() const noexcept;
    const std::string& display_name() const noexcept;
    AstronomicalBodyClass body_class() const noexcept;
    const std::optional<std::string>& parent_stable_id() const noexcept;
    const std::vector<CatalogIdentifier>& identifiers() const noexcept;
    const CoordinateContract& coordinate_contract() const noexcept;
    double gravitational_parameter_m3_s2() const noexcept;
    double mass_kg() const noexcept;
    double mean_radius_m() const noexcept;
    const Vec3& position_m() const noexcept;
    const Vec3& velocity_m_s() const noexcept;
    const BodyProvenance& provenance() const noexcept;

private:
    friend class AstronomicalCatalog;

    AstronomicalBodyRecord(AstronomicalBodyDefinition definition, double mass_kg);

    std::size_t deterministic_order_{0};
    std::string stable_id_;
    std::string display_name_;
    AstronomicalBodyClass body_class_{AstronomicalBodyClass::asteroid};
    std::optional<std::string> parent_stable_id_;
    std::vector<CatalogIdentifier> identifiers_;
    CoordinateContract coordinate_contract_;
    double gravitational_parameter_m3_s2_{0.0};
    double mass_kg_{0.0};
    double mean_radius_m_{0.0};
    Vec3 position_m_;
    Vec3 velocity_m_s_;
    BodyProvenance provenance_;
};

class AstronomicalCatalog {
public:
    explicit AstronomicalCatalog(
        CatalogDefinition definition,
        CatalogAdmissionPolicy policy);

    static AstronomicalCatalog admit_primary_solar_system(CatalogDefinition definition);

    const std::string& catalog_id() const noexcept;
    const CoordinateContract& coordinate_contract() const noexcept;
    const std::vector<AstronomicalBodyRecord>& records() const noexcept;
    const AstronomicalBodyRecord* find(const std::string& stable_id) const noexcept;

private:
    std::string catalog_id_;
    CoordinateContract coordinate_contract_;
    std::vector<AstronomicalBodyRecord> records_;
};

}  // namespace nbody::astronomy
