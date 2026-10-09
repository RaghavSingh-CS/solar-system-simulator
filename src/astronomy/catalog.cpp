#include "nbody/astronomy/catalog.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace nbody::astronomy {
namespace {

bool finite_vector(const Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

bool valid_body_class(AstronomicalBodyClass body_class) {
    switch (body_class) {
    case AstronomicalBodyClass::star:
    case AstronomicalBodyClass::planet:
    case AstronomicalBodyClass::dwarf_planet:
    case AstronomicalBodyClass::natural_satellite:
    case AstronomicalBodyClass::asteroid:
    case AstronomicalBodyClass::comet:
    case AstronomicalBodyClass::trans_neptunian_object:
        return true;
    }
    return false;
}

bool valid_stable_id(const std::string& value) {
    if (value.empty() || !std::isalnum(static_cast<unsigned char>(value.front()))) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](char character) {
        const auto byte = static_cast<unsigned char>(character);
        return std::islower(byte) || std::isdigit(byte) || character == '.' ||
            character == '_' || character == ':' || character == '-';
    });
}

bool valid_sha256(const std::string& value) {
    return value.size() == 64 &&
        std::all_of(value.begin(), value.end(), [](char character) {
            return std::isxdigit(static_cast<unsigned char>(character)) != 0;
        });
}

void validate_coordinate_contract(const CoordinateContract& contract) {
    if (contract.epoch_label.empty() || !(contract.epoch_julian_date > 0.0) ||
        !std::isfinite(contract.epoch_julian_date)) {
        throw std::invalid_argument("catalog coordinate epoch must be named, finite, and positive");
    }
    if (contract.time_scale != EphemerisTimeScale::tdb ||
        contract.reference_frame != EphemerisReferenceFrame::icrf ||
        contract.origin != EphemerisOrigin::solar_system_barycenter ||
        contract.state_vector_kind != StateVectorKind::geometric ||
        contract.position_unit != PositionUnit::metre ||
        contract.velocity_unit != VelocityUnit::metre_per_second) {
        throw std::invalid_argument(
            "catalog coordinate contract must use geometric SSB/ICRF/TDB vectors in SI units");
    }
}

bool same_coordinate_contract(
    const CoordinateContract& left,
    const CoordinateContract& right) {
    return left.epoch_label == right.epoch_label &&
        left.epoch_julian_date == right.epoch_julian_date &&
        left.time_scale == right.time_scale &&
        left.reference_frame == right.reference_frame &&
        left.origin == right.origin &&
        left.state_vector_kind == right.state_vector_kind &&
        left.position_unit == right.position_unit &&
        left.velocity_unit == right.velocity_unit;
}

void validate_provenance_record(const ProvenanceRecord& provenance, const char* quantity) {
    if (provenance.source_name.empty() || provenance.source_uri.empty() ||
        provenance.record_id.empty() || provenance.retrieved_utc.empty() ||
        !valid_sha256(provenance.sha256)) {
        throw std::invalid_argument(
            std::string("catalog ") + quantity +
            " provenance requires source, URI, record, retrieval time, and SHA-256");
    }
}

void validate_body_definition(
    const AstronomicalBodyDefinition& body,
    const CoordinateContract& catalog_contract) {
    if (!valid_stable_id(body.stable_id) || body.display_name.empty()) {
        throw std::invalid_argument("catalog body requires a valid stable ID and display name");
    }
    if (!valid_body_class(body.body_class)) {
        throw std::invalid_argument("catalog body has an unsupported body class");
    }
    if (body.parent_stable_id && !valid_stable_id(*body.parent_stable_id)) {
        throw std::invalid_argument("catalog body parent has an invalid stable ID");
    }
    if (body.parent_stable_id && *body.parent_stable_id == body.stable_id) {
        throw std::invalid_argument("catalog body cannot be its own parent");
    }
    if (body.identifiers.empty()) {
        throw std::invalid_argument("catalog body requires at least one external identifier");
    }
    for (const auto& identifier : body.identifiers) {
        if (identifier.authority.empty() || identifier.value.empty()) {
            throw std::invalid_argument(
                "catalog external identifiers require authority and value");
        }
    }
    validate_coordinate_contract(body.coordinate_contract);
    if (!same_coordinate_contract(body.coordinate_contract, catalog_contract)) {
        throw std::invalid_argument(
            "catalog body coordinate contract differs from the shared catalog contract");
    }
    if (!(body.gravitational_parameter_m3_s2 > 0.0) ||
        !std::isfinite(body.gravitational_parameter_m3_s2)) {
        throw std::invalid_argument(
            "catalog body requires a finite positive sourced gravitational parameter");
    }
    if (!(body.mean_radius_m > 0.0) || !std::isfinite(body.mean_radius_m)) {
        throw std::invalid_argument("catalog body requires a finite positive sourced mean radius");
    }
    if (!finite_vector(body.position_m) || !finite_vector(body.velocity_m_s)) {
        throw std::invalid_argument("catalog body state vector must be finite");
    }
    validate_provenance_record(body.provenance.state_vector, "state-vector");
    validate_provenance_record(
        body.provenance.gravitational_parameter, "gravitational-parameter");
    validate_provenance_record(body.provenance.mean_radius, "mean-radius");
    if (body.provenance.model_limitations.empty()) {
        throw std::invalid_argument("catalog body requires explicit model limitations");
    }
}

std::string identifier_key(const CatalogIdentifier& identifier) {
    return identifier.authority + '\n' + identifier.value;
}

}  // namespace

CatalogAdmissionPolicy primary_solar_system_catalog_policy() noexcept {
    return {
        primary_extended_catalog_minimum_body_count,
        true,
        true,
        true,
        true,
    };
}

double mass_from_gravitational_parameter(
    double gravitational_parameter_m3_s2,
    double gravitational_constant_value) {
    if (!(gravitational_parameter_m3_s2 > 0.0) ||
        !std::isfinite(gravitational_parameter_m3_s2) ||
        !(gravitational_constant_value > 0.0) ||
        !std::isfinite(gravitational_constant_value)) {
        throw std::invalid_argument("GM and G must be finite and positive");
    }
    const double mass_kg = gravitational_parameter_m3_s2 / gravitational_constant_value;
    if (!(mass_kg > 0.0) || !std::isfinite(mass_kg)) {
        throw std::invalid_argument("GM-to-mass conversion did not produce a finite positive mass");
    }
    return mass_kg;
}

AstronomicalBodyRecord::AstronomicalBodyRecord(
    AstronomicalBodyDefinition definition,
    double mass_kg)
    : deterministic_order_(definition.deterministic_order),
      stable_id_(std::move(definition.stable_id)),
      display_name_(std::move(definition.display_name)),
      body_class_(definition.body_class),
      parent_stable_id_(std::move(definition.parent_stable_id)),
      identifiers_(std::move(definition.identifiers)),
      coordinate_contract_(std::move(definition.coordinate_contract)),
      gravitational_parameter_m3_s2_(definition.gravitational_parameter_m3_s2),
      mass_kg_(mass_kg),
      mean_radius_m_(definition.mean_radius_m),
      position_m_(definition.position_m),
      velocity_m_s_(definition.velocity_m_s),
      provenance_(std::move(definition.provenance)) {}

std::size_t AstronomicalBodyRecord::deterministic_order() const noexcept {
    return deterministic_order_;
}
const std::string& AstronomicalBodyRecord::stable_id() const noexcept { return stable_id_; }
const std::string& AstronomicalBodyRecord::display_name() const noexcept {
    return display_name_;
}
AstronomicalBodyClass AstronomicalBodyRecord::body_class() const noexcept {
    return body_class_;
}
const std::optional<std::string>& AstronomicalBodyRecord::parent_stable_id() const noexcept {
    return parent_stable_id_;
}
const std::vector<CatalogIdentifier>& AstronomicalBodyRecord::identifiers() const noexcept {
    return identifiers_;
}
const CoordinateContract& AstronomicalBodyRecord::coordinate_contract() const noexcept {
    return coordinate_contract_;
}
double AstronomicalBodyRecord::gravitational_parameter_m3_s2() const noexcept {
    return gravitational_parameter_m3_s2_;
}
double AstronomicalBodyRecord::mass_kg() const noexcept { return mass_kg_; }
double AstronomicalBodyRecord::mean_radius_m() const noexcept { return mean_radius_m_; }
const Vec3& AstronomicalBodyRecord::position_m() const noexcept { return position_m_; }
const Vec3& AstronomicalBodyRecord::velocity_m_s() const noexcept { return velocity_m_s_; }
const BodyProvenance& AstronomicalBodyRecord::provenance() const noexcept {
    return provenance_;
}

AstronomicalCatalog::AstronomicalCatalog(
    CatalogDefinition definition,
    CatalogAdmissionPolicy policy)
    : catalog_id_(std::move(definition.catalog_id)),
      coordinate_contract_(std::move(definition.coordinate_contract)) {
    if (!valid_stable_id(catalog_id_)) {
        throw std::invalid_argument("catalog requires a valid stable catalog ID");
    }
    validate_coordinate_contract(coordinate_contract_);
    if (policy.minimum_body_count == 0) {
        throw std::invalid_argument("catalog admission minimum body count must be positive");
    }
    if (definition.bodies.size() < policy.minimum_body_count) {
        throw std::invalid_argument("catalog does not meet the admission minimum body count");
    }

    std::sort(
        definition.bodies.begin(),
        definition.bodies.end(),
        [](const auto& left, const auto& right) {
            return left.deterministic_order < right.deterministic_order;
        });

    std::unordered_set<std::string> stable_ids;
    std::unordered_set<std::string> display_names;
    std::unordered_set<std::string> external_identifiers;
    records_.reserve(definition.bodies.size());
    for (std::size_t index = 0; index < definition.bodies.size(); ++index) {
        auto& body = definition.bodies[index];
        if (body.deterministic_order != index) {
            throw std::invalid_argument(
                "catalog deterministic order must be unique and contiguous from zero");
        }
        validate_body_definition(body, coordinate_contract_);
        if (!stable_ids.insert(body.stable_id).second) {
            throw std::invalid_argument("catalog contains a duplicate stable body ID");
        }
        if (!display_names.insert(body.display_name).second) {
            throw std::invalid_argument("catalog contains a duplicate display name");
        }
        for (const auto& identifier : body.identifiers) {
            if (!external_identifiers.insert(identifier_key(identifier)).second) {
                throw std::invalid_argument("catalog contains a duplicate external identifier");
            }
        }
        const double mass_kg = mass_from_gravitational_parameter(
            body.gravitational_parameter_m3_s2);
        records_.push_back(AstronomicalBodyRecord(std::move(body), mass_kg));
    }

    std::unordered_map<std::string, std::size_t> indices;
    indices.reserve(records_.size());
    for (std::size_t index = 0; index < records_.size(); ++index) {
        indices.emplace(records_[index].stable_id(), index);
    }

    std::size_t parentless_stars = 0;
    for (std::size_t index = 0; index < records_.size(); ++index) {
        const auto& body = records_[index];
        const bool star = body.body_class() == AstronomicalBodyClass::star;
        if (!body.parent_stable_id()) {
            if (star) {
                ++parentless_stars;
            } else if (policy.require_parent_for_non_star) {
                throw std::invalid_argument("primary catalog non-star body requires a parent");
            }
            continue;
        }
        if (star) {
            throw std::invalid_argument("catalog star cannot have a parent body");
        }
        const auto parent = indices.find(*body.parent_stable_id());
        if (parent == indices.end()) {
            throw std::invalid_argument("catalog body refers to an unknown parent");
        }
        if (policy.require_parent_before_child && parent->second >= index) {
            throw std::invalid_argument(
                "catalog parent must precede its child in deterministic order");
        }
        if (policy.enforce_solar_parent_classes) {
            const auto parent_class = records_[parent->second].body_class();
            if (body.body_class() == AstronomicalBodyClass::natural_satellite) {
                const bool valid_satellite_parent =
                    parent_class == AstronomicalBodyClass::planet ||
                    parent_class == AstronomicalBodyClass::dwarf_planet ||
                    parent_class == AstronomicalBodyClass::asteroid ||
                    parent_class == AstronomicalBodyClass::trans_neptunian_object;
                if (!valid_satellite_parent) {
                    throw std::invalid_argument(
                        "natural satellite requires an admitted planetary or minor-body parent");
                }
            } else if (parent_class != AstronomicalBodyClass::star) {
                throw std::invalid_argument(
                    "primary catalog non-satellite bodies must use the star as parent");
            }
        }
    }
    if (policy.require_single_parentless_star && parentless_stars != 1) {
        throw std::invalid_argument("primary catalog requires exactly one parentless star");
    }
}

AstronomicalCatalog AstronomicalCatalog::admit_primary_solar_system(
    CatalogDefinition definition) {
    return AstronomicalCatalog(
        std::move(definition), primary_solar_system_catalog_policy());
}

const std::string& AstronomicalCatalog::catalog_id() const noexcept { return catalog_id_; }
const CoordinateContract& AstronomicalCatalog::coordinate_contract() const noexcept {
    return coordinate_contract_;
}
const std::vector<AstronomicalBodyRecord>& AstronomicalCatalog::records() const noexcept {
    return records_;
}
const AstronomicalBodyRecord* AstronomicalCatalog::find(
    const std::string& stable_id) const noexcept {
    const auto match = std::find_if(records_.begin(), records_.end(), [&](const auto& body) {
        return body.stable_id() == stable_id;
    });
    return match == records_.end() ? nullptr : &*match;
}

}  // namespace nbody::astronomy
