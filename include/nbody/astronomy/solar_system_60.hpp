#pragma once

#include "nbody/astronomy/catalog.hpp"

namespace nbody::astronomy {

inline constexpr const char* solar_system_60_dataset_id = "jpl-ssd-j2000-60-v1";
inline constexpr const char* solar_system_60_dataset_version = "1.0.0";
inline constexpr const char* solar_system_60_roster_sha256 =
    "2df8e58be8924f7a49b44ca761db8ffb0e0a1204689844044624c907d65273cd";
inline constexpr const char* solar_system_60_snapshot_manifest_sha256 =
    "82b271f3fe7470172693c9ab4810a664153ab59f94f293498000b19504749229";

CatalogDefinition make_solar_system_60_definition();
AstronomicalCatalog make_solar_system_60_catalog();

}  // namespace nbody::astronomy
