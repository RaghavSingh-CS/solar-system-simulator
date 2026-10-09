#include "nbody/astronomy/solar_system_60.hpp"

#include <utility>

namespace nbody::astronomy {
namespace {

#include "generated/solar_system_60_v1.generated.inc"

}  // namespace

CatalogDefinition make_solar_system_60_definition() {
    return generated_solar_system_60_v1_definition();
}

AstronomicalCatalog make_solar_system_60_catalog() {
    return AstronomicalCatalog::admit_primary_solar_system(
        make_solar_system_60_definition());
}

}  // namespace nbody::astronomy
