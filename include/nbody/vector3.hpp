#pragma once

#include <cmath>

namespace nbody {

struct Vec3 {
    double x{0.0};
    double y{0.0};
    double z{0.0};

    Vec3& operator+=(const Vec3& rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }

    Vec3& operator-=(const Vec3& rhs) {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }

    Vec3& operator*=(double scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }
};

inline Vec3 operator+(Vec3 lhs, const Vec3& rhs) { return lhs += rhs; }
inline Vec3 operator-(Vec3 lhs, const Vec3& rhs) { return lhs -= rhs; }
inline Vec3 operator-(const Vec3& value) { return {-value.x, -value.y, -value.z}; }
inline Vec3 operator*(Vec3 value, double scalar) { return value *= scalar; }
inline Vec3 operator*(double scalar, Vec3 value) { return value *= scalar; }
inline Vec3 operator/(Vec3 value, double scalar) { return value *= (1.0 / scalar); }

inline double dot(const Vec3& lhs, const Vec3& rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

inline Vec3 cross(const Vec3& lhs, const Vec3& rhs) {
    return {
        lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.x * rhs.y - lhs.y * rhs.x,
    };
}

inline double norm_squared(const Vec3& value) { return dot(value, value); }
inline double norm(const Vec3& value) { return std::sqrt(norm_squared(value)); }

}  // namespace nbody
