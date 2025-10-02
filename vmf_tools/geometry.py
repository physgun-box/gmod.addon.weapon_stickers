"""Geometry helpers for VMF parsing and rendering."""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, sqrt
from typing import Iterable, List, Tuple

EPSILON = 1e-4


@dataclass(frozen=True)
class Vector3:
    """Simple 3D vector with basic operations."""

    x: float
    y: float
    z: float

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    __rmul__ = __mul__

    def dot(self, other: "Vector3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self) -> "Vector3":
        length = self.length()
        if length < EPSILON:
            return Vector3(0.0, 0.0, 0.0)
        return Vector3(self.x / length, self.y / length, self.z / length)

    def to_tuple(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z


@dataclass
class Plane:
    normal: Vector3
    distance: float

    @classmethod
    def from_points(cls, p1: Vector3, p2: Vector3, p3: Vector3) -> "Plane":
        normal = (p2 - p1).cross(p3 - p1).normalize()
        distance = -normal.dot(p1)
        return cls(normal, distance)

    def distance_to_point(self, point: Vector3) -> float:
        return self.normal.dot(point) + self.distance


def intersect_planes(p1: Plane, p2: Plane, p3: Plane) -> Vector3 | None:
    """Find the intersection point of three planes."""

    cross23 = p2.normal.cross(p3.normal)
    cross31 = p3.normal.cross(p1.normal)
    cross12 = p1.normal.cross(p2.normal)

    denom = p1.normal.dot(cross23)
    if abs(denom) < EPSILON:
        return None

    term1 = cross23 * (-p1.distance)
    term2 = cross31 * (-p2.distance)
    term3 = cross12 * (-p3.distance)

    point = (term1 + term2 + term3) * (1.0 / denom)
    return point


def unique_points(points: Iterable[Vector3], eps: float = EPSILON) -> List[Vector3]:
    """Deduplicate points within the given epsilon."""

    unique: List[Vector3] = []
    for point in points:
        if not any((point - other).length() < eps for other in unique):
            unique.append(point)
    return unique


def sort_polygon_vertices(vertices: List[Vector3], normal: Vector3) -> List[Vector3]:
    """Sort polygon vertices in a consistent winding order."""

    if len(vertices) <= 2:
        return vertices

    center = Vector3(
        sum(v.x for v in vertices) / len(vertices),
        sum(v.y for v in vertices) / len(vertices),
        sum(v.z for v in vertices) / len(vertices),
    )

    # Build an orthonormal basis for the plane.
    u = Vector3(0.0, 0.0, 1.0).cross(normal)
    if u.length() < EPSILON:
        u = Vector3(1.0, 0.0, 0.0).cross(normal)
    u = u.normalize()
    v = normal.cross(u)

    def angle(vertex: Vector3) -> float:
        rel = vertex - center
        x = rel.dot(u)
        y = rel.dot(v)
        return atan2(y, x)

    return sorted(vertices, key=angle)


def triangulate(vertices: List[Vector3]) -> List[Tuple[Vector3, Vector3, Vector3]]:
    """Triangulate a convex polygon using a fan from the first vertex."""

    if len(vertices) < 3:
        return []
    triangles = []
    for i in range(1, len(vertices) - 1):
        triangles.append((vertices[0], vertices[i], vertices[i + 1]))
    return triangles
