"""Programmatic VMF builder."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from .geometry import Vector3


@dataclass
class BrushFace:
    a: Vector3
    b: Vector3
    c: Vector3
    material: str = "TOOLS/TOOLSNODRAW"
    uaxis: str = "[1 0 0 0] 0.25"
    vaxis: str = "[0 1 0 0] 0.25"
    rotation: float = 0.0
    lightmapscale: float = 16.0
    smoothing_groups: int = 0

    def to_vmf(self, face_id: int) -> str:
        return (
            '\t\tside\n'
            '\t\t{\n'
            f'\t\t\t"id" "{face_id}"\n'
            f'\t\t\t"plane" "({self.a.x} {self.a.y} {self.a.z}) ({self.b.x} {self.b.y} {self.b.z}) ({self.c.x} {self.c.y} {self.c.z})"\n'
            f'\t\t\t"material" "{self.material}"\n'
            f'\t\t\t"uaxis" "{self.uaxis}"\n'
            f'\t\t\t"vaxis" "{self.vaxis}"\n'
            f'\t\t\t"rotation" "{self.rotation}"\n'
            f'\t\t\t"lightmapscale" "{self.lightmapscale}"\n'
            f'\t\t\t"smoothing_groups" "{self.smoothing_groups}"\n'
            '\t\t}\n'
        )


@dataclass
class Brush:
    faces: List[BrushFace]
    editor_color: str = "0 255 0 255"

    def to_vmf(self, solid_id: int, face_id_start: int) -> tuple[str, int]:
        parts = ['\tsolid\n\t{\n', f'\t\t"id" "{solid_id}"\n']
        face_id = face_id_start
        for face in self.faces:
            parts.append(face.to_vmf(face_id))
            face_id += 1
        parts.append(
            '\t\teditor\n'
            '\t\t{\n'
            f'\t\t\t"color" "{self.editor_color}"\n'
            '\t\t\t"visgroupshown" "1"\n'
            '\t\t\t"visgroupautoshown" "1"\n'
            '\t\t}\n'
            '\t}\n'
        )
        return ''.join(parts), face_id


@dataclass
class Entity:
    classname: str
    properties: Dict[str, str] = field(default_factory=dict)
    brushes: List[Brush] = field(default_factory=list)

    def to_vmf(self, entity_id: int, solid_id_start: int, face_id_start: int) -> tuple[str, int, int]:
        extra = {key: str(value) for key, value in self.properties.items()}
        props = {"id": str(entity_id), "classname": self.classname, **extra}
        parts = ['entity\n{\n']
        for key, value in props.items():
            parts.append(f'\t"{key}" "{value}"\n')
        solid_id = solid_id_start
        face_id = face_id_start
        for brush in self.brushes:
            solid_text, face_id = brush.to_vmf(solid_id, face_id)
            solid_id += 1
            parts.append(solid_text)
        parts.append(
            '\teditor\n'
            '\t{\n'
            '\t\t"color" "220 30 220"\n'
            '\t\t"visgroupshown" "1"\n'
            '\t\t"visgroupautoshown" "1"\n'
            '\t}\n'
            '}\n'
        )
        return ''.join(parts), solid_id, face_id


class VMFBuilder:
    def __init__(self) -> None:
        self.world_brushes: List[Brush] = []
        self.entities: List[Entity] = []

    @staticmethod
    def axis_aligned_brush(
        min_corner: Vector3,
        max_corner: Vector3,
        material: str = "DEV/DEV_MEASUREWALL01C",
        editor_color: str = "0 255 0 255",
    ) -> Brush:
        """Create a rectangular brush aligned to the world axes."""

        x0, y0, z0 = min_corner.x, min_corner.y, min_corner.z
        x1, y1, z1 = max_corner.x, max_corner.y, max_corner.z
        faces = [
            BrushFace(Vector3(x0, y0, z1), Vector3(x1, y0, z1), Vector3(x1, y1, z1), material),  # top
            BrushFace(Vector3(x1, y0, z0), Vector3(x0, y0, z0), Vector3(x0, y1, z0), material),  # bottom
            BrushFace(Vector3(x0, y0, z0), Vector3(x0, y0, z1), Vector3(x0, y1, z1), material),  # west
            BrushFace(Vector3(x1, y1, z0), Vector3(x1, y1, z1), Vector3(x1, y0, z1), material),  # east
            BrushFace(Vector3(x0, y1, z0), Vector3(x0, y1, z1), Vector3(x1, y1, z1), material),  # north
            BrushFace(Vector3(x1, y0, z0), Vector3(x1, y0, z1), Vector3(x0, y0, z1), material),  # south
        ]
        return Brush(faces=faces, editor_color=editor_color)

    def add_axis_aligned_block(
        self,
        min_corner: Vector3,
        max_corner: Vector3,
        material: str = "DEV/DEV_MEASUREWALL01C",
    ) -> None:
        """Add a rectangular block aligned to world axes to the worldspawn."""

        self.world_brushes.append(self.axis_aligned_brush(min_corner, max_corner, material))

    def add_entity(self, entity: Entity) -> None:
        self.entities.append(entity)

    def _world_to_vmf(self, solid_id: int, face_id: int) -> tuple[str, int, int]:
        parts = [
            'world\n{\n',
            '\t"id" "1"\n',
            '\t"mapversion" "1"\n',
            '\t"classname" "worldspawn"\n',
            '\t"skyname" "sky_day01_01"\n',
        ]
        for brush in self.world_brushes:
            solid_text, face_id = brush.to_vmf(solid_id, face_id)
            solid_id += 1
            parts.append(solid_text)
        parts.append('}\n')
        return ''.join(parts), solid_id, face_id

    def _entities_to_vmf(
        self, entity_id: int, solid_id: int, face_id: int
    ) -> tuple[str, int, int, int]:
        parts: List[str] = []
        for entity in self.entities:
            text, solid_id, face_id = entity.to_vmf(entity_id, solid_id, face_id)
            parts.append(text)
            entity_id += 1
        return ''.join(parts), entity_id, solid_id, face_id

    def build(self) -> str:
        solid_id = 2  # reserve 1 for world entity id
        face_id = 1
        entity_id = 2  # 1 is implicitly used by worldspawn

        world_text, solid_id, face_id = self._world_to_vmf(solid_id, face_id)
        entities_text, entity_id, solid_id, face_id = self._entities_to_vmf(entity_id, solid_id, face_id)

        parts = [
            'versioninfo\n{\n\t"editorversion" "400"\n\t"editorbuild" "9087"\n}\n',
            'visgroups\n{\n}\n',
            'viewsettings\n{\n\t"bSnapToGrid" "1"\n\t"nGridSpacing" "64"\n}\n',
            world_text,
            entities_text,
        ]
        return ''.join(parts)

    def save(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.build())
