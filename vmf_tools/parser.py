"""VMF parser capable of building brush geometry."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Tuple

from .geometry import Plane, Vector3, intersect_planes, sort_polygon_vertices, triangulate, unique_points


@dataclass
class VMFFace:
    plane: Plane
    points: Tuple[Vector3, Vector3, Vector3]
    material: str
    uaxis: str = ""
    vaxis: str = ""
    rotation: float = 0.0
    lightmapscale: float = 16.0
    smoothing_groups: str = "0"
    id: Optional[int] = None

    def polygon(self, planes: List[Plane]) -> List[Vector3]:
        """Compute polygon vertices by intersecting this plane with others."""

        candidates: List[Vector3] = []
        for i in range(len(planes)):
            if planes[i] is self.plane:
                continue
            for j in range(i + 1, len(planes)):
                if planes[j] is self.plane:
                    continue
                point = intersect_planes(self.plane, planes[i], planes[j])
                if point is None:
                    continue
                if abs(self.plane.distance_to_point(point)) > 0.1:
                    continue
                inside = True
                for plane in planes:
                    if plane is self.plane:
                        continue
                    if plane.distance_to_point(point) < -0.1:
                        inside = False
                        break
                if inside:
                    candidates.append(point)
        unique = unique_points(candidates, eps=0.1)
        return sort_polygon_vertices(unique, self.plane.normal)


@dataclass
class VMFSolid:
    id: int
    faces: List[VMFFace]

    def build_geometry(self) -> List[Tuple[Vector3, Vector3, Vector3]]:
        """Return triangles for rendering."""

        planes = [face.plane for face in self.faces]
        triangles: List[Tuple[Vector3, Vector3, Vector3]] = []
        for face in self.faces:
            vertices = face.polygon(planes)
            triangles.extend(triangulate(vertices))
        return triangles


@dataclass
class VMFEntity:
    id: int
    classname: str
    properties: Dict[str, str]
    solids: List[VMFSolid] = field(default_factory=list)


@dataclass
class VMFMap:
    solids: List[VMFSolid]
    entities: List[VMFEntity]


class VMFParserError(RuntimeError):
    pass


class TokenStream:
    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.index = 0

    def peek(self) -> Optional[str]:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def pop(self) -> str:
        token = self.peek()
        if token is None:
            raise VMFParserError("Unexpected end of file")
        self.index += 1
        return token


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    i = 0
    length = len(text)
    while i < length:
        ch = text[i]
        if ch in "\n\r\t \v\f":
            i += 1
            continue
        if ch == '/' and i + 1 < length and text[i + 1] == '/':
            while i < length and text[i] not in "\n\r":
                i += 1
            continue
        if ch in '{}':
            tokens.append(ch)
            i += 1
            continue
        if ch == '"':
            i += 1
            value = []
            while i < length:
                if text[i] == '\\' and i + 1 < length:
                    value.append(text[i + 1])
                    i += 2
                    continue
                if text[i] == '"':
                    break
                value.append(text[i])
                i += 1
            else:
                raise VMFParserError("Unterminated string literal")
            tokens.append(''.join(value))
            i += 1
            continue
        # bare word
        start = i
        while i < length and text[i] not in "\n\r\t \v\f{}":
            i += 1
        tokens.append(text[start:i])
    return tokens


def parse_block(stream: TokenStream) -> List[Tuple[str, object]]:
    result: List[Tuple[str, object]] = []
    while True:
        token = stream.peek()
        if token is None:
            raise VMFParserError("Unexpected end of block")
        if token == '}':
            stream.pop()
            break
        key = stream.pop()
        next_token = stream.peek()
        if next_token == '{':
            stream.pop()
            value = parse_block(stream)
            result.append((key, value))
        else:
            if next_token is None:
                raise VMFParserError("Unexpected end of file after key")
            value = stream.pop()
            result.append((key, value))
    return result


def parse_tree(tokens: List[str]) -> List[Tuple[str, object]]:
    stream = TokenStream(tokens)
    result: List[Tuple[str, object]] = []
    while stream.peek() is not None:
        key = stream.pop()
        if stream.peek() != '{':
            raise VMFParserError(f"Expected '{{' after {key}")
        stream.pop()
        value = parse_block(stream)
        result.append((key, value))
    return result


def tree_to_dict(block: List[Tuple[str, object]]) -> Dict[str, List[object]]:
    result: Dict[str, List[object]] = {}
    for key, value in block:
        result.setdefault(key, []).append(value)
    return result


def parse_vector(text: str) -> Vector3:
    stripped = text.replace('(', '').replace(')', '')
    parts = [p for p in stripped.split() if p]
    if len(parts) != 3:
        raise VMFParserError(f"Invalid vector: {text}")
    x, y, z = map(float, parts)
    return Vector3(x, y, z)


def parse_solid(block: List[Tuple[str, object]]) -> VMFSolid:
    data = tree_to_dict(block)
    solid_id = int(data.get('id', ['0'])[0])
    faces: List[VMFFace] = []
    for side_block in data.get('side', []):
        faces.append(parse_face(side_block))
    return VMFSolid(id=solid_id, faces=faces)


def parse_face(block: List[Tuple[str, object]]) -> VMFFace:
    props = {key: value[0] if isinstance(value, list) else value for key, value in tree_to_dict(block).items()}
    matches = re.findall(r"\(([^)]+)\)", props['plane'])
    plane_points = [parse_vector(match) for match in matches]
    if len(plane_points) != 3:
        raise VMFParserError("Face plane must contain three points")
    p1, p2, p3 = plane_points
    plane = Plane.from_points(p1, p2, p3)
    return VMFFace(
        plane=plane,
        points=(p1, p2, p3),
        material=props.get('material', 'TOOLS/TOOLSNODRAW'),
        uaxis=props.get('uaxis', ''),
        vaxis=props.get('vaxis', ''),
        rotation=float(props.get('rotation', '0')),
        lightmapscale=float(props.get('lightmapscale', '16')),
        smoothing_groups=props.get('smoothing_groups', '0'),
        id=int(props.get('id', '0')),
    )


def parse_entity(block: List[Tuple[str, object]]) -> VMFEntity:
    data = tree_to_dict(block)
    props: Dict[str, str] = {}
    for key, values in data.items():
        if key in {'solid', 'hidden', 'editor', 'connections'}:
            continue
        props[key] = values[0] if isinstance(values[0], str) else values[0][0]
    entity_id = int(props.get('id', props.get('hammerid', '0')))
    classname = props.get('classname', '')
    solids: List[VMFSolid] = []
    for solid_block in data.get('solid', []):
        if isinstance(solid_block, list):
            solids.append(parse_solid(solid_block))
    return VMFEntity(id=entity_id, classname=classname, properties=props, solids=solids)


def parse_vmf(text: str) -> VMFMap:
    tokens = tokenize(text)
    tree = parse_tree(tokens)
    solids: List[VMFSolid] = []
    entities: List[VMFEntity] = []
    for key, block in tree:
        if key == 'world':
            world_data = tree_to_dict(block)
            for solid_block in world_data.get('solid', []):
                if isinstance(solid_block, list):
                    solids.append(parse_solid(solid_block))
        elif key == 'entity':
            entities.append(parse_entity(block))
    # include brush entities geometry as part of solids list for rendering
    for entity in entities:
        solids.extend(entity.solids)
    return VMFMap(solids=solids, entities=entities)


def load_vmf(path: str) -> VMFMap:
    with open(path, 'r', encoding='utf-8') as f:
        return parse_vmf(f.read())
