"""Interactive 3D viewer for VMF geometry."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pyglet
from pyglet import gl
from pyglet.graphics import shader
from pyglet.math import Mat4, Vec3
from pyglet.window import key, mouse

from .geometry import Vector3, triangulate
from .parser import VMFMap

DEFAULT_FOV = 60.0
NEAR_CLIP = 1.0
FAR_CLIP = 65536.0
SKIP_MATERIAL_KEYWORDS = (
    "nodraw",
    "sky",
    "skybox",
    "clip",
    "trigger",
    "skip",
    "illusionary",
    "origin",
    "areaportal",
    "water",
)

VERTEX_SHADER = """
#version 330 core
uniform mat4 u_view_projection;
uniform vec4 u_color;
in vec3 position;
out vec4 v_color;
void main() {
    v_color = u_color;
    gl_Position = u_view_projection * vec4(position, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec4 v_color;
out vec4 fragColor;
void main() {
    fragColor = v_color;
}
"""


@dataclass
class RenderFace:
    normal: Vector3
    triangles: List[Tuple[Vector3, Vector3, Vector3]]
    outline: List[Vector3]


class Camera:
    def __init__(self) -> None:
        self.distance = 1024.0
        self.yaw = -45.0
        self.pitch = 35.0
        self.target = Vector3(0.0, 0.0, 0.0)

    def view_matrix(self) -> Mat4:
        matrix = Mat4()
        matrix = matrix @ Mat4.from_translation(Vec3(0.0, 0.0, -self.distance))
        matrix = matrix @ Mat4.from_rotation(math.radians(self.pitch), Vec3(1.0, 0.0, 0.0))
        matrix = matrix @ Mat4.from_rotation(math.radians(self.yaw), Vec3(0.0, 0.0, 1.0))
        matrix = matrix @ Mat4.from_translation(Vec3(-self.target.x, -self.target.y, -self.target.z))
        return matrix

    def fit_radius(self, radius: float) -> None:
        if radius <= 0.0:
            return
        self.distance = max(self.distance, radius * 1.2 + 128.0)


class VMFViewerWindow(pyglet.window.Window):
    def __init__(self, faces: Sequence[RenderFace], **kwargs) -> None:
        super().__init__(resizable=True, caption="VMF Viewer", **kwargs)
        self.faces = faces
        self.camera = Camera()
        self.program = shader.ShaderProgram(
            shader.Shader(VERTEX_SHADER, "vertex"),
            shader.Shader(FRAGMENT_SHADER, "fragment"),
        )
        vlists = self._build_vertex_list(faces)
        if vlists is not None:
            self._solid_vlist, self._outline_vlist = vlists
        else:
            self._solid_vlist, self._outline_vlist = None, None
        self._projection = Mat4()
        self._update_projection(self.width, self.height)
        self._init_gl()

    def _build_vertex_list(self, faces: Sequence[RenderFace]) -> tuple[Optional[shader.VertexList], Optional[shader.VertexList]] | None:
        if not faces:
            return None
        vertex_data: List[float] = []
        outline_data: List[float] = []
        for face in faces:
            for tri in face.triangles:
                for vertex in tri:
                    vertex_data.extend((vertex.x, vertex.y, vertex.z))
            outline = face.outline
            if len(outline) >= 2:
                for i in range(len(outline)):
                    a = outline[i]
                    b = outline[(i + 1) % len(outline)]
                    outline_data.extend((a.x, a.y, a.z, b.x, b.y, b.z))
        solid_vlist = self.program.vertex_list(
            len(vertex_data) // 3,
            gl.GL_TRIANGLES,
            position=("f", vertex_data),
        ) if vertex_data else None
        outline_vlist = self.program.vertex_list(
            len(outline_data) // 3,
            gl.GL_LINES,
            position=("f", outline_data),
        ) if outline_data else None
        if solid_vlist is None and outline_vlist is None:
            return None
        return solid_vlist, outline_vlist

    def _init_gl(self) -> None:
        gl.glClearColor(0.85, 0.85, 0.85, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glLineWidth(2.0)

    def _update_projection(self, width: int, height: int) -> None:
        width = max(1, int(width))
        height = max(1, int(height))
        gl.glViewport(0, 0, width, height)
        aspect = width / float(height)
        self._projection = Mat4.perspective_projection(aspect, NEAR_CLIP, FAR_CLIP, fov=DEFAULT_FOV)

    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        self._update_projection(width, height)

    def on_draw(self) -> None:
        self.clear()
        if self._solid_vlist is None and self._outline_vlist is None:
            return
        view_projection = self._projection @ self.camera.view_matrix()
        with self.program:
            self.program["u_view_projection"] = tuple(view_projection)
            if self._solid_vlist is not None:
                self.program["u_color"] = (1.0, 1.0, 1.0, 1.0)
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
                self._solid_vlist.draw(gl.GL_TRIANGLES)
            if self._outline_vlist is not None:
                self.program["u_color"] = (0.0, 0.0, 0.0, 1.0)
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
                gl.glLineWidth(2.0)
                self._outline_vlist.draw(gl.GL_LINES)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:
        if buttons & mouse.LEFT:
            self.camera.yaw += dx * 0.5
            self.camera.pitch = max(-89.0, min(89.0, self.camera.pitch + dy * 0.5))
        elif buttons & mouse.RIGHT:
            self.camera.distance = max(64.0, self.camera.distance * math.exp(-dy * 0.01))

    def on_mouse_scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        self.camera.distance = max(64.0, self.camera.distance * math.exp(-scroll_y * 0.1))

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        step = 32.0
        if symbol == key.W:
            self.camera.target = self.camera.target + Vector3(0.0, step, 0.0)
        elif symbol == key.S:
            self.camera.target = self.camera.target - Vector3(0.0, step, 0.0)
        elif symbol == key.A:
            self.camera.target = self.camera.target - Vector3(step, 0.0, 0.0)
        elif symbol == key.D:
            self.camera.target = self.camera.target + Vector3(step, 0.0, 0.0)
        elif symbol == key.Q:
            self.camera.target = self.camera.target - Vector3(0.0, 0.0, step)
        elif symbol == key.E:
            self.camera.target = self.camera.target + Vector3(0.0, 0.0, step)


def build_render_faces(vmf: VMFMap) -> List[RenderFace]:
    faces: List[RenderFace] = []
    for solid in vmf.solids:
        planes = [face.plane for face in solid.faces]
        for face in solid.faces:
            material_name = face.material.replace("\\", "/").lower()
            if any(keyword in material_name for keyword in SKIP_MATERIAL_KEYWORDS):
                continue
            vertices = face.polygon(planes)
            if len(vertices) < 3:
                continue
            triangles = triangulate(vertices)
            if not triangles:
                continue
            faces.append(
                RenderFace(
                    normal=face.plane.normal.normalize(),
                    triangles=triangles,
                    outline=list(vertices),
                )
            )
    return faces


def preview_vmf(vmf: VMFMap) -> None:
    faces = build_render_faces(vmf)
    window = VMFViewerWindow(faces, width=1280, height=720)
    if faces:
        min_x = float("inf")
        min_y = float("inf")
        min_z = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")
        max_z = float("-inf")
        for face in faces:
            for tri in face.triangles:
                for vertex in tri:
                    min_x = min(min_x, vertex.x)
                    min_y = min(min_y, vertex.y)
                    min_z = min(min_z, vertex.z)
                    max_x = max(max_x, vertex.x)
                    max_y = max(max_y, vertex.y)
                    max_z = max(max_z, vertex.z)
        if (min_x < max_x) and (min_y < max_y) and (min_z < max_z):
            center = Vector3((min_x + max_x) * 0.5, (min_y + max_y) * 0.5, (min_z + max_z) * 0.5)
            window.camera.target = center
            max_radius = 0.0
            for face in faces:
                for tri in face.triangles:
                    for vertex in tri:
                        dx = vertex.x - center.x
                        dy = vertex.y - center.y
                        dz = vertex.z - center.z
                        radius = math.sqrt(dx * dx + dy * dy + dz * dz)
                        if radius > max_radius:
                            max_radius = radius
            window.camera.fit_radius(max_radius)
    pyglet.app.run()


def preview_file(path: str) -> None:
    from .parser import load_vmf

    vmf = load_vmf(path)
    preview_vmf(vmf)
