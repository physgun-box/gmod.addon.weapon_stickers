"""Interactive 3D viewer for VMF geometry."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import pyglet
from pyglet import gl, shapes
from pyglet.graphics import Group, shader
from pyglet.math import Mat4, Vec3
from pyglet.window import key, mouse

from .geometry import Vector3, triangulate
from .parser import VMFMap

DEFAULT_FOV = 60.0
NEAR_CLIP = 1.0
FAR_CLIP = 1_000_000_000.0
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

SIDEBAR_WIDTH = 260
INSPECTOR_WIDTH = 300
TOP_BAR_HEIGHT = 36
BOTTOM_BAR_HEIGHT = 28
LINE_HEIGHT = 18
LIST_MARGIN = 14

PANEL_COLOR = (30, 34, 40)
HEADER_COLOR = (42, 48, 56)
STATUS_COLOR = (36, 42, 48)

HIERARCHY_COLOR = (215, 220, 228, 255)
HIERARCHY_SELECTED_COLOR = (255, 180, 110, 255)
HIERARCHY_HOVER_COLOR = (200, 205, 215, 255)

HIGHLIGHT_OUTLINE_COLOR = (0.25, 0.65, 1.0, 1.0)

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
    solid_index: int


@dataclass
class SolidEntry:
    label: str
    focus_point: Vector3
    face_indices: List[int]
    bounds_min: Vector3
    bounds_max: Vector3
    original_id: Optional[int]
    translation: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, 0.0))

    def world_focus(self) -> Vector3:
        return self.focus_point + self.translation

    def world_bounds(self) -> Tuple[Vector3, Vector3]:
        offset = self.translation
        return self.bounds_min + offset, self.bounds_max + offset


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
    def __init__(self, faces: Sequence[RenderFace], solids: Sequence[SolidEntry], **kwargs) -> None:
        super().__init__(resizable=True, caption="VMF Studio", **kwargs)
        self.faces = list(faces)
        self.solids = list(solids)
        self.camera = Camera()

        self._sidebar_width = SIDEBAR_WIDTH
        self._inspector_width = INSPECTOR_WIDTH
        self._top_bar_height = TOP_BAR_HEIGHT
        self._bottom_bar_height = BOTTOM_BAR_HEIGHT
        self._line_height = LINE_HEIGHT
        self._list_margin = LIST_MARGIN

        self._selected_index: Optional[int] = None
        self._hovered_index: Optional[int] = None
        self._hierarchy_scroll = 0.0
        self._selected_outline_data: List[float] = []
        self._hierarchy_labels: List[pyglet.text.Label] = []
        self._inspector_labels: List[pyglet.text.Label] = []
        self._edit_step = 32.0
        self._dirty = False

        self.program = shader.ShaderProgram(
            shader.Shader(VERTEX_SHADER, "vertex"),
            shader.Shader(FRAGMENT_SHADER, "fragment"),
        )

        self._solid_vlist: Optional[shader.VertexList] = None
        self._outline_vlist: Optional[shader.VertexList] = None
        self._projection = Mat4()

        self._update_projection(self.width, self.height)
        self._init_gl()
        self._rebuild_geometry()
        self._init_ui()
        self._layout_ui()
        self._rebuild_hierarchy_labels()
        self._update_inspector_labels()
        self._update_header_text()
        self._update_status_label()

    def _build_vertex_list(
        self, faces: Sequence[RenderFace]
    ) -> tuple[Optional[shader.VertexList], Optional[shader.VertexList]] | None:
        if not faces:
            return None
        vertex_data: List[float] = []
        outline_data: List[float] = []
        for face in faces:
            offset = self.solids[face.solid_index].translation if 0 <= face.solid_index < len(self.solids) else Vector3(0.0, 0.0, 0.0)
            for tri in face.triangles:
                for vertex in tri:
                    v = vertex + offset
                    vertex_data.extend((v.x, v.y, v.z))
            outline = face.outline
            if len(outline) >= 2:
                for i in range(len(outline)):
                    a = outline[i] + offset
                    b = outline[(i + 1) % len(outline)] + offset
                    outline_data.extend((a.x, a.y, a.z, b.x, b.y, b.z))
        solid_vlist = (
            self.program.vertex_list(
                len(vertex_data) // 3,
                gl.GL_TRIANGLES,
                position=("f", vertex_data),
            )
            if vertex_data
            else None
        )
        outline_vlist = (
            self.program.vertex_list(
                len(outline_data) // 3,
                gl.GL_LINES,
                position=("f", outline_data),
            )
            if outline_data
            else None
        )
        if solid_vlist is None and outline_vlist is None:
            return None
        return solid_vlist, outline_vlist

    def _rebuild_geometry(self) -> None:
        if self._solid_vlist is not None:
            self._solid_vlist.delete()
            self._solid_vlist = None
        if self._outline_vlist is not None:
            self._outline_vlist.delete()
            self._outline_vlist = None
        vlists = self._build_vertex_list(self.faces)
        if vlists is not None:
            self._solid_vlist, self._outline_vlist = vlists

    def _init_gl(self) -> None:
        gl.glClearColor(0.05, 0.05, 0.05, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glLineWidth(1.5)

    def _init_ui(self) -> None:
        self._ui_batch = pyglet.graphics.Batch()
        self._ui_group_bg = Group(order=0)
        self._ui_group_text = Group(order=1)

        self._left_panel = shapes.Rectangle(
            0,
            0,
            self._sidebar_width,
            self.height,
            color=PANEL_COLOR,
            batch=self._ui_batch,
            group=self._ui_group_bg,
        )
        self._right_panel = shapes.Rectangle(
            self.width - self._inspector_width,
            0,
            self._inspector_width,
            self.height,
            color=PANEL_COLOR,
            batch=self._ui_batch,
            group=self._ui_group_bg,
        )
        self._top_bar = shapes.Rectangle(
            self._sidebar_width,
            self.height - self._top_bar_height,
            max(0, self.width - self._sidebar_width - self._inspector_width),
            self._top_bar_height,
            color=HEADER_COLOR,
            batch=self._ui_batch,
            group=self._ui_group_bg,
        )
        self._bottom_bar = shapes.Rectangle(
            self._sidebar_width,
            0,
            max(0, self.width - self._sidebar_width - self._inspector_width),
            self._bottom_bar_height,
            color=STATUS_COLOR,
            batch=self._ui_batch,
            group=self._ui_group_bg,
        )
        for panel in (self._left_panel, self._right_panel, self._top_bar, self._bottom_bar):
            panel.opacity = 235

        self._hierarchy_title = pyglet.text.Label(
            "Hierarchy",
            font_size=12,
            weight="bold",
            x=16,
            y=self.height - 40,
            color=(240, 240, 240, 255),
            batch=self._ui_batch,
            group=self._ui_group_text,
        )
        self._inspector_title = pyglet.text.Label(
            "Inspector",
            font_size=12,
            weight="bold",
            x=self.width - self._inspector_width + 16,
            y=self.height - 40,
            color=(240, 240, 240, 255),
            batch=self._ui_batch,
            group=self._ui_group_text,
        )
        self._header_label = pyglet.text.Label(
            "",
            font_size=11,
            x=self._sidebar_width + 12,
            y=self.height - self._top_bar_height / 2,
            anchor_y="center",
            batch=self._ui_batch,
            group=self._ui_group_text,
            color=(230, 230, 230, 255),
        )
        self._header_right_label = pyglet.text.Label(
            "",
            font_size=11,
            x=self.width - self._inspector_width - 12,
            y=self.height - self._top_bar_height / 2,
            anchor_x="right",
            anchor_y="center",
            batch=self._ui_batch,
            group=self._ui_group_text,
            color=(200, 200, 200, 255),
        )
        self._status_label = pyglet.text.Label(
            "",
            font_size=11,
            x=self._sidebar_width + 12,
            y=self._bottom_bar_height / 2,
            anchor_y="center",
            batch=self._ui_batch,
            group=self._ui_group_text,
            color=(210, 210, 210, 255),
        )

    def _layout_ui(self) -> None:
        self._left_panel.width = self._sidebar_width
        self._left_panel.height = self.height

        self._right_panel.x = self.width - self._inspector_width
        self._right_panel.width = self._inspector_width
        self._right_panel.height = self.height

        self._top_bar.x = self._sidebar_width
        self._top_bar.y = self.height - self._top_bar_height
        self._top_bar.width = max(0, self.width - self._sidebar_width - self._inspector_width)
        self._top_bar.height = self._top_bar_height

        self._bottom_bar.x = self._sidebar_width
        self._bottom_bar.y = 0
        self._bottom_bar.width = max(0, self.width - self._sidebar_width - self._inspector_width)
        self._bottom_bar.height = self._bottom_bar_height

        self._hierarchy_title.x = 16
        self._hierarchy_title.y = self.height - self._top_bar_height - 6
        self._inspector_title.x = self.width - self._inspector_width + 16
        self._inspector_title.y = self.height - self._top_bar_height - 6

        self._header_label.x = self._sidebar_width + 12
        self._header_label.y = self.height - self._top_bar_height / 2
        self._header_right_label.x = self.width - self._inspector_width - 12
        self._header_right_label.y = self.height - self._top_bar_height / 2

        self._status_label.x = self._sidebar_width + 12
        self._status_label.y = self._bottom_bar_height / 2

        self._layout_hierarchy_labels()
        self._update_inspector_labels()

    def _layout_hierarchy_labels(self) -> None:
        start_y = self.height - self._top_bar_height - self._list_margin
        for index, label in enumerate(self._hierarchy_labels):
            label.x = 16
            label.y = start_y - index * self._line_height + self._hierarchy_scroll

    def _refresh_label_colors(self) -> None:
        for index, label in enumerate(self._hierarchy_labels):
            if index == self._selected_index:
                label.color = HIERARCHY_SELECTED_COLOR
            elif index == self._hovered_index:
                label.color = HIERARCHY_HOVER_COLOR
            else:
                label.color = HIERARCHY_COLOR

    def _adjust_hierarchy_scroll(self, delta: float) -> None:
        content_height = len(self._hierarchy_labels) * self._line_height
        available_height = self.height - self._top_bar_height - self._bottom_bar_height - 2 * self._list_margin
        if content_height <= available_height:
            self._hierarchy_scroll = 0.0
        else:
            min_scroll = available_height - content_height
            self._hierarchy_scroll = max(min(self._hierarchy_scroll + delta, 0.0), min_scroll)
        self._layout_hierarchy_labels()

    def _rebuild_hierarchy_labels(self) -> None:
        for label in self._hierarchy_labels:
            label.delete()
        self._hierarchy_labels = []
        for solid in self.solids:
            label = pyglet.text.Label(
                solid.label,
                font_size=12,
                x=16,
                y=0,
                anchor_y="top",
                batch=self._ui_batch,
                group=self._ui_group_text,
                color=HIERARCHY_COLOR,
            )
            self._hierarchy_labels.append(label)
        self._hierarchy_scroll = 0.0
        self._layout_hierarchy_labels()
        self._refresh_label_colors()

    def _update_inspector_labels(self) -> None:
        for label in self._inspector_labels:
            label.delete()
        self._inspector_labels = []
        x = self.width - self._inspector_width + 16
        top = self.height - self._top_bar_height - self._list_margin
        lines: List[str]
        if self._selected_index is None:
            lines = [
                "Select a solid to inspect.",
                "",
                "Tips:",
                "- Click to focus a solid",
                "- Ctrl+Arrows/PgUp/PgDn moves",
                "- [ and ] adjust move step",
                "- F frames the selection",
            ]
        else:
            solid = self.solids[self._selected_index]
            bounds_min, bounds_max = solid.world_bounds()
            focus = solid.world_focus()
            lines = [
                f"Name: {solid.label}",
                f"Solid ID: {solid.original_id if solid.original_id is not None else '-'}",
                f"Faces: {len(solid.face_indices)}",
                f"Center: ({focus.x:.1f}, {focus.y:.1f}, {focus.z:.1f})",
                f"Bounds min: ({bounds_min.x:.1f}, {bounds_min.y:.1f}, {bounds_min.z:.1f})",
                f"Bounds max: ({bounds_max.x:.1f}, {bounds_max.y:.1f}, {bounds_max.z:.1f})",
                f"Offset: ({solid.translation.x:.1f}, {solid.translation.y:.1f}, {solid.translation.z:.1f})",
                "",
                "Editing:",
                "Ctrl+Arrows / PgUp / PgDn",
                f"Step: {self._edit_step:.1f} units",
                "[ / ] adjust step",
                "F to frame selection",
            ]
        for index, text in enumerate(lines):
            label = pyglet.text.Label(
                text,
                font_size=11,
                x=x,
                y=top - index * (self._line_height + 2),
                anchor_y="top",
                color=(220, 220, 220, 255),
                batch=self._ui_batch,
                group=self._ui_group_text,
            )
            self._inspector_labels.append(label)

    def _update_status_label(self) -> None:
        if self._selected_index is None:
            name = "None"
            offset_x = offset_y = offset_z = 0.0
        else:
            solid = self.solids[self._selected_index]
            name = solid.label
            offset_x, offset_y, offset_z = solid.translation.x, solid.translation.y, solid.translation.z
        self._status_label.text = (
            f"Selected: {name} | Offset: ({offset_x:.1f}, {offset_y:.1f}, {offset_z:.1f}) "
            f"| Step: {self._edit_step:.1f} | Camera Dist: {self.camera.distance:.1f}"
        )

    def _update_header_text(self) -> None:
        self._header_label.text = (
            "Camera: LMB orbit | RMB drag zoom | Wheel zoom | Edit: Ctrl+Arrows / PgUp / PgDn"
        )
        suffix = " * modified" if self._dirty else ""
        self._header_right_label.text = f"Solids: {len(self.solids)}{suffix}"

    def _rebuild_selected_outline(self) -> None:
        if self._selected_index is None:
            self._selected_outline_data = []
            return
        solid = self.solids[self._selected_index]
        offset = solid.translation
        data: List[float] = []
        for face_index in solid.face_indices:
            face = self.faces[face_index]
            outline = face.outline
            if len(outline) < 2:
                continue
            for i in range(len(outline)):
                a = outline[i] + offset
                b = outline[(i + 1) % len(outline)] + offset
                data.extend((a.x, a.y, a.z, b.x, b.y, b.z))
        self._selected_outline_data = data

    def _update_projection(self, width: int, height: int) -> None:
        width = max(1, int(width))
        height = max(1, int(height))
        gl.glViewport(0, 0, width, height)
        aspect = width / float(height)
        self._projection = Mat4.perspective_projection(aspect, NEAR_CLIP, FAR_CLIP, fov=DEFAULT_FOV)

    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        self._update_projection(width, height)
        self._layout_ui()
        self._refresh_label_colors()
        self._rebuild_selected_outline()
        self._update_status_label()

    def on_draw(self) -> None:
        self.clear()
        if self._solid_vlist is not None or self._outline_vlist is not None:
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
                    gl.glLineWidth(1.5)
                    self._outline_vlist.draw(gl.GL_LINES)
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        if self._selected_outline_data:
            count = len(self._selected_outline_data) // 3
            color_data = HIGHLIGHT_OUTLINE_COLOR * count
            gl.glLineWidth(3.0)
            pyglet.graphics.draw(
                count,
                gl.GL_LINES,
                position=("f", self._selected_outline_data),
                colors=("f", color_data),
            )
            gl.glLineWidth(1.0)
        gl.glDisable(gl.GL_DEPTH_TEST)
        self._ui_batch.draw()
        gl.glEnable(gl.GL_DEPTH_TEST)

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:
        if buttons & mouse.LEFT:
            self.camera.yaw += dx * 0.4
            self.camera.pitch = max(-89.0, min(89.0, self.camera.pitch + dy * 0.4))
        elif buttons & mouse.RIGHT:
            self.camera.distance = max(64.0, self.camera.distance * math.exp(-dy * 0.01))
        self._update_status_label()

    def on_mouse_scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        if x < self._sidebar_width:
            self._adjust_hierarchy_scroll(scroll_y * self._line_height * 2)
            return
        if x > self.width - self._inspector_width:
            return
        self.camera.distance = max(64.0, self.camera.distance * math.exp(-scroll_y * 0.1))
        self._update_status_label()

    def _hierarchy_at_position(self, x: int, y: int) -> Optional[int]:
        if x >= self._sidebar_width:
            return None
        top = self.height - self._top_bar_height - self._list_margin
        bottom = self._bottom_bar_height + self._list_margin
        if y > top or y < bottom:
            return None
        local = top - y + self._hierarchy_scroll
        if local < 0:
            return None
        index = int(local // self._line_height)
        if 0 <= index < len(self.solids):
            return index
        return None

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        hovered = self._hierarchy_at_position(x, y)
        if hovered != self._hovered_index:
            self._hovered_index = hovered
            self._refresh_label_colors()

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        if button != mouse.LEFT:
            return
        index = self._hierarchy_at_position(x, y)
        if index is not None:
            self._set_selected_index(index)
        elif x < self._sidebar_width:
            self._set_selected_index(None)

    def _set_selected_index(self, index: Optional[int]) -> None:
        if index == self._selected_index:
            return
        self._selected_index = index
        if index is not None:
            solid = self.solids[index]
            self.camera.target = solid.world_focus()
            self.camera.distance = max(self.camera.distance, 256.0)
        self._refresh_label_colors()
        self._update_inspector_labels()
        self._rebuild_selected_outline()
        self._update_status_label()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if modifiers & key.MOD_CTRL:
            if symbol == key.LEFT:
                self._nudge_selected(Vector3(-self._edit_step, 0.0, 0.0))
                return
            if symbol == key.RIGHT:
                self._nudge_selected(Vector3(self._edit_step, 0.0, 0.0))
                return
            if symbol == key.UP:
                self._nudge_selected(Vector3(0.0, self._edit_step, 0.0))
                return
            if symbol == key.DOWN:
                self._nudge_selected(Vector3(0.0, -self._edit_step, 0.0))
                return
            if symbol == key.PAGEUP:
                self._nudge_selected(Vector3(0.0, 0.0, self._edit_step))
                return
            if symbol == key.PAGEDOWN:
                self._nudge_selected(Vector3(0.0, 0.0, -self._edit_step))
                return
        if symbol == key.ESCAPE:
            self._set_selected_index(None)
        elif symbol == key.W:
            self.camera.target = self.camera.target + Vector3(0.0, 32.0, 0.0)
        elif symbol == key.S:
            self.camera.target = self.camera.target - Vector3(0.0, 32.0, 0.0)
        elif symbol == key.A:
            self.camera.target = self.camera.target - Vector3(32.0, 0.0, 0.0)
        elif symbol == key.D:
            self.camera.target = self.camera.target + Vector3(32.0, 0.0, 0.0)
        elif symbol == key.Q:
            self.camera.target = self.camera.target - Vector3(0.0, 0.0, 32.0)
        elif symbol == key.E:
            self.camera.target = self.camera.target + Vector3(0.0, 0.0, 32.0)
        elif symbol == key.F:
            self._frame_selected()
        elif symbol == key.BRACKETLEFT:
            self._adjust_edit_step(0.5)
        elif symbol == key.BRACKETRIGHT:
            self._adjust_edit_step(2.0)
        self._update_status_label()

    def _nudge_selected(self, delta: Vector3) -> None:
        if self._selected_index is None:
            return
        solid = self.solids[self._selected_index]
        solid.translation = Vector3(
            solid.translation.x + delta.x,
            solid.translation.y + delta.y,
            solid.translation.z + delta.z,
        )
        self.camera.target = solid.world_focus()
        self._dirty = True
        self._rebuild_geometry()
        self._rebuild_selected_outline()
        self._update_inspector_labels()
        self._update_status_label()
        self._update_header_text()

    def _frame_selected(self) -> None:
        if self._selected_index is None:
            return
        solid = self.solids[self._selected_index]
        center = solid.world_focus()
        bounds_min, bounds_max = solid.world_bounds()
        dx = bounds_max.x - bounds_min.x
        dy = bounds_max.y - bounds_min.y
        dz = bounds_max.z - bounds_min.z
        radius = 0.5 * math.sqrt(dx * dx + dy * dy + dz * dz)
        self.camera.target = center
        self.camera.fit_radius(max(radius, 64.0))
        self._update_status_label()

    def _adjust_edit_step(self, factor: float) -> None:
        self._edit_step = max(0.25, min(8192.0, self._edit_step * factor))
        self._update_inspector_labels()
        self._update_status_label()


def build_render_data(vmf: VMFMap) -> Tuple[List[RenderFace], List[SolidEntry]]:
    faces: List[RenderFace] = []
    solids: List[SolidEntry] = []
    for index, solid in enumerate(vmf.solids):
        planes = [face.plane for face in solid.faces]
        solid_index = len(solids)
        face_indices: List[int] = []
        focus_point: Optional[Vector3] = None
        face_count = 0
        min_x = float("inf")
        min_y = float("inf")
        min_z = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")
        max_z = float("-inf")
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
            if focus_point is None:
                sx = sum(v.x for v in vertices) / len(vertices)
                sy = sum(v.y for v in vertices) / len(vertices)
                sz = sum(v.z for v in vertices) / len(vertices)
                focus_point = Vector3(sx, sy, sz)
            for vertex in vertices:
                min_x = min(min_x, vertex.x)
                min_y = min(min_y, vertex.y)
                min_z = min(min_z, vertex.z)
                max_x = max(max_x, vertex.x)
                max_y = max(max_y, vertex.y)
                max_z = max(max_z, vertex.z)
            face_index = len(faces)
            faces.append(
                RenderFace(
                    normal=face.plane.normal.normalize(),
                    triangles=triangles,
                    outline=list(vertices),
                    solid_index=solid_index,
                )
            )
            face_indices.append(face_index)
            face_count += 1
        if focus_point is not None and face_count:
            if min_x == float("inf"):
                min_x = max_x = focus_point.x
                min_y = max_y = focus_point.y
                min_z = max_z = focus_point.z
            entry = SolidEntry(
                label=f"Solid {solid.id if solid.id is not None else index}",
                focus_point=focus_point,
                face_indices=face_indices,
                bounds_min=Vector3(min_x, min_y, min_z),
                bounds_max=Vector3(max_x, max_y, max_z),
                original_id=getattr(solid, "id", None),
            )
            solids.append(entry)
    return faces, solids


def preview_vmf(vmf: VMFMap) -> None:
    faces, solids = build_render_data(vmf)
    window = VMFViewerWindow(faces, solids, width=1280, height=720)
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
