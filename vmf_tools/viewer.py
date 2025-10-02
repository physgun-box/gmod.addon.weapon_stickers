"""Interactive 3D viewer for VMF geometry."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import pyglet
from pyglet.gl import (
    GL_COLOR_MATERIAL,
    GL_CULL_FACE,
    GL_DEPTH_TEST,
    GL_LIGHT0,
    GL_LIGHTING,
    GL_POSITION,
    GL_TRIANGLES,
    GLfloat,
    glBegin,
    glClearColor,
    glColor3f,
    glDisable,
    glEnable,
    glEnd,
    glLightfv,
    glLightModelfv,
    glLoadIdentity,
    glMatrixMode,
    glNormal3f,
    glRotatef,
    glTranslatef,
    glVertex3f,
)
from pyglet.gl import GL_LIGHT_MODEL_AMBIENT, GL_MODELVIEW, GL_PROJECTION
from pyglet.window import key, mouse

from .geometry import Vector3, triangulate


def _vec4(x: float, y: float, z: float, w: float) -> pyglet.gl.GLfloat * 4:
    return (GLfloat * 4)(x, y, z, w)
from .parser import VMFMap


@dataclass
class RenderFace:
    normal: Vector3
    triangles: List[Tuple[Vector3, Vector3, Vector3]]


class Camera:
    def __init__(self) -> None:
        self.distance = 1024.0
        self.yaw = -45.0
        self.pitch = 35.0
        self.target = Vector3(0.0, 0.0, 0.0)

    def apply(self) -> None:
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -self.distance)
        glRotatef(self.pitch, 1.0, 0.0, 0.0)
        glRotatef(self.yaw, 0.0, 0.0, 1.0)
        glTranslatef(-self.target.x, -self.target.y, -self.target.z)


class VMFViewerWindow(pyglet.window.Window):
    def __init__(self, faces: Sequence[RenderFace], **kwargs) -> None:
        super().__init__(resizable=True, caption="VMF Viewer", **kwargs)
        self.faces = faces
        self.camera = Camera()
        self._dragging = False
        self._last_mouse = (0, 0)
        self._init_gl()

    def _init_gl(self) -> None:
        glClearColor(0.08, 0.08, 0.08, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, _vec4(0.3, 0.4, 1.0, 0.0))
        glLightfv(GL_LIGHT0, pyglet.gl.GL_DIFFUSE, _vec4(0.8, 0.8, 0.8, 1.0))
        glLightfv(GL_LIGHT0, pyglet.gl.GL_SPECULAR, _vec4(0.4, 0.4, 0.4, 1.0))
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, _vec4(0.2, 0.2, 0.2, 1.0))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        pyglet.gl.gluPerspective(60.0, self.width / max(1.0, float(self.height)), 1.0, 8192.0)

    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        pyglet.gl.gluPerspective(60.0, width / max(1.0, float(height)), 1.0, 8192.0)
        glMatrixMode(GL_MODELVIEW)

    def on_draw(self) -> None:
        self.clear()
        self.camera.apply()
        glBegin(GL_TRIANGLES)
        light_dir = Vector3(0.3, 0.4, 1.0).normalize()
        for face in self.faces:
            intensity = max(0.15, face.normal.normalize().dot(light_dir))
            glColor3f(intensity, intensity, intensity)
            glNormal3f(face.normal.x, face.normal.y, face.normal.z)
            for tri in face.triangles:
                for vertex in tri:
                    glVertex3f(vertex.x, vertex.y, vertex.z)
        glEnd()

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
            vertices = face.polygon(planes)
            if len(vertices) < 3:
                continue
            triangles = triangulate(vertices)
            if not triangles:
                continue
            faces.append(RenderFace(normal=face.plane.normal.normalize(), triangles=triangles))
    return faces


def preview_vmf(vmf: VMFMap) -> None:
    faces = build_render_faces(vmf)
    window = VMFViewerWindow(faces, width=1280, height=720)
    if faces:
        total = Vector3(0.0, 0.0, 0.0)
        count = 0
        for face in faces:
            for tri in face.triangles:
                for vertex in tri:
                    total = total + vertex
                    count += 1
        if count:
            window.camera.target = Vector3(total.x / count, total.y / count, total.z / count)
    pyglet.app.run()


def preview_file(path: str) -> None:
    from .parser import load_vmf

    vmf = load_vmf(path)
    preview_vmf(vmf)
