# SPDX-License-Identifier: GPL-3.0-or-later
"""Depth Scene - turn a single image into a 2.5D / 3D scene with AI depth.

Inspired by the TrueDepth add-on workflow; not affiliated with it.
"""

bl_info = {
    "name": "Depth Scene",
    "author": "saranunt-0",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport > Sidebar (N) > Depth Scene",
    "description": "AI depth estimation and image-to-3D scene reconstruction",
    "category": "3D View",
}

try:
    import bpy  # noqa: F401
except ImportError:  # imported by unit tests outside Blender
    bpy = None


def register():
    from . import operators, preferences, properties, ui

    preferences.register()
    properties.register()
    operators.register()
    ui.register()


def unregister():
    from . import operators, preferences, properties, ui

    ui.unregister()
    operators.unregister()
    properties.unregister()
    preferences.unregister()
