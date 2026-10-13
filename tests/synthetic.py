"""Analytic test scenes with exact ground-truth depth (no Blender needed)."""

import math

import numpy as np


def corridor_depth(width, height, fov, half_w=1.0, half_h=1.2, length=12.0, pillar=None):
    """Planar depth of a box corridor seen from its centre line.

    Camera at origin looking down -Z (Blender convention). Walls at x = +-half_w,
    floor/ceiling at y = -+half_h, end wall at z = -length. ``pillar`` =
    (x0, x1, z_front) adds a camera-facing slab occluding part of the view.
    Returns (depth (H, W), plane_id (H, W)).
    """
    t = math.tan(fov / 2)
    long_side = max(width, height)
    hw, hh = t * width / long_side, t * height / long_side
    xs = ((np.arange(width) + 0.5) / width - 0.5) * 2 * hw
    ys = (0.5 - (np.arange(height) + 0.5) / height) * 2 * hh
    x, y = np.meshgrid(xs, ys, indexing="xy")
    big = np.float32(1e9)
    with np.errstate(divide="ignore", invalid="ignore"):
        cand = [
            np.where(x > 0, half_w / x, big),      # right wall
            np.where(x < 0, -half_w / x, big),     # left wall
            np.where(y < 0, -half_h / y, big),     # floor
            np.where(y > 0, half_h / y, big),      # ceiling
            np.full_like(x, length),               # end wall
        ]
        if pillar is not None:
            x0, x1, zf = pillar
            hit = (x * zf >= x0) & (x * zf <= x1)
            cand.append(np.where(hit, zf, big))
    stack = np.stack(cand)
    plane = np.argmin(stack, axis=0)
    depth = np.min(stack, axis=0).astype(np.float32)
    return depth, plane


def camera_points(depth, fov):
    """Ground-truth camera-space points (Blender convention) for a depth map."""
    h, w = depth.shape
    t = math.tan(fov / 2)
    long_side = max(w, h)
    hw, hh = t * w / long_side, t * h / long_side
    xs = ((np.arange(w) + 0.5) / w - 0.5) * 2 * hw
    ys = (0.5 - (np.arange(h) + 0.5) / h) * 2 * hh
    x, y = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([x * depth, y * depth, -depth], axis=-1)


def corridor_image(width, height, fov, pillar=(-0.9, -0.45, 3.2)):
    """Textured RGB corridor (walls, doors, planked floor) + its planar depth.

    Used as a licence-free stand-in for the user's corridor paintings, with
    exact ground truth to validate reconstruction.
    """
    depth, plane = corridor_depth(width, height, fov, pillar=pillar)
    p = camera_points(depth, fov)
    x, y, z = p[..., 0], p[..., 1], -p[..., 2]
    rgb = np.zeros((height, width, 3), np.float32)
    wall = np.array([0.33, 0.42, 0.62], np.float32)
    door = np.array([0.62, 0.70, 0.80], np.float32)
    floor_a = np.array([0.55, 0.30, 0.18], np.float32)
    floor_b = np.array([0.45, 0.24, 0.14], np.float32)

    is_wall = (plane == 0) | (plane == 1)
    door_mask = is_wall & (((z % 4.0) > 1.6) & ((z % 4.0) < 2.6)) & (y < 0.75)
    rgb[is_wall] = wall
    rgb[door_mask] = door
    frame = is_wall & (np.abs((z % 4.0) - 1.6) < 0.05) | (np.abs((z % 4.0) - 2.6) < 0.05) & is_wall
    rgb[frame] = wall * 0.5

    fl = plane == 2
    plank = (np.floor(x * 5.0) % 2 == 0)
    rgb[fl & plank] = floor_a
    rgb[fl & ~plank] = floor_b
    seam = fl & (np.abs((x * 5.0) - np.round(x * 5.0)) < 0.06)
    rgb[seam] = floor_b * 0.5

    rgb[plane == 3] = np.array([0.70, 0.76, 0.85], np.float32)
    end = plane == 4
    rgb[end] = np.array([0.85, 0.88, 0.92], np.float32)
    rgb[end & (np.abs(x) < 0.35) & (y < 0.7)] = np.array([0.40, 0.45, 0.55], np.float32)
    rgb[plane == 5] = np.array([0.60, 0.20, 0.18], np.float32)

    shade = 1.0 / (1.0 + 0.04 * z)
    return np.clip(rgb * shade[..., None], 0, 1), depth


def corridor_normals(plane):
    """Camera-space unit normals (x right, y up, z towards the camera) for ``corridor_depth``'s plane ids."""
    table = np.array([
        [-1, 0, 0],      # right wall faces left
        [1, 0, 0],       # left wall faces right
        [0, 1, 0],       # floor faces up
        [0, -1, 0],      # ceiling faces down
        [0, 0, 1],       # end wall faces the camera
        [0, 0, 1],       # pillar front faces the camera
    ], np.float32)
    return table[plane]
