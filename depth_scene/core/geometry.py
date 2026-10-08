"""Mesh generation from a colour image + depth map (no bpy).

Coordinate frame: everything is built in *projector-camera space*, which
matches a Blender camera object's local axes: +X right, +Y up, the camera
looks down -Z. The add-on parents the mesh and the projector camera to the
same rig empty, so the image re-projects exactly from the projector camera.

Two reconstruction modes mirror TrueDepth:

PERSPECTIVE  Each grid vertex is pushed along its camera ray to the mapped
             distance (true unprojection: a corridor becomes a corridor).
RELIEF       A flat plane displaced toward the camera by nearness (bas-relief).
"""

import math

import numpy as np

from . import depth as depth_mod
from . import imaging


def grid_shape(width_px, height_px, resolution):
    """Vertex counts (nx, ny); ``resolution`` = vertices along the longer side."""
    resolution = max(int(resolution), 2)
    if width_px >= height_px:
        nx = resolution
        ny = max(2, int(round(resolution * height_px / width_px)))
    else:
        ny = resolution
        nx = max(2, int(round(resolution * width_px / height_px)))
    return nx, ny


def grid_coords(nx, ny):
    """Normalised (u, v) per vertex, shape (ny, nx). v = 0 is the top row."""
    u = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    v = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    return np.meshgrid(u, v, indexing="xy")


def quad_faces(nx, ny):
    """Quad indices (F, 4), counter-clockwise seen from the camera (+Z normals)."""
    idx = np.arange(nx * ny, dtype=np.int32).reshape(ny, nx)
    tl = idx[:-1, :-1]
    tr = idx[:-1, 1:]
    bl = idx[1:, :-1]
    br = idx[1:, 1:]
    return np.stack([tl, bl, br, tr], axis=-1).reshape(-1, 4)


def image_plane_half_extent(fov, width_px, height_px):
    """Half width/height of the image plane at unit distance.

    ``fov`` is the angle across the *longer* image side, which is what
    Blender cameras use with sensor fit AUTO.
    """
    t = math.tan(fov * 0.5)
    long_side = float(max(width_px, height_px))
    return t * width_px / long_side, t * height_px / long_side


def fov_from_focal_px(focal_px, width_px, height_px):
    return 2.0 * math.atan(0.5 * max(width_px, height_px) / focal_px)


def unproject(u, v, z, fov, width_px, height_px):
    """Planar distance -> camera-space points (Blender camera convention)."""
    hw, hh = image_plane_half_extent(fov, width_px, height_px)
    x = (u - 0.5) * 2.0 * hw
    y = (0.5 - v) * 2.0 * hh
    return np.stack([x * z, y * z, -z], axis=-1).astype(np.float32)


def stretched_face_mask(z, fov, width_px, height_px, max_angle):
    """Flag quads whose surface is seen at more than ``max_angle`` from its ray.

    For a plane with normal n, d(log z)/d(image-plane coords) = n_xy / (n . r),
    so |grad log z| * |r| equals 1/cos(angle between ray and surface normal)
    exactly for walls/floors parallel to the optical axis, and is a lower
    bound otherwise. Occlusion edges produce huge log-depth jumps across one
    face and therefore near-90 degree apparent angles; real surfaces seen at a
    grazing angle (corridor walls) stay below the threshold except right at
    the vanishing point. Being expressed as an angle, the test does not depend
    on grid resolution.
    """
    ny, nx = z.shape
    lz = np.log(np.maximum(z, 1e-6))
    hw, hh = image_plane_half_extent(fov, width_px, height_px)
    du = 2.0 * hw / (nx - 1)
    dv = 2.0 * hh / (ny - 1)
    gx = np.maximum(np.abs(lz[:-1, 1:] - lz[:-1, :-1]), np.abs(lz[1:, 1:] - lz[1:, :-1])) / du
    gy = np.maximum(np.abs(lz[1:, :-1] - lz[:-1, :-1]), np.abs(lz[1:, 1:] - lz[:-1, 1:])) / dv
    slope = np.sqrt(gx * gx + gy * gy)
    uc = (np.arange(nx - 1, dtype=np.float32) + 0.5) / (nx - 1)
    vc = (np.arange(ny - 1, dtype=np.float32) + 0.5) / (ny - 1)
    xc = (uc[None, :] - 0.5) * 2.0 * hw
    yc = (0.5 - vc[:, None]) * 2.0 * hh
    ray_len = np.sqrt(1.0 + xc * xc + yc * yc)
    limit = 1.0 / max(math.cos(max_angle), 1e-6)
    return (slope * ray_len > limit).reshape(-1)


def compact(verts, faces, uv, keep_faces):
    """Drop removed faces and any vertex no longer referenced."""
    faces = faces[keep_faces]
    used = np.zeros(len(verts), dtype=bool)
    used[faces.ravel()] = True
    remap = np.full(len(verts), -1, dtype=np.int32)
    remap[used] = np.arange(int(used.sum()), dtype=np.int32)
    return verts[used], remap[faces], uv[used]


class MeshParams:
    """Plain container for every reconstruction setting (mirrors the UI)."""

    def __init__(self, **kw):
        self.mode = kw.pop("mode", "PERSPECTIVE")
        self.resolution = kw.pop("resolution", 256)
        # perspective
        self.fov = kw.pop("fov", math.radians(60.0))
        self.mapping = kw.pop("mapping", "INVERSE")
        self.near = kw.pop("near", 1.0)
        self.far = kw.pop("far", 10.0)
        self.metric_scale = kw.pop("metric_scale", 1.0)
        self.remove_stretched = kw.pop("remove_stretched", True)
        self.stretch_angle = kw.pop("stretch_angle", math.radians(86.0))
        # relief
        self.size = kw.pop("size", 2.0)
        self.base_displacement = kw.pop("base_displacement", 0.3)
        self.invert = kw.pop("invert", False)
        # shared
        self.smooth = kw.pop("smooth", 0.0)                 # % of longer side
        self.detail_strength = kw.pop("detail_strength", 0.0)
        self.detail_radius = kw.pop("detail_radius", 1.0)   # % of longer side
        self.use_alpha = kw.pop("use_alpha", False)
        self.remove_masked = kw.pop("remove_masked", False)
        if kw:
            raise TypeError(f"unknown parameters: {sorted(kw)}")


def build(params, depth_map, depth_kind, image_size, gray=None, alpha=None, valid=None):
    """Build mesh arrays.

    depth_map   (h, w) raw model output (any resolution).
    depth_kind  'disparity' | 'depth' | 'metric'.
    image_size  (width_px, height_px) of the colour image; sets aspect + FOV.
    gray        optional (H, W) luminance for detail displacement.
    alpha       optional (H, W) image alpha (faces below 0.5 are removed).
    valid       optional (h, w) bool, model validity mask (e.g. sky).

    Returns (verts (N, 3) float32, faces (F, 4) int32, uv (N, 2) float32, info).
    """
    width_px, height_px = image_size
    long_px = max(width_px, height_px)
    nx, ny = grid_shape(width_px, height_px, params.resolution)
    u, v = grid_coords(nx, ny)

    depth_map = np.asarray(depth_map, dtype=np.float32)
    dh, dw = depth_map.shape
    smooth_px = params.smooth / 100.0 * max(dh, dw)

    if params.mode == "PERSPECTIVE" and params.mapping == "METRIC":
        if depth_kind != "metric":
            raise ValueError("METRIC mapping needs a metric depth map")
        field = depth_mod.metric_distance(depth_map, params.metric_scale, valid)
        if smooth_px > 0:
            field = imaging.gaussian_blur(field, smooth_px)
        z = imaging.sample_bilinear(field, u, v)
        t = None
    else:
        t_map = depth_mod.nearness(depth_map, depth_kind, valid)
        if valid is not None:
            t_map[~valid] = 0.0   # invalid (sky) goes to the far plane
        if smooth_px > 0:
            t_map = imaging.gaussian_blur(t_map, smooth_px)
        t = imaging.sample_bilinear(t_map, u, v)
        z = None
        if params.mode == "PERSPECTIVE":
            z = depth_mod.map_to_distance(t, params.mapping, params.near, params.far)

    detail = None
    if params.detail_strength != 0.0 and gray is not None:
        gray = np.asarray(gray, dtype=np.float32)
        # Detail finer than the vertex grid would only alias: work at 2x grid size.
        gh, gw = gray.shape
        scale = min(1.0, 2.0 * nx / gw, 2.0 * ny / gh)
        if scale < 1.0:
            gray = imaging.resize(gray, max(2, round(gh * scale)), max(2, round(gw * scale)))
        hp = depth_mod.detail_signal(gray, params.detail_radius / 100.0 * max(gray.shape))
        detail = imaging.sample_bilinear(hp, u, v)

    faces = quad_faces(nx, ny)
    keep = np.ones(len(faces), dtype=bool)
    info = {"grid": (nx, ny), "stretched_removed": 0}

    if params.mode == "PERSPECTIVE":
        if params.remove_stretched:
            stretched = stretched_face_mask(z, params.fov, width_px, height_px, params.stretch_angle)
            info["stretched_removed"] = int(stretched.sum())
            keep &= ~stretched
        z_final = z if detail is None else z * (1.0 - params.detail_strength * detail)
        verts = unproject(u, v, np.maximum(z_final, 1e-4), params.fov, width_px, height_px)
    elif params.mode == "RELIEF":
        hw = params.size * 0.5 * width_px / long_px
        hh = params.size * 0.5 * height_px / long_px
        h = (1.0 - t) if params.invert else t
        disp = h * params.base_displacement
        if detail is not None:
            disp = disp + params.detail_strength * params.size * detail
        verts = np.stack([(u - 0.5) * 2 * hw, (0.5 - v) * 2 * hh, disp], axis=-1).astype(np.float32)
    else:
        raise ValueError(f"unknown mode {params.mode!r}")

    def face_any(per_vertex_bad):
        return per_vertex_bad.reshape(-1)[faces].any(axis=1)

    if params.use_alpha and alpha is not None:
        keep &= ~face_any(imaging.sample_bilinear(alpha, u, v) < 0.5)
    if params.remove_masked and valid is not None:
        keep &= ~face_any(~imaging.sample_nearest(valid, u, v))

    uv = np.stack([u, 1.0 - v], axis=-1).reshape(-1, 2).astype(np.float32)
    verts = verts.reshape(-1, 3)
    if not keep.all():
        verts, faces, uv = compact(verts, faces, uv, keep)
    info["faces"] = len(faces)
    info["verts"] = len(verts)
    if z is not None:
        info["z_range"] = (float(z.min()), float(z.max()))
    return verts, faces, uv, info
