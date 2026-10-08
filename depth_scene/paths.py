"""Where the add-on keeps downloaded models and pip-installed packages."""

import os
import sys

import bpy

ADDON_ID = __package__


def data_dir():
    """Per-user writable directory, preserved across add-on updates."""
    try:
        return bpy.utils.extension_path_user(ADDON_ID, create=True)
    except (ValueError, AttributeError):
        # Installed as a legacy add-on (not an extension).
        return bpy.utils.user_resource("DATAFILES", path="depth_scene", create=True)


def prefs():
    addon = bpy.context.preferences.addons.get(ADDON_ID)
    return addon.preferences if addon else None


def models_dir():
    p = prefs()
    custom = bpy.path.abspath(p.models_dir) if p and p.models_dir else ""
    path = custom or os.path.join(data_dir(), "models")
    os.makedirs(path, exist_ok=True)
    return path


def site_packages_dir(variant):
    """pip --target directory, one per Python version and ORT variant."""
    tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    return os.path.join(data_dir(), f"site-packages-{tag}-{variant}")


def model_path(info, precision):
    return os.path.join(models_dir(), info.filename(precision))
