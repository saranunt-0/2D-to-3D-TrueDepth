"""Build an installable Blender extension zip (Blender 4.2+).

    python tools/build_zip.py            -> dist/depth_scene-<version>.zip

Same layout as `blender --command extension build`: the manifest sits at the
zip root. Install via Edit > Preferences > Get Extensions > Install from Disk.
"""

import os
import re
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "depth_scene")


def main():
    manifest = open(os.path.join(SRC, "blender_manifest.toml"), encoding="utf-8").read()
    version = re.search(r'^version\s*=\s*"([^"]+)"', manifest, re.M).group(1)
    os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)
    out = os.path.join(ROOT, "dist", f"depth_scene-{version}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for base, dirs, files in os.walk(SRC):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in sorted(files):
                if name.endswith((".pyc", ".zip")):
                    continue
                path = os.path.join(base, name)
                zf.write(path, os.path.relpath(path, SRC))
    print(out)


if __name__ == "__main__":
    main()
