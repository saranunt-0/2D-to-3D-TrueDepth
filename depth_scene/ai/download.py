"""Model download with progress, safe for use from a background thread."""

import os
import ssl
import urllib.request


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


class Download:
    """Downloads ``url`` to ``dest`` via a ``.part`` file; poll ``progress``."""

    def __init__(self, url, dest):
        self.url = url
        self.dest = dest
        self.progress = 0.0
        self.done = False
        self.error = None
        self.cancel = False

    def run(self):
        part = self.dest + ".part"
        try:
            os.makedirs(os.path.dirname(self.dest), exist_ok=True)
            req = urllib.request.Request(self.url, headers={"User-Agent": "depth-scene-blender"})
            with urllib.request.urlopen(req, context=_ssl_context(), timeout=60) as resp, open(part, "wb") as fh:
                total = int(resp.headers.get("Content-Length") or 0)
                read = 0
                while True:
                    if self.cancel:
                        raise RuntimeError("cancelled")
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    read += len(chunk)
                    if total:
                        self.progress = read / total
            if total and read != total:
                raise RuntimeError(f"incomplete download ({read} of {total} bytes)")
            os.replace(part, self.dest)
            self.progress = 1.0
        except Exception as exc:  # reported to the UI
            self.error = str(exc)
            if os.path.exists(part):
                os.remove(part)
        finally:
            self.done = True
