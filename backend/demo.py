"""Read-only hosted demo and static frontend (task 23a).

Demo mode (env ``ROADSCOUT_DEMO`` = 1/true/on/yes, read on every request):

- ``DemoGuardMiddleware``: every request under ``/api`` whose method is not GET, HEAD or OPTIONS is answered
  with ``403 {"detail": "This is a read-only demo."}`` before routing, so it fires for invalid bodies and
  unknown ids too (upload, process, delete video, export, PATCH review actions, mark reported). ``/health``
  and every GET stay open.
- ``GET /api/demo`` answers ``{"demo": true|false}`` (the frontend uses it to show a banner and hide controls).
- The GET routes were audited: none returns the raw upload or an unblurred image. The photo routes serve only
  privacy-blurred files and keep their fail-closed 404s, and export videos are blurred-only files. The
  live provider is never built and ``META_API_KEY`` never read in demo mode (see ``backend/config.py`` and
  ``backend/providers/factory.py``).

Static frontend (env ``ROADSCOUT_STATIC_DIR`` = the built ``frontend/dist``, read on every request):
``StaticFrontendMiddleware`` serves GET/HEAD requests that are not ``/api``, ``/health`` or the FastAPI docs:
a file of the directory when the path names one, otherwise ``index.html`` (the app uses hash routing). A path
that tries to leave the directory (``..`` segments, backslashes, NUL) gets 404. With the variable unset, or
without an ``index.html`` in the directory, nothing changes. Unknown ``/api`` paths still 404 as JSON.
"""
from __future__ import annotations

import mimetypes
from pathlib import Path

from starlette.responses import FileResponse, JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from backend import config

SAFE_METHODS = ("GET", "HEAD", "OPTIONS")
DEMO_MESSAGE = "This is a read-only demo."
_PASS_THROUGH = ("/api", "/health", "/docs", "/redoc", "/openapi.json")


def _is_under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


class DemoGuardMiddleware:
    """403 for every non-GET/HEAD/OPTIONS request under /api while demo mode is on."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] == "http"
            and scope["method"].upper() not in SAFE_METHODS
            and _is_under(scope.get("path", ""), "/api")
            and config.demo_mode()
        ):
            await JSONResponse({"detail": DEMO_MESSAGE}, status_code=403)(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _safe_file(root: Path, url_path: str) -> Path | None | bool:
    """The file under ``root`` that ``url_path`` names: a Path, None when there is no such file, False when the
    path tries to leave ``root``."""
    rel = url_path.lstrip("/")
    if "\x00" in rel or "\\" in rel or ":" in rel:
        return False
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        return False
    if not parts:
        return None
    try:
        candidate = (root / Path(*parts)).resolve()
        base = root.resolve()
    except (OSError, ValueError):
        return False
    if not candidate.is_relative_to(base):
        return False
    return candidate if candidate.is_file() else None


class StaticFrontendMiddleware:
    """Serve the built frontend at / when ROADSCOUT_STATIC_DIR is set."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"].upper() in ("GET", "HEAD"):
            root = config.static_dir()
            path = scope.get("path", "/")
            if root is not None and not any(_is_under(path, p) for p in _PASS_THROUGH):
                response = self._respond(root, path)
                if response is not None:
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)

    @staticmethod
    def _respond(root: Path, path: str) -> Response | None:
        index = root / "index.html"
        if not index.is_file():
            return None
        found = _safe_file(root, path)
        if found is False:
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        if found is None:
            # unknown path: the single-page app decides (hash routing); never cached
            return FileResponse(index, media_type="text/html", headers={"Cache-Control": "no-cache"})
        media_type = mimetypes.guess_type(found.name)[0] or "application/octet-stream"
        immutable = path.startswith("/assets/")
        headers = {"Cache-Control": "public, max-age=31536000, immutable" if immutable else "no-cache"}
        return FileResponse(found, media_type=media_type, headers=headers)
