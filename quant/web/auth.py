"""HTTP Basic auth guard for the web control panel.

Behaviour:
 - If ``WEB_PASSWORD`` is unset, the app runs unauthenticated (useful for
   local development). A warning is logged on startup.
 - If ``WEB_PASSWORD`` is set, every request outside ``/healthz`` requires
   Basic auth. ``WEB_USERNAME`` defaults to ``admin``.
"""
from __future__ import annotations

import os
import secrets

from fastapi import HTTPException, Request, status


def _expected() -> tuple[str, str] | None:
    pw = os.getenv("WEB_PASSWORD", "").strip()
    if not pw:
        return None
    user = os.getenv("WEB_USERNAME", "admin").strip() or "admin"
    return user, pw


def check_auth(request: Request) -> None:
    """Dependency: raises 401 if credentials are missing or wrong."""
    expected = _expected()
    if expected is None:
        return  # auth disabled
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("basic "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
            headers={"WWW-Authenticate": 'Basic realm="botipman1000"'},
        )
    import base64

    try:
        raw = base64.b64decode(auth.split(" ", 1)[1]).decode("utf-8", "replace")
        user, _, pw = raw.partition(":")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid credentials encoding",
            headers={"WWW-Authenticate": 'Basic realm="botipman1000"'},
        )
    exp_user, exp_pw = expected
    ok_user = secrets.compare_digest(user, exp_user)
    ok_pw = secrets.compare_digest(pw, exp_pw)
    if not (ok_user and ok_pw):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="bad credentials",
            headers={"WWW-Authenticate": 'Basic realm="botipman1000"'},
        )
