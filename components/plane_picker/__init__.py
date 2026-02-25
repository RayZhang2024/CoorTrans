from __future__ import annotations

import os
from typing import Any, Dict

import streamlit.components.v1 as components

_COMPONENT_PATH = os.path.join(os.path.dirname(__file__), "frontend", "build")
_DECLARED = None


def plane_picker(
    *,
    p0: list[float],
    u: list[float],
    v: list[float],
    umin: float,
    umax: float,
    vmin: float,
    vmax: float,
    segments: list[list[list[float]]] | None = None,
    picks: list[list[float]] | None = None,
    picks_rev: int = 0,
    width: int = 420,
    height: int = 420,
    key: str | None = None,
) -> Dict[str, Any] | None:
    global _DECLARED
    if not os.path.isdir(_COMPONENT_PATH):
        return None
    if _DECLARED is None:
        _DECLARED = components.declare_component(
            "plane_picker",
            path=_COMPONENT_PATH,
        )
    return _DECLARED(
        p0=p0,
        u=u,
        v=v,
        umin=umin,
        umax=umax,
        vmin=vmin,
        vmax=vmax,
        segments=segments or [],
        picks=picks or [],
        picks_rev=picks_rev,
        width=width,
        height=height,
        key=key,
        default=None,
    )
