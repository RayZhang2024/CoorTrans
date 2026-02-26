import base64
import io
import json
import os
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from stl import mesh

from components.plane_picker import plane_picker


POINT_COLUMNS = ["Label", "X", "Y", "Z"]
COPLANAR_THRESHOLD_MM = 1.0
RMS_WARN_MM = 0.1
MAX_WARN_MM = 0.2
STAGE_DIMS_MM = (500.0, 500.0, 100.0)  # width (X), length (Y), height (Z)
AXIS_SNAP_DIRECTIONS = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}
PLANE_PICKER_BUILD = os.path.join(
    os.path.dirname(__file__), "components", "plane_picker", "frontend", "build"
)


st.set_page_config(page_title="CoorTrans", layout="wide")
st.title("CoorTrans")


def empty_points_df() -> pd.DataFrame:
    return pd.DataFrame(columns=POINT_COLUMNS)


def init_state() -> None:
    defaults = {
        "feature_sample_df": empty_points_df(),
        "feature_instrument_df": empty_points_df(),
        "target_sample_df": empty_points_df(),
        "stl_bytes": None,
        "stl_name": None,
        "handedness_sample": "right",
        "handedness_instrument": "right",
        "axis_flip": "none",
        "scale_value": 1.0,
        "auto_scale": False,
        "model_color": "#4c78a8",
        "model_transparency": 0.7,
        "show_symbols": True,
        "symbol_size": 2,
        "symbol_color": "#e45756",
        "view_frame": "Sample",
        "axis_flip_suggest_msg": "",
        "axis_flip_suggest_level": "",
        "plane_mode": "Three Points",
        "plane_points": [(0.0, 0.0, 0.0)] * 3,
        "plane_point": (0.0, 0.0, 0.0),
        "plane_normal": (0.0, 0.0, 1.0),
        "plane_axis": "XY",
        "plane_offset": 0.0,
        "plane_enabled": False,
        "plane_view_normal": False,
        "plane_snap_to_curve": True,
        "click_add_to": "Target Points (Sample)",
        "last_plane_pick_id": None,
        "plane_pick_uv": [],
        "plane_signature": "",
        "plane_pick_rev": 0,
        "plane_show_3d_points_live": False,
        "plane_preview_uv": [],
        "last_plane_preview_id": None,
        "last_plane_commit_id": None,
        "apply_plane_camera_once": False,
        "viewer_camera_rev": 0,
        "viewer_camera_override": None,
        "viewer_camera_override_direction": None,
        "mount_fit_apply_status_level": "",
        "mount_fit_apply_status_msg": "",
        "instrument_pose_source": "Theodolite fit",
        "beam_show": True,
        "beam_size_y_mm": 4.0,
        "beam_size_z_mm": 4.0,
        "stage_show": True,
        "stage_center_gx": 0.0,
        "stage_center_gy": 0.0,
        "stage_center_gz": 0.0,
        "stage_tx": 0.0,
        "stage_ty": 0.0,
        "stage_tz": 0.0,
        "stage_rx": 0.0,
        "stage_ry": 0.0,
        "stage_rz": 0.0,
        "sample_mount_tx": 0.0,
        "sample_mount_ty": 0.0,
        "sample_mount_tz": 0.0,
        "sample_mount_rx": 0.0,
        "sample_mount_ry": 0.0,
        "sample_mount_rz": 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def _read_table(file_bytes: bytes, extension: str, header) -> pd.DataFrame:
    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(file_bytes), header=header)
    try:
        return pd.read_csv(
            io.BytesIO(file_bytes), header=header, sep=None, engine="python"
        )
    except Exception:
        return pd.read_csv(
            io.BytesIO(file_bytes), header=header, delim_whitespace=True
        )


def _extract_points_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or df.shape[1] < 3:
        return empty_points_df()
    cols = [str(col).strip().lower() for col in df.columns]
    label_col = None
    for candidate in ["label", "name", "id"]:
        if candidate in cols:
            label_col = df.columns[cols.index(candidate)]
            break
    if {"x", "y", "z"}.issubset(cols):
        x_col = df.columns[cols.index("x")]
        y_col = df.columns[cols.index("y")]
        z_col = df.columns[cols.index("z")]
    else:
        x_col, y_col, z_col = df.columns[:3]
        if label_col is None and df.shape[1] >= 4:
            label_col = df.columns[3]
    out = pd.DataFrame(
        {
            "Label": df[label_col] if label_col is not None else "",
            "X": pd.to_numeric(df[x_col], errors="coerce"),
            "Y": pd.to_numeric(df[y_col], errors="coerce"),
            "Z": pd.to_numeric(df[z_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["X", "Y", "Z"]).reset_index(drop=True)
    if out.empty:
        return empty_points_df()
    out["Label"] = out["Label"].fillna("").astype(str)
    return out


def load_points_file(uploaded) -> pd.DataFrame:
    file_bytes = uploaded.getvalue()
    extension = os.path.splitext(uploaded.name)[1].lower()
    df_with_header = _read_table(file_bytes, extension, header=0)
    points_with_header = _extract_points_df(df_with_header)
    names = [str(col).strip().lower() for col in df_with_header.columns]
    has_xyz_header = {"x", "y", "z"}.issubset(names)
    if has_xyz_header:
        return points_with_header
    df_no_header = _read_table(file_bytes, extension, header=None)
    points_no_header = _extract_points_df(df_no_header)
    if len(points_no_header) > len(points_with_header):
        return points_no_header
    return points_with_header


def ensure_point_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_points_df()
    df = df.copy()
    lower_cols = {col.lower(): col for col in df.columns}
    if "label" in lower_cols and "Label" not in df.columns:
        df["Label"] = df[lower_cols["label"]]
    if "x" in lower_cols and "X" not in df.columns:
        df["X"] = df[lower_cols["x"]]
    if "y" in lower_cols and "Y" not in df.columns:
        df["Y"] = df[lower_cols["y"]]
    if "z" in lower_cols and "Z" not in df.columns:
        df["Z"] = df[lower_cols["z"]]
    for col in POINT_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col == "Label" else np.nan
    df = df[POINT_COLUMNS]
    df["Label"] = df["Label"].fillna("").astype(str)
    return df


def clean_points_df(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_point_columns(df)
    if df.empty:
        return df
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df["Z"] = pd.to_numeric(df["Z"], errors="coerce")
    df = df.dropna(subset=["X", "Y", "Z"]).reset_index(drop=True)
    return df


def df_to_points(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    if df is None or df.empty:
        return np.zeros((0, 3)), []
    df = clean_points_df(df)
    if df.empty:
        return np.zeros((0, 3)), []
    points = df[["X", "Y", "Z"]].to_numpy(dtype=float)
    labels = df["Label"].fillna("").astype(str).tolist()
    return points, labels


def points_to_export_df(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_points_df(df)
    if df.empty:
        return df
    export_df = df.copy()
    export_df.insert(0, "id", range(1, len(export_df) + 1))
    export_df.columns = ["id", "label", "x", "y", "z"]
    return export_df


def export_points(df: pd.DataFrame, file_format: str):
    export_df = points_to_export_df(df)
    if file_format == "XLSX":
        buffer = io.BytesIO()
        export_df.to_excel(buffer, index=False)
        data = buffer.getvalue()
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ext = ".xlsx"
    elif file_format == "TXT":
        data = export_df.to_csv(index=False, sep="\t")
        mime = "text/plain"
        ext = ".txt"
    else:
        data = export_df.to_csv(index=False)
        mime = "text/csv"
        ext = ".csv"
    return data, mime, ext


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0:
        return vec
    return vec / norm


def rotation_matrix_xyz_deg(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    r_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    r_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    r_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return r_z @ r_y @ r_x


def pose_matrix_xyz(
    tx: float, ty: float, tz: float, rx_deg: float, ry_deg: float, rz_deg: float
) -> np.ndarray:
    mat = np.eye(4)
    mat[:3, :3] = rotation_matrix_xyz_deg(rx_deg, ry_deg, rz_deg)
    mat[:3, 3] = [tx, ty, tz]
    return mat


def stage_pose_matrix_from_state() -> np.ndarray:
    # Stage translations are entered along stage axes (local frame), not instrument axes.
    # Rotation occurs about the correlated stage-center point C_G that coincides with the
    # instrument beam/theodolite center (instrument origin) when Stage X/Y/Z = 0.
    # p_I = R * p_G + R * (d_G - C_G), where d_G is local-stage translation.
    tx_local = float(st.session_state["stage_tx"])
    ty_local = float(st.session_state["stage_ty"])
    tz_local = float(st.session_state["stage_tz"])
    center_g = np.array(
        [
            float(st.session_state["stage_center_gx"]),
            float(st.session_state["stage_center_gy"]),
            float(st.session_state["stage_center_gz"]),
        ],
        dtype=float,
    )
    rz_deg = float(st.session_state["stage_rz"])
    R = rotation_matrix_xyz_deg(0.0, 0.0, rz_deg)
    d_local = np.array([tx_local, ty_local, tz_local], dtype=float)
    t_instr = R @ (d_local - center_g)
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = t_instr
    return mat


def apply_pose_matrix(points: np.ndarray, mat: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    r = mat[:3, :3]
    t = mat[:3, 3]
    return (points @ r.T) + t


def transform_vectors_by_matrix(vectors: np.ndarray, mat: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    pts = vectors.reshape(-1, 3)
    pts_t = apply_pose_matrix(pts, mat)
    return pts_t.reshape(vectors.shape)


def transform_mesh_by_matrix(stl_mesh: mesh.Mesh, mat: np.ndarray) -> mesh.Mesh:
    new_mesh = mesh.Mesh(np.copy(stl_mesh.data))
    new_mesh.vectors = transform_vectors_by_matrix(new_mesh.vectors, mat)
    return new_mesh


def matrix_to_rigid_transform(mat: np.ndarray) -> Dict[str, np.ndarray]:
    r = mat[:3, :3].copy()
    t = mat[:3, 3].copy()
    return {
        "scale": 1.0,
        "rotation": r.copy(),
        "rotation_effective": r,
        "translation": t,
        "axis_flip": "none",
    }


def rotation_matrix_to_xyz_deg(rot: np.ndarray) -> Tuple[float, float, float]:
    # Project to the nearest proper rotation to reduce numerical noise.
    U, _, Vt = np.linalg.svd(rot)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    if np.linalg.det(R) < 0:
        raise ValueError("Rotation matrix is not a proper rotation.")

    r20 = float(np.clip(R[2, 0], -1.0, 1.0))
    ry = float(np.arcsin(-r20))
    cy = float(np.cos(ry))
    if abs(cy) > 1e-8:
        rx = float(np.arctan2(R[2, 1], R[2, 2]))
        rz = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        # Gimbal lock: choose rx=0 and solve rz from remaining terms.
        rx = 0.0
        if r20 <= -1.0 + 1e-8:
            ry = np.pi / 2.0
            rz = float(np.arctan2(-R[0, 1], R[1, 1]))
        else:
            ry = -np.pi / 2.0
            rz = float(np.arctan2(R[0, 1], R[1, 1]))
    return tuple(np.rad2deg([rx, ry, rz]))


def _set_mount_fit_status(level: str, msg: str) -> None:
    st.session_state["mount_fit_apply_status_level"] = level
    st.session_state["mount_fit_apply_status_msg"] = msg


def apply_theodolite_fit_to_stage_mount() -> None:
    P_feat, _ = df_to_points(st.session_state["feature_sample_df"])
    Q_feat, _ = df_to_points(st.session_state["feature_instrument_df"])
    if P_feat.shape[0] != Q_feat.shape[0] or not (4 <= P_feat.shape[0] <= 6):
        _set_mount_fit_status(
            "warning",
            "Need 4 to 6 matching feature points in sample and instrument tables.",
        )
        return

    try:
        transform_result = solve_transform(
            P_feat,
            Q_feat,
            scale_value=float(st.session_state["scale_value"]),
            auto_scale=bool(st.session_state["auto_scale"]),
            axis_flip=st.session_state["axis_flip"],
            enforce_proper=True,
        )
    except Exception as exc:
        _set_mount_fit_status("error", f"Failed to solve transform: {exc}")
        return

    scale = float(transform_result["scale"])
    if abs(scale - 1.0) > 1e-3:
        _set_mount_fit_status(
            "warning",
            f"Fit scale is {scale:.6f}. Apply to stage mount only when scale is ~1.0.",
        )
        return

    if str(transform_result.get("axis_flip", "none")) != "none":
        _set_mount_fit_status(
            "warning",
            "Axis flip is enabled. Reflections cannot be written into rigid stage mount angles.",
        )
        return

    R_is = np.array(transform_result["rotation_effective"], dtype=float)
    det_r = float(np.linalg.det(R_is))
    if det_r <= 0 or not np.isfinite(det_r):
        _set_mount_fit_status(
            "warning",
            "Solved transform is not a proper rigid rotation; cannot apply to stage mount.",
        )
        return

    T_I_S = np.eye(4)
    T_I_S[:3, :3] = R_is
    T_I_S[:3, 3] = np.array(transform_result["translation"], dtype=float)

    T_I_G = stage_pose_matrix_from_state()
    try:
        T_G_I = np.linalg.inv(T_I_G)
    except np.linalg.LinAlgError:
        _set_mount_fit_status("error", "Stage pose matrix is singular; cannot invert.")
        return

    T_G_S = T_G_I @ T_I_S
    t_gs = T_G_S[:3, 3]
    try:
        rx_deg, ry_deg, rz_deg = rotation_matrix_to_xyz_deg(T_G_S[:3, :3])
    except Exception as exc:
        _set_mount_fit_status("error", f"Failed to derive sample mount angles: {exc}")
        return

    st.session_state["sample_mount_tx"] = float(t_gs[0])
    st.session_state["sample_mount_ty"] = float(t_gs[1])
    st.session_state["sample_mount_tz"] = float(t_gs[2])
    st.session_state["sample_mount_rx"] = float(rx_deg)
    st.session_state["sample_mount_ry"] = float(ry_deg)
    st.session_state["sample_mount_rz"] = float(rz_deg)
    _set_mount_fit_status(
        "success",
        "Applied theodolite fit to sample mount pose (stage frame).",
    )


def cuboid_vectors(width: float, length: float, height: float) -> np.ndarray:
    # Stage local frame convention: top surface at z=0, body extends down to z=-height.
    x0, x1 = -width / 2.0, width / 2.0
    y0, y1 = -length / 2.0, length / 2.0
    z0, z1 = -height, 0.0
    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # bottom
            [4, 6, 5],
            [4, 7, 6],  # top
            [0, 4, 5],
            [0, 5, 1],  # side y0
            [1, 5, 6],
            [1, 6, 2],  # side x1
            [2, 6, 7],
            [2, 7, 3],  # side y1
            [3, 7, 4],
            [3, 4, 0],  # side x0
        ],
        dtype=int,
    )
    return vertices[faces]


def beam_cuboid_vectors(
    x_start: float, x_end: float, size_y: float, size_z: float
) -> np.ndarray:
    x0, x1 = sorted([float(x_start), float(x_end)])
    half_y = max(float(size_y), 0.1) / 2.0
    half_z = max(float(size_z), 0.1) / 2.0
    y0, y1 = -half_y, half_y
    z0, z1 = -half_z, half_z
    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=int,
    )
    return vertices[faces]


def viewer_axis_camera(direction: str) -> Dict:
    if direction == "Theodolite":
        diag_dir = normalize_vector(np.array([-1.0, -1.0, 0.0], dtype=float))
        distance = 2.1
        eye = diag_dir * distance
        return {
            "eye": {"x": float(eye[0]), "y": float(eye[1]), "z": float(eye[2])},
            "center": {"x": 0.0, "y": 0.0, "z": 0.0},
            "up": {"x": 0.0, "y": 0.0, "z": 1.0},
            "projection": {"type": "orthographic"},
        }
    distance = 1.8
    eye_map = {
        "+X": (distance, 0.0, 0.0),
        "-X": (-distance, 0.0, 0.0),
        "+Y": (0.0, distance, 0.0),
        "-Y": (0.0, -distance, 0.0),
        "+Z": (0.0, 0.0, distance),
        "-Z": (0.0, 0.0, -distance),
    }
    up_map = {
        "+X": (0.0, 0.0, 1.0),
        "-X": (0.0, 0.0, 1.0),
        "+Y": (0.0, 0.0, 1.0),
        "-Y": (0.0, 0.0, 1.0),
        "+Z": (0.0, 1.0, 0.0),
        "-Z": (0.0, 1.0, 0.0),
    }
    eye = eye_map.get(direction, (distance, distance, distance))
    up = up_map.get(direction, (0.0, 0.0, 1.0))
    return {
        "eye": {"x": eye[0], "y": eye[1], "z": eye[2]},
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "up": {"x": up[0], "y": up[1], "z": up[2]},
        "projection": {"type": "orthographic"},
    }


def set_viewer_axis_view(direction: str) -> None:
    st.session_state["viewer_camera_override"] = viewer_axis_camera(direction)
    st.session_state["viewer_camera_override_direction"] = direction
    st.session_state["viewer_camera_rev"] += 1


def plane_from_inputs() -> Tuple[np.ndarray, np.ndarray]:
    mode = st.session_state["plane_mode"]
    if mode == "Three Points":
        p1, p2, p3 = [np.array(p) for p in st.session_state["plane_points"]]
        n = np.cross(p2 - p1, p3 - p1)
        return p1, normalize_vector(n)
    if mode == "Point + Normal":
        p0 = np.array(st.session_state["plane_point"])
        n = np.array(st.session_state["plane_normal"])
        return p0, normalize_vector(n)
    axis = st.session_state["plane_axis"]
    offset = float(st.session_state["plane_offset"])
    if axis == "XY":
        return np.array([0.0, 0.0, offset]), np.array([0.0, 0.0, 1.0])
    if axis == "YZ":
        return np.array([offset, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    return np.array([0.0, offset, 0.0]), np.array([0.0, 1.0, 0.0])


def triangle_plane_intersection(
    tri: np.ndarray, p0: np.ndarray, n: np.ndarray, eps: float = 1e-6
) -> List[Tuple[np.ndarray, np.ndarray]]:
    d = (tri - p0) @ n
    if np.all(d > eps) or np.all(d < -eps):
        return []
    if np.all(np.abs(d) <= eps):
        return []
    points = []
    edges = [(0, 1), (1, 2), (2, 0)]
    for i, j in edges:
        d1, d2 = d[i], d[j]
        if abs(d1) <= eps and abs(d2) <= eps:
            continue
        if d1 * d2 < -eps**2:
            t = d1 / (d1 - d2)
            points.append(tri[i] + t * (tri[j] - tri[i]))
        elif abs(d1) <= eps:
            points.append(tri[i])
        elif abs(d2) <= eps:
            points.append(tri[j])
    if len(points) < 2:
        return []
    if len(points) > 2:
        unique = []
        for p in points:
            if not any(np.allclose(p, q, atol=1e-6) for q in unique):
                unique.append(p)
        points = unique
    if len(points) >= 2:
        return [(points[0], points[1])]
    return []


def mesh_plane_intersections(
    stl_mesh: mesh.Mesh, p0: np.ndarray, n: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray]]:
    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    for tri in stl_mesh.vectors:
        segments.extend(triangle_plane_intersection(tri, p0, n))
    return segments


def segments_to_trace_data(
    segments: List[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[List[float], List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for a, b in segments:
        xs.extend([a[0], b[0], None])
        ys.extend([a[1], b[1], None])
        zs.extend([a[2], b[2], None])
    return xs, ys, zs


def plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = normalize_vector(normal)
    if abs(n[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    u = normalize_vector(np.cross(n, ref))
    v = normalize_vector(np.cross(n, u))
    return u, v


def plane_uv_bounds(
    p0: np.ndarray, n: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    u, v = plane_basis(n)
    corners = np.array(
        [
            [bounds_min[0], bounds_min[1], bounds_min[2]],
            [bounds_min[0], bounds_min[1], bounds_max[2]],
            [bounds_min[0], bounds_max[1], bounds_min[2]],
            [bounds_min[0], bounds_max[1], bounds_max[2]],
            [bounds_max[0], bounds_min[1], bounds_min[2]],
            [bounds_max[0], bounds_min[1], bounds_max[2]],
            [bounds_max[0], bounds_max[1], bounds_min[2]],
            [bounds_max[0], bounds_max[1], bounds_max[2]],
        ]
    )
    rel = corners - p0
    us = rel @ u
    vs = rel @ v
    return u, v, float(us.min()), float(us.max()), float(vs.min()), float(vs.max())


def sample_segment_points(
    segments: List[Tuple[np.ndarray, np.ndarray]], max_points: int = 2000
) -> np.ndarray:
    if not segments:
        return np.zeros((0, 3))
    pts = []
    for a, b in segments:
        pts.append(a)
        pts.append(b)
        mid = (a + b) / 2.0
        pts.append(mid)
    pts_arr = np.array(pts)
    if pts_arr.shape[0] > max_points:
        idx = np.linspace(0, pts_arr.shape[0] - 1, max_points).astype(int)
        pts_arr = pts_arr[idx]
    return pts_arr


def nearest_point_on_segments(
    point: np.ndarray, segments: List[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, float]:
    best_point = None
    best_dist = None
    for a, b in segments:
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom == 0:
            proj = a
        else:
            t = float(np.dot(point - a, ab) / denom)
            t = max(0.0, min(1.0, t))
            proj = a + t * ab
        dist = float(np.linalg.norm(point - proj))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_point = proj
    if best_point is None:
        return point, float("inf")
    return best_point, best_dist


def project_segments_uv(
    segments: List[Tuple[np.ndarray, np.ndarray]],
    p0: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> List[List[List[float]]]:
    out: List[List[List[float]]] = []
    for a, b in segments:
        a_uv = [float((a - p0) @ u), float((a - p0) @ v)]
        b_uv = [float((b - p0) @ u), float((b - p0) @ v)]
        out.append([a_uv, b_uv])
    return out


def point_to_uv(point: np.ndarray, p0: np.ndarray, u: np.ndarray, v: np.ndarray) -> List[float]:
    rel = point - p0
    return [float(rel @ u), float(rel @ v)]

def is_coplanar(points: np.ndarray, threshold: float) -> bool:
    if points.shape[0] < 4:
        return True
    centroid = points.mean(axis=0)
    centered = points - centroid
    if np.allclose(centered, 0):
        return True
    _, _, v_t = np.linalg.svd(centered, full_matrices=False)
    normal = v_t[-1]
    distances = np.abs(centered @ normal)
    return float(np.max(distances)) <= threshold


def kabsch_rotation(P: np.ndarray, Q: np.ndarray, enforce_proper: bool) -> np.ndarray:
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if enforce_proper and np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R


def solve_transform(
    P: np.ndarray,
    Q: np.ndarray,
    scale_value: float,
    auto_scale: bool,
    axis_flip: str,
    enforce_proper: bool,
) -> Dict[str, np.ndarray]:
    flip_vector = np.array([1.0, 1.0, 1.0])
    if axis_flip == "X":
        flip_vector = np.array([-1.0, 1.0, 1.0])
    elif axis_flip == "Y":
        flip_vector = np.array([1.0, -1.0, 1.0])
    elif axis_flip == "Z":
        flip_vector = np.array([1.0, 1.0, -1.0])
    P_adj = P * flip_vector
    Pc = P_adj.mean(axis=0)
    Qc = Q.mean(axis=0)
    P0 = P_adj - Pc
    Q0 = Q - Qc
    R = kabsch_rotation(P0, Q0, enforce_proper=enforce_proper)
    if auto_scale:
        denom = np.sum(P0 ** 2)
        if denom <= 0:
            s = scale_value
        else:
            s = float(np.sum(Q0 * (P0 @ R.T)) / denom)
    else:
        s = scale_value
    t = Qc - s * (R @ Pc)
    R_eff = R.copy()
    R_eff[:, 0] *= flip_vector[0]
    R_eff[:, 1] *= flip_vector[1]
    R_eff[:, 2] *= flip_vector[2]
    return {
        "scale": s,
        "rotation": R,
        "rotation_effective": R_eff,
        "translation": t,
        "axis_flip": axis_flip,
    }


def apply_transform(points: np.ndarray, transform: Dict[str, np.ndarray]) -> np.ndarray:
    if points.size == 0:
        return points
    R_eff = transform["rotation_effective"]
    s = transform["scale"]
    t = transform["translation"]
    return (points @ (s * R_eff).T) + t


def compute_errors(
    P: np.ndarray, Q: np.ndarray, transform: Dict[str, np.ndarray]
) -> Tuple[float, float, np.ndarray]:
    pred = apply_transform(P, transform)
    residuals = Q - pred
    norms = np.linalg.norm(residuals, axis=1)
    rms = float(np.sqrt(np.mean(norms ** 2))) if norms.size else 0.0
    max_err = float(np.max(norms)) if norms.size else 0.0
    return rms, max_err, residuals


def build_matrix(transform: Dict[str, np.ndarray]) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = transform["scale"] * transform["rotation_effective"]
    M[:3, 3] = transform["translation"]
    return M


def matrix_to_csv(matrix: np.ndarray) -> str:
    df = pd.DataFrame(matrix)
    df.columns = [f"m0{i}" for i in range(4)]
    df.index = [f"r{i}" for i in range(4)]
    return df.to_csv(index=False)


def mesh_from_bytes(stl_bytes: bytes) -> mesh.Mesh:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_file:
        tmp_file.write(stl_bytes)
        tmp_path = tmp_file.name
    try:
        return mesh.Mesh.from_file(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def transform_mesh(stl_mesh: mesh.Mesh, transform: Dict[str, np.ndarray]) -> mesh.Mesh:
    new_mesh = mesh.Mesh(np.copy(stl_mesh.data))
    pts = new_mesh.vectors.reshape(-1, 3)
    pts_transformed = apply_transform(pts, transform)
    new_mesh.vectors = pts_transformed.reshape(new_mesh.vectors.shape)
    return new_mesh


def format_labels(labels: List[str], points: np.ndarray) -> List[str]:
    out = []
    for idx, label in enumerate(labels):
        x, y, z = points[idx]
        if label:
            out.append(f"{label} ({x:.2f}, {y:.2f}, {z:.2f})")
        else:
            out.append(f"({x:.2f}, {y:.2f}, {z:.2f})")
    return out


def add_points_trace(
    fig: go.Figure,
    points: np.ndarray,
    labels: List[str],
    name: str,
    color: str,
    size: int,
    show_labels: bool,
) -> None:
    if points.size == 0:
        return
    text = format_labels(labels, points) if show_labels else None
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers+text" if show_labels else "markers",
            text=text,
            textposition="top center",
            marker=dict(size=size, color=color, symbol="cross"),
            name=name,
        )
    )


def add_mesh_trace(
    fig: go.Figure,
    vectors: np.ndarray,
    *,
    color: str,
    opacity: float,
    name: str,
    showlegend: bool = True,
) -> None:
    if vectors is None or vectors.size == 0:
        return
    x = vectors[:, :, 0].ravel()
    y = vectors[:, :, 1].ravel()
    z = vectors[:, :, 2].ravel()
    i = np.arange(0, len(x), 3)
    j = i + 1
    k = i + 2
    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            color=color,
            opacity=opacity,
            flatshading=True,
            name=name,
            showlegend=showlegend,
        )
    )


def render_scene(
    stl_mesh: mesh.Mesh,
    *,
    opacity: float,
    color: str,
    point_traces: List[Tuple[np.ndarray, List[str], str, str]],
    show_labels: bool,
    symbol_size: int,
    intersection_segments: List[Tuple[np.ndarray, np.ndarray]] | None = None,
    camera: Dict | None = None,
    extra_mesh_traces: List[Dict] | None = None,
) -> None:
    vectors = stl_mesh.vectors
    fig = go.Figure()
    add_mesh_trace(fig, vectors, color=color, opacity=opacity, name="Model")
    for extra in extra_mesh_traces or []:
        add_mesh_trace(
            fig,
            extra.get("vectors"),
            color=extra.get("color", "#999999"),
            opacity=float(extra.get("opacity", 0.35)),
            name=str(extra.get("name", "Mesh")),
            showlegend=bool(extra.get("showlegend", True)),
        )

    # Draw a small XYZ triad inside the scene to show coordinate directions.
    mesh_points = vectors.reshape(-1, 3)
    bounds_min = mesh_points.min(axis=0).astype(float)
    bounds_max = mesh_points.max(axis=0).astype(float)
    span = bounds_max - bounds_min
    diag = float(np.linalg.norm(span)) or 1.0
    span_safe = np.where(span > 0, span, diag * 0.2)
    triad_origin = bounds_min + 0.08 * span_safe
    triad_len = max(diag * 0.12, 1.0)
    triad_axes = [
        ("X", np.array([1.0, 0.0, 0.0]), "#d62728"),
        ("Y", np.array([0.0, 1.0, 0.0]), "#2ca02c"),
        ("Z", np.array([0.0, 0.0, 1.0]), "#1f77b4"),
    ]
    for axis_label, axis_dir, axis_color in triad_axes:
        tip = triad_origin + axis_dir * triad_len
        fig.add_trace(
            go.Scatter3d(
                x=[triad_origin[0], tip[0]],
                y=[triad_origin[1], tip[1]],
                z=[triad_origin[2], tip[2]],
                mode="lines+text",
                text=["", axis_label],
                textposition="top center",
                line=dict(color=axis_color, width=6),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Cone(
                x=[tip[0]],
                y=[tip[1]],
                z=[tip[2]],
                u=[axis_dir[0]],
                v=[axis_dir[1]],
                w=[axis_dir[2]],
                anchor="tip",
                sizemode="absolute",
                sizeref=triad_len * 0.22,
                showscale=False,
                colorscale=[[0, axis_color], [1, axis_color]],
                showlegend=False,
                hoverinfo="skip",
            )
        )
    if intersection_segments:
        xs, ys, zs = segments_to_trace_data(intersection_segments)
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color="#ff7f0e", width=4),
                name="Plane Intersection",
            )
        )
    for points, labels, name, trace_color in point_traces:
        add_points_trace(
            fig,
            points,
            labels,
            name=name,
            color=trace_color,
            size=symbol_size,
            show_labels=show_labels,
        )
    uirev = f"coortrans-{st.session_state.get('viewer_camera_rev', 0)}"
    snap_camera = st.session_state.get("viewer_camera_override")
    snap_direction = st.session_state.get("viewer_camera_override_direction")
    is_axis_snap_view = (
        camera is not None
        and snap_direction is not None
        and isinstance(snap_camera, dict)
        and camera == snap_camera
        and snap_direction in AXIS_SNAP_DIRECTIONS
    )
    is_theodolite_view = (
        camera is not None
        and snap_direction == "Theodolite"
        and isinstance(snap_camera, dict)
        and camera == snap_camera
    )
    if is_theodolite_view:
        reticle_len = max(diag * 0.035, 1.0)
        reticle_radius = max(diag * 0.012, 0.4)
        h_dir = normalize_vector(np.array([1.0, -1.0, 0.0], dtype=float))
        v_dir = np.array([0.0, 0.0, 1.0], dtype=float)
        origin = np.array([0.0, 0.0, 0.0], dtype=float)
        h0 = origin - h_dir * reticle_len
        h1 = origin + h_dir * reticle_len
        v0 = origin - v_dir * reticle_len
        v1 = origin + v_dir * reticle_len
        fig.add_trace(
            go.Scatter3d(
                x=[h0[0], h1[0], None, v0[0], v1[0]],
                y=[h0[1], h1[1], None, v0[1], v1[1]],
                z=[h0[2], h1[2], None, v0[2], v1[2]],
                mode="lines",
                line=dict(color="rgba(20,20,20,0.9)", width=3),
                name="Theo Crosshair",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        theta = np.linspace(0.0, 2.0 * np.pi, 49)
        circle_pts = origin + np.outer(np.cos(theta), h_dir) * reticle_radius + np.outer(
            np.sin(theta), v_dir
        ) * reticle_radius
        fig.add_trace(
            go.Scatter3d(
                x=circle_pts[:, 0],
                y=circle_pts[:, 1],
                z=circle_pts[:, 2],
                mode="lines",
                line=dict(color="rgba(20,20,20,0.9)", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    axis_style = dict(
        showbackground=False,
        showline=True,
        linewidth=2,
        linecolor="rgba(40, 40, 40, 0.65)",
        gridcolor="rgba(120, 120, 120, 0.18)",
        zeroline=True,
        zerolinecolor="rgba(40, 40, 40, 0.45)",
        zerolinewidth=1,
        ticks="outside",
        showspikes=False,
    )
    axis_title_x = "X (mm)"
    axis_title_y = "Y (mm)"
    axis_title_z = "Z (mm)"
    hidden_tick_axis = None
    if is_axis_snap_view and snap_direction:
        hidden_tick_axis = snap_direction[-1].upper()
    tick_font_size = 12 if is_axis_snap_view else 14
    axis_tick_style = dict(
        tickfont=dict(size=tick_font_size),
        nticks=5 if is_axis_snap_view else 8,
    )
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            uirevision=uirev,
            camera=camera,
            xaxis=dict(
                title=axis_title_x,
                showticklabels=hidden_tick_axis != "X",
                **axis_style,
                **axis_tick_style,
            ),
            yaxis=dict(
                title=axis_title_y,
                showticklabels=hidden_tick_axis != "Y",
                **axis_style,
                **axis_tick_style,
            ),
            zaxis=dict(
                title=axis_title_z,
                showticklabels=hidden_tick_axis != "Z",
                **axis_style,
                **axis_tick_style,
            ),
        ),
        margin=dict(l=18, r=14, t=36, b=18),
        height=500,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.78)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
        ),
        uirevision=uirev,
    )
    viewer_container = st.container(border=True)
    viewer_container.plotly_chart(fig, use_container_width=True, key="stl_viewer")


def points_editor(
    title: str,
    key: str,
    help_text: str,
    import_key: str,
    export_key: str,
) -> None:
    st.subheader(title)
    st.caption(help_text)

    df = st.session_state[key]
    display_df = df.copy()
    if display_df.empty:
        display_df = pd.DataFrame(columns=POINT_COLUMNS)
    display_df.insert(0, "#", range(1, len(display_df) + 1))

    edited_df = st.data_editor(
        display_df,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "#": st.column_config.NumberColumn("#", disabled=True, width="small"),
            "Label": st.column_config.TextColumn("Label"),
            "X": st.column_config.NumberColumn("X", format="%.3f"),
            "Y": st.column_config.NumberColumn("Y", format="%.3f"),
            "Z": st.column_config.NumberColumn("Z", format="%.3f"),
        },
        key=f"{key}_editor",
    )

    edited_df = edited_df.drop(columns=["#"], errors="ignore")
    st.session_state[key] = ensure_point_columns(edited_df)

    with st.expander("Import / Export", expanded=False):
        coords_file = st.file_uploader(
            "Choose a coordinates file", type=["csv", "txt", "xlsx"], key=import_key
        )
        replace_coords = st.checkbox(
            "Replace existing coordinates", value=False, key=f"{import_key}_replace"
        )
        load_coords = st.button(
            "Load coordinates", disabled=coords_file is None, key=f"{import_key}_load"
        )
        if load_coords and coords_file is not None:
            try:
                loaded_df = load_points_file(coords_file)
                if loaded_df.empty:
                    st.warning("No valid coordinates found in the file.")
                else:
                    if replace_coords:
                        st.session_state[key] = loaded_df
                    else:
                        st.session_state[key] = pd.concat(
                            [st.session_state[key], loaded_df], ignore_index=True
                        )
                    st.success(f"Loaded {len(loaded_df)} coordinates.")
            except Exception as exc:
                st.error(f"Failed to load coordinates: {exc}")

        export_df = st.session_state[key]
        if not export_df.empty:
            file_format = st.selectbox(
                "Save format", ["CSV", "TXT", "XLSX"], key=f"{export_key}_format"
            )
            file_name = st.text_input(
                "Filename", value=key, key=f"{export_key}_name"
            )
            base_name = file_name.strip() or key
            if os.path.splitext(base_name)[1].lower() in {".csv", ".txt", ".xlsx"}:
                base_name = os.path.splitext(base_name)[0]
            data, mime, ext = export_points(export_df, file_format)
            st.download_button(
                "Save coordinates",
                data=data,
                file_name=f"{base_name}{ext}",
                mime=mime,
                key=f"{export_key}_download",
            )

def build_project_payload(transform: Dict[str, np.ndarray] | None, errors=None) -> Dict:
    stl_block = None
    if st.session_state["stl_bytes"]:
        stl_block = {
            "filename": st.session_state.get("stl_name") or "model.stl",
            "encoding": "base64",
            "data": base64.b64encode(st.session_state["stl_bytes"]).decode("ascii"),
        }
    payload = {
        "version": "1.0",
        "units": "mm",
        "stl_embedded": stl_block,
        "handedness": {
            "sample": st.session_state["handedness_sample"],
            "instrument": st.session_state["handedness_instrument"],
            "axis_flip": st.session_state["axis_flip"],
        },
        "points": {
            "feature_sample": points_to_export_df(
                st.session_state["feature_sample_df"]
            ).to_dict(orient="records"),
            "feature_instrument": points_to_export_df(
                st.session_state["feature_instrument_df"]
            ).to_dict(orient="records"),
            "target_sample": points_to_export_df(
                st.session_state["target_sample_df"]
            ).to_dict(orient="records"),
        },
        "transform": None,
        "settings": {
            "model_color": st.session_state["model_color"],
            "model_opacity": 1.0 - st.session_state["model_transparency"],
            "symbol_size": st.session_state["symbol_size"],
            "symbol_color": st.session_state["symbol_color"],
            "beam_show": st.session_state["beam_show"],
            "beam_size_y_mm": st.session_state["beam_size_y_mm"],
            "beam_size_z_mm": st.session_state["beam_size_z_mm"],
        },
    }
    if transform is not None:
        payload["transform"] = {
            "scale": float(transform["scale"]),
            "rotation": transform["rotation"].tolist(),
            "translation": transform["translation"].tolist(),
            "matrix_4x4": build_matrix(transform).tolist(),
            "fit_error": errors or {},
        }
    return payload


def load_project(payload: Dict) -> None:
    stl_block = payload.get("stl_embedded") or None
    if stl_block and stl_block.get("encoding") == "base64":
        st.session_state["stl_name"] = stl_block.get("filename", "model.stl")
        st.session_state["stl_bytes"] = base64.b64decode(stl_block.get("data", ""))
    else:
        st.session_state["stl_name"] = None
        st.session_state["stl_bytes"] = None

    handed = payload.get("handedness", {})
    st.session_state["handedness_sample"] = handed.get("sample", "right")
    st.session_state["handedness_instrument"] = handed.get("instrument", "right")
    st.session_state["axis_flip"] = handed.get("axis_flip", "none")

    points = payload.get("points", {})
    st.session_state["feature_sample_df"] = ensure_point_columns(
        pd.DataFrame(points.get("feature_sample", []))
    )
    st.session_state["feature_instrument_df"] = ensure_point_columns(
        pd.DataFrame(points.get("feature_instrument", []))
    )
    st.session_state["target_sample_df"] = ensure_point_columns(
        pd.DataFrame(points.get("target_sample", []))
    )

    transform = payload.get("transform") or {}
    st.session_state["scale_value"] = float(transform.get("scale", 1.0))

    settings = payload.get("settings", {})
    st.session_state["model_color"] = settings.get("model_color", "#4c78a8")
    model_opacity = settings.get("model_opacity", 1.0)
    st.session_state["model_transparency"] = 1.0 - float(model_opacity)
    st.session_state["symbol_size"] = int(settings.get("symbol_size", 2))
    st.session_state["symbol_color"] = settings.get("symbol_color", "#e45756")
    st.session_state["beam_show"] = bool(settings.get("beam_show", True))
    st.session_state["beam_size_y_mm"] = float(settings.get("beam_size_y_mm", 4.0))
    st.session_state["beam_size_z_mm"] = float(settings.get("beam_size_z_mm", 4.0))


project_download_slot = None
with st.sidebar.expander("Project", expanded=False):
    project_file = st.file_uploader("Load project", type=["json"], key="project_file")
    load_project_btn = st.button("Load project", disabled=project_file is None)
    if load_project_btn and project_file is not None:
        try:
            payload = json.loads(project_file.getvalue().decode("utf-8"))
            load_project(payload)
            st.success("Project loaded.")
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to load project: {exc}")

    project_download_slot = st.empty()

uploaded_file = st.sidebar.file_uploader("Choose an STL file", type=["stl"])
if uploaded_file is not None:
    st.session_state["stl_bytes"] = uploaded_file.getvalue()
    st.session_state["stl_name"] = uploaded_file.name

st.sidebar.caption("Upload a binary or ASCII STL file to preview the model.")

st.sidebar.slider(
    "Model transparency (0=opaque, 1=transparent)",
    0.0,
    1.0,
    st.session_state["model_transparency"],
    0.05,
    key="model_transparency",
)
st.sidebar.color_picker(
    "Model color", st.session_state["model_color"], key="model_color"
)

st.sidebar.checkbox(
    "Show point labels", value=st.session_state["show_symbols"], key="show_symbols"
)
st.sidebar.slider(
    "Symbol size", 2, 20, st.session_state["symbol_size"], key="symbol_size"
)
st.sidebar.color_picker(
    "Symbol color", st.session_state["symbol_color"], key="symbol_color"
)

st.sidebar.radio(
    "View frame",
    ["Sample", "Stage", "Instrument"],
    index=["Sample", "Stage", "Instrument"].index(
        st.session_state["view_frame"]
        if st.session_state["view_frame"] in {"Sample", "Stage", "Instrument"}
        else "Sample"
    ),
    key="view_frame",
)
with st.sidebar.expander("Stage Placement", expanded=False):
    st.caption("Built-in stage: 500 x 500 x 100 mm cuboid (top surface at stage z=0).")
    st.checkbox("Show stage in Instrument view", value=True, key="stage_show")
    st.caption(
        "Correlated stage-center point in stage frame (this point coincides with the "
        "instrument beam/theodolite center at Stage X/Y/Z = 0)."
    )
    st.number_input("Stage Center X (stage mm)", step=1.0, key="stage_center_gx")
    st.number_input("Stage Center Y (stage mm)", step=1.0, key="stage_center_gy")
    st.number_input("Stage Center Z (stage mm)", step=1.0, key="stage_center_gz")
    st.caption(
        "Stage pose in instrument frame (Stage X/Y/Z are motions along stage local axes; "
        "rotation is about the correlated stage center)."
    )
    st.number_input("Stage X (mm)", step=1.0, key="stage_tx")
    st.number_input("Stage Y (mm)", step=1.0, key="stage_ty")
    st.number_input("Stage Z (mm)", step=1.0, key="stage_tz")
    st.caption("Stage rotation about instrument Z")
    st.number_input("Stage Rz (deg)", step=1.0, key="stage_rz")
    st.caption("Sample mount pose in stage frame")
    mount_pos_cols = st.columns(3)
    with mount_pos_cols[0]:
        st.number_input("Sample X (mm)", step=1.0, key="sample_mount_tx")
    with mount_pos_cols[1]:
        st.number_input("Sample Y (mm)", step=1.0, key="sample_mount_ty")
    with mount_pos_cols[2]:
        st.number_input("Sample Z (mm)", step=1.0, key="sample_mount_tz")
    mount_rot_cols = st.columns(3)
    with mount_rot_cols[0]:
        st.number_input("Sample Rx (deg)", step=1.0, key="sample_mount_rx")
    with mount_rot_cols[1]:
        st.number_input("Sample Ry (deg)", step=1.0, key="sample_mount_ry")
    with mount_rot_cols[2]:
        st.number_input("Sample Rz (deg)", step=1.0, key="sample_mount_rz")
    st.selectbox(
        "Instrument sample pose source",
        ["Stage preview", "Theodolite fit", "Both"],
        index=["Stage preview", "Theodolite fit", "Both"].index(
            st.session_state.get("instrument_pose_source", "Theodolite fit")
            if st.session_state.get("instrument_pose_source")
            in {"Stage preview", "Theodolite fit", "Both"}
            else "Theodolite fit"
        ),
        key="instrument_pose_source",
    )
with st.sidebar.expander("Beam (Instrument)", expanded=False):
    st.caption("Beam travels along +X toward the instrument origin (0, 0, 0).")
    st.checkbox("Show beam in Instrument view", value=True, key="beam_show")
    beam_cols = st.columns(2)
    with beam_cols[0]:
        st.number_input(
            "Beam size Y (mm)",
            min_value=0.1,
            step=0.1,
            format="%.1f",
            key="beam_size_y_mm",
        )
    with beam_cols[1]:
        st.number_input(
            "Beam size Z (mm)",
            min_value=0.1,
            step=0.1,
            format="%.1f",
            key="beam_size_z_mm",
        )
with st.sidebar.expander("Transform", expanded=True):
    st.number_input(
        "Scale (unitless)",
        value=float(st.session_state["scale_value"]),
        step=0.0001,
        format="%.6f",
        key="scale_value",
    )
    st.checkbox(
        "Auto-estimate scale", value=st.session_state["auto_scale"], key="auto_scale"
    )

    st.selectbox(
        "Sample handedness",
        ["right", "left"],
        index=0 if st.session_state["handedness_sample"] == "right" else 1,
        key="handedness_sample",
    )
    st.selectbox(
        "Instrument handedness",
        ["right", "left"],
        index=0 if st.session_state["handedness_instrument"] == "right" else 1,
        key="handedness_instrument",
    )

    axis_flip_needed = (
        st.session_state["handedness_sample"]
        != st.session_state["handedness_instrument"]
    )
    if axis_flip_needed:
        st.info("Handedness differs. Choose an axis flip.")
        st.selectbox(
            "Axis flip",
            ["X", "Y", "Z"],
            index=["X", "Y", "Z"].index(
                st.session_state["axis_flip"]
                if st.session_state["axis_flip"] in {"X", "Y", "Z"}
                else "X"
            ),
            key="axis_flip",
        )
        def suggest_axis_flip() -> None:
            P, _ = df_to_points(st.session_state["feature_sample_df"])
            Q, _ = df_to_points(st.session_state["feature_instrument_df"])
            best_axis = None
            best_rms = None
            for axis in ["X", "Y", "Z"]:
                if P.shape[0] == 0 or Q.shape[0] == 0:
                    continue
                if P.shape[0] != Q.shape[0]:
                    continue
                if P.shape[0] < 4:
                    continue
                result = solve_transform(
                    P,
                    Q,
                    scale_value=st.session_state["scale_value"],
                    auto_scale=st.session_state["auto_scale"],
                    axis_flip=axis,
                    enforce_proper=True,
                )
                rms, _, _ = compute_errors(P, Q, result)
                if best_rms is None or rms < best_rms:
                    best_rms = rms
                    best_axis = axis
            if best_axis:
                st.session_state["axis_flip"] = best_axis
                st.session_state["axis_flip_suggest_level"] = "success"
                st.session_state[
                    "axis_flip_suggest_msg"
                ] = f"Auto-suggested axis flip: {best_axis}"
            else:
                st.session_state["axis_flip_suggest_level"] = "warning"
                st.session_state[
                    "axis_flip_suggest_msg"
                ] = "Not enough points to auto-suggest an axis flip."

        st.button("Auto-suggest flip", on_click=suggest_axis_flip)
        if st.session_state["axis_flip_suggest_msg"]:
            if st.session_state["axis_flip_suggest_level"] == "success":
                st.success(st.session_state["axis_flip_suggest_msg"])
            else:
                st.warning(st.session_state["axis_flip_suggest_msg"])
    else:
        st.session_state["axis_flip"] = "none"
        st.session_state["axis_flip_suggest_msg"] = ""
        st.session_state["axis_flip_suggest_level"] = ""
        st.caption("Handedness matches; no axis flip required.")

with st.sidebar.expander("Plane Selection", expanded=False):
    prev_plane_view_normal = st.session_state["plane_view_normal"]
    st.checkbox(
        "Enable plane selection",
        value=st.session_state["plane_enabled"],
        key="plane_enabled",
    )
    st.selectbox(
        "Plane definition",
        ["Three Points", "Point + Normal", "Point + Axis"],
        index=["Three Points", "Point + Normal", "Point + Axis"].index(
            st.session_state["plane_mode"]
        ),
        key="plane_mode",
    )
    st.checkbox(
        "View along plane normal",
        value=st.session_state["plane_view_normal"],
        key="plane_view_normal",
    )
    if st.session_state["plane_view_normal"] != prev_plane_view_normal:
        st.session_state["apply_plane_camera_once"] = True
        st.session_state["viewer_camera_rev"] += 1
    st.checkbox(
        "Snap clicks to intersection curve",
        value=st.session_state["plane_snap_to_curve"],
        key="plane_snap_to_curve",
    )
    st.checkbox(
        "Show picked points in 3D while picking (slower)",
        value=st.session_state["plane_show_3d_points_live"],
        key="plane_show_3d_points_live",
    )
    st.selectbox(
        "Add clicked points to",
        ["Target Points (Sample)", "Feature Points (Sample)"],
        index=0
        if st.session_state["click_add_to"] == "Target Points (Sample)"
        else 1,
        key="click_add_to",
    )
    if st.button("Clear Plane Markers"):
        st.session_state["plane_pick_uv"] = []
        st.session_state["plane_preview_uv"] = []
        st.session_state["last_plane_preview_id"] = None
        st.session_state["plane_pick_rev"] += 1

    if st.session_state["plane_mode"] == "Three Points":
        pts = list(st.session_state["plane_points"])
        p1 = st.text_input("P1 (x,y,z)", value=",".join(map(str, pts[0])))
        p2 = st.text_input("P2 (x,y,z)", value=",".join(map(str, pts[1])))
        p3 = st.text_input("P3 (x,y,z)", value=",".join(map(str, pts[2])))
        def parse_point(text: str) -> Tuple[float, float, float]:
            parts = [float(x.strip()) for x in text.split(",")]
            if len(parts) != 3:
                raise ValueError("Need three values")
            return parts[0], parts[1], parts[2]
        try:
            st.session_state["plane_points"] = [
                parse_point(p1),
                parse_point(p2),
                parse_point(p3),
            ]
        except Exception:
            st.warning("Enter points as x,y,z.")
    elif st.session_state["plane_mode"] == "Point + Normal":
        p0 = st.text_input(
            "Point (x,y,z)", value=",".join(map(str, st.session_state["plane_point"]))
        )
        n0 = st.text_input(
            "Normal (nx,ny,nz)",
            value=",".join(map(str, st.session_state["plane_normal"])),
        )
        def parse_point(text: str) -> Tuple[float, float, float]:
            parts = [float(x.strip()) for x in text.split(",")]
            if len(parts) != 3:
                raise ValueError("Need three values")
            return parts[0], parts[1], parts[2]
        try:
            st.session_state["plane_point"] = parse_point(p0)
            st.session_state["plane_normal"] = parse_point(n0)
        except Exception:
            st.warning("Enter values as x,y,z.")
    else:
        st.selectbox(
            "Axis plane",
            ["XY", "YZ", "XZ"],
            index=["XY", "YZ", "XZ"].index(st.session_state["plane_axis"]),
            key="plane_axis",
        )
        st.number_input(
            "Offset (mm)",
            value=float(st.session_state["plane_offset"]),
            step=0.1,
            format="%.3f",
            key="plane_offset",
        )

left_col, right_col = st.columns([1.5, 1.0])

with left_col:
    tab1, tab2, tab3 = st.tabs(
        [
            "Target Points (Sample)",
            "Feature Points (Sample)",
            "Feature Points (Instrument)",
        ]
    )
    with tab1:
        points_editor(
            "Target Points (Sample)",
            "target_sample_df",
            "Coordinates inside/outside the sample, in sample frame.",
            import_key="target_sample",
            export_key="target_sample",
        )
    with tab2:
        points_editor(
            "Feature Points (Sample)",
            "feature_sample_df",
            "Use 4 to 6 sharp exterior features. Row order must match instrument points.",
            import_key="feature_sample",
            export_key="feature_sample",
        )
    with tab3:
        points_editor(
            "Feature Points (Instrument)",
            "feature_instrument_df",
            "Measured with theodolite in instrument frame. Row order must match sample features.",
            import_key="feature_instrument",
            export_key="feature_instrument",
        )

transform_result = None
transform_errors = None
transform_warnings = []

P_feat, labels_feat = df_to_points(st.session_state["feature_sample_df"])
Q_feat, labels_instr = df_to_points(st.session_state["feature_instrument_df"])

if P_feat.shape[0] != Q_feat.shape[0]:
    if P_feat.shape[0] > 0 or Q_feat.shape[0] > 0:
        transform_warnings.append("Feature point counts do not match.")

point_count = P_feat.shape[0]
if point_count and not (4 <= point_count <= 6):
    transform_warnings.append("Provide 4 to 6 feature points for solving.")

if point_count >= 4:
    if is_coplanar(P_feat, COPLANAR_THRESHOLD_MM):
        transform_warnings.append(
            "Sample feature points are coplanar within 1.0 mm; mirror ambiguity may exist."
        )
    if is_coplanar(Q_feat, COPLANAR_THRESHOLD_MM):
        transform_warnings.append(
            "Instrument feature points are coplanar within 1.0 mm; mirror ambiguity may exist."
        )

if (
    point_count >= 4
    and point_count <= 6
    and P_feat.shape[0] == Q_feat.shape[0]
    and P_feat.shape[0] > 0
):
    transform_result = solve_transform(
        P_feat,
        Q_feat,
        scale_value=float(st.session_state["scale_value"]),
        auto_scale=st.session_state["auto_scale"],
        axis_flip=st.session_state["axis_flip"],
        enforce_proper=True,
    )
    rms, max_err, residuals = compute_errors(P_feat, Q_feat, transform_result)
    transform_errors = {
        "rms_mm": rms,
        "max_mm": max_err,
    }
    if rms > RMS_WARN_MM:
        transform_warnings.append(f"RMS error {rms:.4f} mm exceeds 0.1 mm.")
    if max_err > MAX_WARN_MM:
        transform_warnings.append(f"Max error {max_err:.4f} mm exceeds 0.2 mm.")

with right_col:
    st.subheader("Transform Summary")
    if transform_result is None:
        st.info("Provide 4 to 6 matching feature points to compute the transform.")
    else:
        st.write(f"Scale: {transform_result['scale']:.6f}")
        st.write(f"Axis flip: {transform_result['axis_flip']}")
        st.write("Rotation (3x3):")
        st.dataframe(
            pd.DataFrame(transform_result["rotation"], columns=["x", "y", "z"]),
            use_container_width=True,
        )
        st.write("Translation (mm):")
        st.dataframe(
            pd.DataFrame(
                [transform_result["translation"]], columns=["x", "y", "z"]
            ),
            use_container_width=True,
        )
        if transform_errors:
            st.write(
                f"RMS error: {transform_errors['rms_mm']:.4f} mm | "
                f"Max error: {transform_errors['max_mm']:.4f} mm"
            )

        matrix = build_matrix(transform_result)
        st.write("Transform matrix (4x4):")
        st.dataframe(pd.DataFrame(matrix), use_container_width=True)
        st.download_button(
            "Download transform matrix CSV",
            data=matrix_to_csv(matrix),
            file_name="transform_matrix.csv",
            mime="text/csv",
        )
        st.caption(
            "Use the current stage pose (instrument frame) to convert the solved sample pose "
            "into sample mount coordinates on the stage."
        )
        st.button(
            "Apply Theodolite Fit To Stage Mount",
            on_click=apply_theodolite_fit_to_stage_mount,
            disabled=transform_result is None,
            use_container_width=True,
        )

        mount_apply_level = st.session_state.get("mount_fit_apply_status_level", "")
        mount_apply_msg = st.session_state.get("mount_fit_apply_status_msg", "")
        if mount_apply_msg:
            if mount_apply_level == "success":
                st.success(mount_apply_msg)
            elif mount_apply_level == "warning":
                st.warning(mount_apply_msg)
            elif mount_apply_level == "error":
                st.error(mount_apply_msg)
            else:
                st.info(mount_apply_msg)

    if transform_warnings:
        for msg in transform_warnings:
            st.warning(msg)

if project_download_slot is not None:
    project_payload = build_project_payload(transform_result, transform_errors)
    project_download_slot.download_button(
        "Download project",
        data=json.dumps(project_payload, indent=2),
        file_name="coortrans_project.json",
        mime="application/json",
    )
st.subheader("Viewer")

view_btn_cols = st.columns(7)
for btn_col, direction in zip(
    view_btn_cols, ["+X", "-X", "+Y", "-Y", "+Z", "-Z", "Theodolite"]
):
    with btn_col:
        st.button(
            "Theo" if direction == "Theodolite" else direction,
            key=f"viewer_snap_{direction}",
            use_container_width=True,
            on_click=set_viewer_axis_view,
            args=(direction,),
        )

if st.session_state["stl_bytes"] is None:
    st.info("Upload an STL file to view it here.")
else:
    try:
        stl_mesh = mesh_from_bytes(st.session_state["stl_bytes"])
        intersection_segments = None
        plane_params = None
        plane_origin = None
        plane_normal = None
        camera = st.session_state.get("viewer_camera_override")
        stage_extra_meshes = []
        stage_preview_transform = None
        stage_matrix = stage_pose_matrix_from_state()
        stage_center_stage = np.array(
            [
                [
                    float(st.session_state["stage_center_gx"]),
                    float(st.session_state["stage_center_gy"]),
                    float(st.session_state["stage_center_gz"]),
                ]
            ],
            dtype=float,
        )
        stage_center_instr = apply_pose_matrix(stage_center_stage, stage_matrix)
        sample_mount_matrix = pose_matrix_xyz(
            float(st.session_state["sample_mount_tx"]),
            float(st.session_state["sample_mount_ty"]),
            float(st.session_state["sample_mount_tz"]),
            float(st.session_state["sample_mount_rx"]),
            float(st.session_state["sample_mount_ry"]),
            float(st.session_state["sample_mount_rz"]),
        )
        stage_preview_matrix = stage_matrix @ sample_mount_matrix
        stage_preview_transform = matrix_to_rigid_transform(stage_preview_matrix)
        if st.session_state["stage_show"]:
            stage_vectors = cuboid_vectors(*STAGE_DIMS_MM)
            stage_vectors_instr = transform_vectors_by_matrix(stage_vectors, stage_matrix)
            stage_extra_meshes.append(
                {
                    "vectors": stage_vectors_instr,
                    "color": "#8d99ae",
                    "opacity": 0.35,
                    "name": "Stage",
                    "showlegend": True,
                }
            )
        if st.session_state["plane_enabled"] and st.session_state["view_frame"] == "Sample":
            p0, n = plane_from_inputs()
            if np.linalg.norm(n) == 0:
                st.warning("Plane normal is zero; cannot compute intersection.")
            else:
                plane_origin = p0
                plane_normal = n
                intersection_segments = mesh_plane_intersections(stl_mesh, p0, n)
                vectors = stl_mesh.vectors.reshape(-1, 3)
                bounds_min = vectors.min(axis=0)
                bounds_max = vectors.max(axis=0)
                plane_params = plane_uv_bounds(p0, n, bounds_min, bounds_max)
                signature = (
                    f"{st.session_state['plane_mode']}|"
                    f"{p0[0]:.6f},{p0[1]:.6f},{p0[2]:.6f}|"
                    f"{n[0]:.6f},{n[1]:.6f},{n[2]:.6f}"
                )
                if st.session_state["plane_signature"] != signature:
                    st.session_state["plane_signature"] = signature
                    st.session_state["plane_pick_uv"] = []
                    st.session_state["plane_preview_uv"] = []
                    st.session_state["last_plane_preview_id"] = None
                    st.session_state["plane_pick_rev"] += 1
                if (
                    st.session_state["plane_view_normal"]
                    and st.session_state["apply_plane_camera_once"]
                ):
                    diag = float(np.linalg.norm(bounds_max - bounds_min)) or 1.0
                    eye = normalize_vector(n) * (0.6 * diag)
                    camera = dict(
                        eye=dict(x=eye[0], y=eye[1], z=eye[2]),
                        center=dict(x=0.0, y=0.0, z=0.0),
                        up=dict(x=0.0, y=0.0, z=1.0),
                        projection=dict(type="orthographic"),
                    )
                    st.session_state["apply_plane_camera_once"] = False
        point_traces = []
        if st.session_state["view_frame"] == "Sample":
            target_points, target_labels = df_to_points(
                st.session_state["target_sample_df"]
            )
            feature_points, feature_labels = df_to_points(
                st.session_state["feature_sample_df"]
            )
            # Always render committed/manual points in sample view.
            # Plane-picker performance toggles should not hide user-entered coordinates.
            if target_points.size:
                point_traces.append(
                    (target_points, target_labels, "Target Points", "#2ca02c")
                )
            if feature_points.size:
                point_traces.append(
                    (feature_points, feature_labels, "Feature Points", "#d62728")
                )
            if (
                st.session_state["plane_enabled"]
                and st.session_state["plane_show_3d_points_live"]
                and plane_params is not None
                and plane_origin is not None
                and st.session_state["plane_preview_uv"]
            ):
                u_axis_preview, v_axis_preview, *_ = plane_params
                preview_rows = []
                for uv in st.session_state["plane_preview_uv"]:
                    uu = float(uv[0])
                    vv = float(uv[1])
                    picked = plane_origin + u_axis_preview * uu + v_axis_preview * vv
                    if st.session_state["plane_snap_to_curve"] and intersection_segments:
                        picked, _ = nearest_point_on_segments(picked, intersection_segments)
                    preview_rows.append(picked)
                if preview_rows:
                    point_traces.append(
                        (
                            np.array(preview_rows, dtype=float),
                            [""] * len(preview_rows),
                            "Pending Picks (Preview)",
                            "#9467bd",
                        )
                    )
            render_scene(
                stl_mesh,
                opacity=1.0 - st.session_state["model_transparency"],
                color=st.session_state["model_color"],
                point_traces=point_traces,
                show_labels=st.session_state["show_symbols"],
                symbol_size=st.session_state["symbol_size"],
                intersection_segments=intersection_segments,
                camera=camera,
                extra_mesh_traces=None,
            )
            if (
                st.session_state["plane_enabled"]
                and plane_params is not None
                and plane_origin is not None
                and plane_normal is not None
            ):
                if not os.path.isdir(PLANE_PICKER_BUILD):
                    st.warning(
                        "Plane picker component is not built yet. Run npm install && npm run build in components/plane_picker/frontend."
                    )
                else:
                    u_axis, v_axis, umin, umax, vmin, vmax = plane_params
                    segments_uv = (
                        project_segments_uv(
                            intersection_segments, plane_origin, u_axis, v_axis
                        )
                        if intersection_segments
                        else []
                    )
                    st.subheader("Plane Picker")
                    pick = plane_picker(
                        p0=plane_origin.tolist(),
                        u=u_axis.tolist(),
                        v=v_axis.tolist(),
                        umin=umin,
                        umax=umax,
                        vmin=vmin,
                        vmax=vmax,
                        segments=segments_uv,
                        picks=st.session_state["plane_pick_uv"],
                        picks_rev=st.session_state["plane_pick_rev"],
                        live_preview=st.session_state["plane_show_3d_points_live"],
                        width=1000,
                        height=520,
                        key="plane_picker",
                    )
                    if pick and "preview_id" in pick:
                        preview_id = pick.get("preview_id")
                        if preview_id != st.session_state["last_plane_preview_id"]:
                            st.session_state["last_plane_preview_id"] = preview_id
                            st.session_state["plane_preview_uv"] = pick.get(
                                "preview_picks", []
                            )
                    if (
                        pick
                        and "commit_id" in pick
                        and pick.get("commit_id")
                        != st.session_state["last_plane_commit_id"]
                    ):
                        st.session_state["last_plane_commit_id"] = pick.get("commit_id")
                        st.session_state["plane_preview_uv"] = []
                        st.session_state["last_plane_preview_id"] = None
                        uv_points = pick.get("picks", [])
                        if uv_points:
                            add_to = st.session_state["click_add_to"]
                            df_key = (
                                "target_sample_df"
                                if add_to == "Target Points (Sample)"
                                else "feature_sample_df"
                            )
                            df = clean_points_df(st.session_state[df_key].copy())
                            rows = []
                            for uv in uv_points:
                                uu = float(uv[0])
                                vv = float(uv[1])
                                picked = plane_origin + u_axis * uu + v_axis * vv
                                if (
                                    st.session_state["plane_snap_to_curve"]
                                    and intersection_segments
                                ):
                                    picked, _ = nearest_point_on_segments(
                                        picked, intersection_segments
                                    )
                                st.session_state["plane_pick_uv"].append(
                                    point_to_uv(picked, plane_origin, u_axis, v_axis)
                                )
                                rows.append(
                                    {"Label": "", "X": picked[0], "Y": picked[1], "Z": picked[2]}
                                )
                            st.session_state["plane_pick_rev"] += 1
                            if rows:
                                df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
                                st.session_state[df_key] = df
        elif st.session_state["view_frame"] == "Stage":
            stage_vectors = cuboid_vectors(*STAGE_DIMS_MM)
            mounted_mesh = transform_mesh_by_matrix(stl_mesh, sample_mount_matrix)
            target_points, target_labels = df_to_points(st.session_state["target_sample_df"])
            feature_points, feature_labels = df_to_points(
                st.session_state["feature_sample_df"]
            )
            stage_point_traces = []
            mounted_targets = apply_pose_matrix(target_points, sample_mount_matrix)
            mounted_features = apply_pose_matrix(feature_points, sample_mount_matrix)
            if mounted_targets.size:
                stage_point_traces.append(
                    (mounted_targets, target_labels, "Target Points (Stage)", "#2ca02c")
                )
            if mounted_features.size:
                stage_point_traces.append(
                    (mounted_features, feature_labels, "Feature Points (Stage)", "#d62728")
                )
            stage_point_traces.append(
                (
                    stage_center_stage,
                    ["Stage Center (Correlated)"],
                    "Stage Center (Stage Frame)",
                    "#ffbf00",
                )
            )
            render_scene(
                mounted_mesh,
                opacity=1.0 - st.session_state["model_transparency"],
                color=st.session_state["model_color"],
                point_traces=stage_point_traces,
                show_labels=st.session_state["show_symbols"],
                symbol_size=st.session_state["symbol_size"],
                intersection_segments=None,
                camera=camera,
                extra_mesh_traces=[
                    {
                        "vectors": stage_vectors,
                        "color": "#8d99ae",
                        "opacity": 0.35,
                        "name": "Stage",
                        "showlegend": True,
                    }
                ],
            )
        else:
            pose_source = st.session_state.get("instrument_pose_source", "Theodolite fit")
            fit_pose = transform_result
            preview_pose = stage_preview_transform
            instrument_pose = None
            secondary_pose = None
            secondary_pose_name = None
            secondary_pose_color = None
            show_stage_preview_info = False

            if pose_source == "Stage preview":
                instrument_pose = preview_pose
                if fit_pose is not None:
                    st.info("Instrument view is showing the stage placement preview pose.")
            elif pose_source == "Theodolite fit":
                if fit_pose is not None:
                    instrument_pose = fit_pose
                else:
                    instrument_pose = preview_pose
                    show_stage_preview_info = True
            elif pose_source == "Both":
                if fit_pose is not None:
                    instrument_pose = fit_pose
                    if preview_pose is not None:
                        secondary_pose = preview_pose
                        secondary_pose_name = "Sample (Stage Preview)"
                        secondary_pose_color = "#6c757d"
                else:
                    instrument_pose = preview_pose
                    show_stage_preview_info = True
            else:
                instrument_pose = fit_pose or preview_pose
                show_stage_preview_info = fit_pose is None and preview_pose is not None

            if instrument_pose is None:
                st.warning("Transform not available; showing sample frame instead.")
                target_points, target_labels = df_to_points(
                    st.session_state["target_sample_df"]
                )
                feature_points, feature_labels = df_to_points(
                    st.session_state["feature_sample_df"]
                )
                if target_points.size:
                    point_traces.append(
                        (
                            target_points,
                            target_labels,
                            "Target Points",
                            "#2ca02c",
                        )
                    )
                if feature_points.size:
                    point_traces.append(
                        (
                            feature_points,
                            feature_labels,
                            "Feature Points",
                            "#d62728",
                        )
                    )
                render_scene(
                    stl_mesh,
                    opacity=1.0 - st.session_state["model_transparency"],
                    color=st.session_state["model_color"],
                    point_traces=point_traces,
                    show_labels=st.session_state["show_symbols"],
                    symbol_size=st.session_state["symbol_size"],
                    intersection_segments=intersection_segments,
                    camera=camera,
                    extra_mesh_traces=None,
                )
            else:
                if show_stage_preview_info:
                    st.info(
                        "Using stage placement preview in instrument frame. "
                        "Compute theodolite transform to refine the final pose."
                    )
                transformed_mesh = transform_mesh(stl_mesh, instrument_pose)
                target_points, target_labels = df_to_points(
                    st.session_state["target_sample_df"]
                )
                transformed_targets = apply_transform(target_points, instrument_pose)
                instrument_extra_meshes = list(stage_extra_meshes)
                point_traces.append(
                    (
                        stage_center_instr,
                        ["Stage Center"],
                        "Stage Center (Instrument)",
                        "#ffbf00",
                    )
                )
                if st.session_state.get("beam_show", True):
                    x_points = [transformed_mesh.vectors[:, :, 0].min()]
                    x_points.append(0.0)
                    for extra_mesh in instrument_extra_meshes:
                        extra_vectors = extra_mesh.get("vectors")
                        if isinstance(extra_vectors, np.ndarray) and extra_vectors.size:
                            x_points.append(float(extra_vectors[:, :, 0].min()))
                    x_min_scene = float(min(x_points))
                    x_max_scene = float(
                        max(
                            [0.0, float(transformed_mesh.vectors[:, :, 0].max())]
                            + [
                                float(extra_mesh["vectors"][:, :, 0].max())
                                for extra_mesh in instrument_extra_meshes
                                if isinstance(extra_mesh.get("vectors"), np.ndarray)
                                and extra_mesh["vectors"].size
                            ]
                        )
                    )
                    x_span_scene = max(x_max_scene - x_min_scene, 1.0)
                    beam_margin = max(20.0, 0.08 * x_span_scene)
                    beam_start_x = min(-500.0, x_min_scene - beam_margin)
                    beam_vectors = beam_cuboid_vectors(
                        beam_start_x,
                        0.0,
                        float(st.session_state.get("beam_size_y_mm", 4.0)),
                        float(st.session_state.get("beam_size_z_mm", 4.0)),
                    )
                    instrument_extra_meshes.append(
                        {
                            "vectors": beam_vectors,
                            "color": "#f4c430",
                            "opacity": 0.55,
                            "name": "Beam",
                            "showlegend": True,
                        }
                    )
                if secondary_pose is not None:
                    secondary_mesh = transform_mesh(stl_mesh, secondary_pose)
                    instrument_extra_meshes.append(
                        {
                            "vectors": secondary_mesh.vectors,
                            "color": secondary_pose_color or "#6c757d",
                            "opacity": 0.22,
                            "name": secondary_pose_name or "Sample (Secondary Pose)",
                            "showlegend": True,
                        }
                    )
                    if target_points.size:
                        point_traces.append(
                            (
                                apply_transform(target_points, secondary_pose),
                                target_labels,
                                "Target Points (Stage Preview)",
                                "#17becf",
                            )
                        )
                feature_instr_points, feature_instr_labels = df_to_points(
                    st.session_state["feature_instrument_df"]
                )
                if transformed_targets.size:
                    point_traces.append(
                        (
                            transformed_targets,
                            target_labels,
                            "Target Points (Instrument)",
                            "#2ca02c",
                        )
                    )
                if feature_instr_points.size:
                    point_traces.append(
                        (
                            feature_instr_points,
                            feature_instr_labels,
                            "Feature Points (Instrument)",
                            "#d62728",
                        )
                    )
                render_scene(
                    transformed_mesh,
                    opacity=1.0 - st.session_state["model_transparency"],
                    color=st.session_state["model_color"],
                    point_traces=point_traces,
                    show_labels=st.session_state["show_symbols"],
                    symbol_size=st.session_state["symbol_size"],
                    intersection_segments=None,
                    camera=camera,
                    extra_mesh_traces=instrument_extra_meshes,
                )
    except Exception as exc:
        st.error(f"Failed to load STL file: {exc}")

st.subheader("Transformed Target Points (Instrument)")
if transform_result is None:
    st.info("Compute a transform to view transformed target points.")
else:
    target_points, target_labels = df_to_points(st.session_state["target_sample_df"])
    transformed_targets = apply_transform(target_points, transform_result)
    if transformed_targets.size == 0:
        st.info("No target points to transform.")
    else:
        out_df = pd.DataFrame(transformed_targets, columns=["X", "Y", "Z"])
        out_df.insert(0, "Label", target_labels)
        out_df.insert(0, "#", range(1, len(out_df) + 1))
        st.dataframe(out_df, use_container_width=True)
        export_df = out_df.drop(columns=["#"], errors="ignore")
        export_df.columns = ["Label", "X", "Y", "Z"]
        data, mime, ext = export_points(export_df, "CSV")
        st.download_button(
            "Download transformed points CSV",
            data=data,
            file_name="target_points_instrument.csv",
            mime=mime,
        )
