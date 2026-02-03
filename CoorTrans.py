import io
import os
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from stl import mesh


st.set_page_config(page_title="CoorTrans", layout="wide")
st.title("CoorTrans")

uploaded_file = st.sidebar.file_uploader("Choose an STL file", type=["stl"])
st.sidebar.caption("Upload a binary or ASCII STL file to preview the model.")

model_transparency = st.sidebar.slider(
    "Model transparency (0=opaque, 1=transparent)", 0.0, 1.0, 0.0, 0.05
)
model_color = st.sidebar.color_picker("Model color", "#4c78a8")
show_symbols = st.sidebar.checkbox("Show symbols", value=True)
symbol_size = st.sidebar.slider("Symbol size", 2, 20, 6)
symbol_color = st.sidebar.color_picker("Symbol color", "#e45756")

if "points" not in st.session_state:
    st.session_state.points = []


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


def _extract_points(df: pd.DataFrame) -> list[tuple[float, float, float]]:
    if df.empty or df.shape[1] < 3:
        return []
    names = [str(col).strip().lower() for col in df.columns]
    if {"x", "y", "z"}.issubset(names):
        col_map = {name: idx for idx, name in enumerate(names)}
        cols = [col_map["x"], col_map["y"], col_map["z"]]
        coords_df = df.iloc[:, cols]
    else:
        coords_df = df.iloc[:, :3]
    coords_df = coords_df.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    return [tuple(row) for row in coords_df.to_numpy()]


def load_coordinates(uploaded) -> list[tuple[float, float, float]]:
    file_bytes = uploaded.getvalue()
    extension = os.path.splitext(uploaded.name)[1].lower()
    df_with_header = _read_table(file_bytes, extension, header=0)
    points_with_header = _extract_points(df_with_header)
    names = [str(col).strip().lower() for col in df_with_header.columns]
    has_xyz_header = {"x", "y", "z"}.issubset(names)
    if has_xyz_header:
        return points_with_header
    df_no_header = _read_table(file_bytes, extension, header=None)
    points_no_header = _extract_points(df_no_header)
    if len(points_no_header) > len(points_with_header):
        return points_no_header
    return points_with_header

with st.sidebar.expander("Coordinate symbols", expanded=True):
    if st.session_state.points:
        coords_df = pd.DataFrame(st.session_state.points, columns=["X", "Y", "Z"])
    else:
        coords_df = pd.DataFrame([[0.0, 0.0, 0.0]], columns=["X", "Y", "Z"])
    coords_df.insert(0, "#", range(1, len(coords_df) + 1))

    edited_df = st.data_editor(
        coords_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "#": st.column_config.NumberColumn("#", disabled=True, width="small"),
            "X": st.column_config.NumberColumn("X", format="%.3f"),
            "Y": st.column_config.NumberColumn("Y", format="%.3f"),
            "Z": st.column_config.NumberColumn("Z", format="%.3f"),
        },
        key="coords_table",
    )

    points_df = edited_df.drop(columns=["#"], errors="ignore")
    points_df = points_df.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    st.session_state.points = [tuple(row) for row in points_df.to_numpy()]

    if st.session_state.points:
        export_df = pd.DataFrame(st.session_state.points, columns=["X", "Y", "Z"])
        file_format = st.selectbox("Save format", ["CSV", "TXT", "XLSX"])
        file_name = st.text_input("Filename", value="coordinates")
        base_name = file_name.strip() or "coordinates"
        ext_map = {"CSV": ".csv", "TXT": ".txt", "XLSX": ".xlsx"}
        if os.path.splitext(base_name)[1].lower() in ext_map.values():
            base_name = os.path.splitext(base_name)[0]
        final_name = f"{base_name}{ext_map[file_format]}"

        if file_format == "XLSX":
            buffer = io.BytesIO()
            export_df.to_excel(buffer, index=False, header=False)
            data = buffer.getvalue()
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif file_format == "TXT":
            data = export_df.to_csv(index=False, header=False, sep="\t")
            mime = "text/plain"
        else:
            data = export_df.to_csv(index=False, header=False)
            mime = "text/csv"

        st.download_button(
            "Save coordinates",
            data=data,
            file_name=final_name,
            mime=mime,
        )

with st.sidebar.expander("Upload coordinates file", expanded=False):
    coords_file = st.file_uploader(
        "Choose a coordinates file", type=["csv", "txt", "xlsx"]
    )
    replace_coords = st.checkbox("Replace existing coordinates", value=False)
    load_coords = st.button("Load coordinates", disabled=coords_file is None)
    if load_coords and coords_file is not None:
        try:
            loaded_points = load_coordinates(coords_file)
            if not loaded_points:
                st.warning("No valid coordinates found in the file.")
            else:
                if replace_coords:
                    st.session_state.points = loaded_points
                else:
                    st.session_state.points.extend(loaded_points)
                st.success(f"Loaded {len(loaded_points)} coordinates.")
        except Exception as exc:
            st.error(f"Failed to load coordinates: {exc}")


def render_stl(
    stl_mesh: mesh.Mesh,
    *,
    opacity: float,
    color: str,
    points: list[tuple[float, float, float]],
    show_symbols: bool,
    symbol_size: int,
    symbol_color: str,
) -> None:
    vectors = stl_mesh.vectors
    x = vectors[:, :, 0].ravel()
    y = vectors[:, :, 1].ravel()
    z = vectors[:, :, 2].ravel()
    i = np.arange(0, len(x), 3)
    j = i + 1
    k = i + 2

    fig = go.Figure(
        data=[
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
            )
        ]
    )
    if show_symbols and points:
        px, py, pz = zip(*points)
        labels = [f"({px[i]:.3f}, {py[i]:.3f}, {pz[i]:.3f})" for i in range(len(points))]
        fig.add_trace(
            go.Scatter3d(
                x=px,
                y=py,
                z=pz,
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker=dict(size=symbol_size, color=symbol_color),
                name="Coordinates",
            )
        )
    fig.update_layout(
        scene=dict(aspectmode="data", uirevision="coortrans"),
        margin=dict(l=0, r=0, t=20, b=0),
        showlegend=False,
        uirevision="coortrans",
    )
    st.plotly_chart(fig, use_container_width=True, key="stl_viewer")


if uploaded_file is None:
    st.info("Upload an STL file from the sidebar to view it here.")
else:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        stl_mesh = mesh.Mesh.from_file(tmp_path)
        render_stl(
            stl_mesh,
            opacity=1.0 - model_transparency,
            color=model_color,
            points=st.session_state.points,
            show_symbols=show_symbols,
            symbol_size=symbol_size,
            symbol_color=symbol_color,
        )
    except Exception as exc:
        st.error(f"Failed to load STL file: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
