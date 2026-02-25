# CoorTrans Specification (Draft)

Status: Draft
Owner: Ray Zhang
Last updated: 2026-02-03

## Vision
Enable neutron imaging beamline scientists and users to align digital sample models to instrument stage coordinates, so they can select internal/external points in the model and obtain accurate instrument-frame coordinates for experiments.

## Goals
- Provide a reliable workflow to register sample coordinates to instrument stage coordinates.
- Support point selection inside/outside the sample geometry and transform those points into instrument coordinates.
- Visualize the transformed STL mesh and points for verification before use on the instrument.

## Non-Goals
- High-fidelity CAD editing or mesh repair.
- Automated feature recognition on STL meshes (manual feature selection is expected initially).

## Personas
- Persona 1: Neutron imaging beamline scientists.
- Persona 2: Beamline users preparing and aligning samples.

## Primary Workflows
1. Load a digital model (STL) of a sample with complex geometry.
2. Select internal and external points on the model; record their sample-coordinate positions.
3. Select at least four sharp exterior feature points on the model; record their sample coordinates.
4. Measure the same feature points using a theodolite on the instrument stage; record instrument coordinates.
5. Compute a coordinate transform between sample and instrument systems, including scale, rotation, and translation.
6. Apply the transform to internal points and to the STL mesh for visualization.
7. Review fit error and warnings; export transformed points for downstream use.

## Functional Requirements
### Core
- Allow users to load an STL model and visualize it in 3D.
- Allow users to create, edit, and label point coordinates associated with the model.

### Data Management
- Store two point sets: sample-coordinate points and instrument-coordinate points.
- Support replacing or appending point sets during import.
- Support a project file that saves STL reference, point sets, transform, and settings.
- Allow only one active transform per project.

### Visualization
- Render STL mesh with adjustable transparency and color.
- Render point markers with labels and adjustable styling.
- Visualize the transformed STL mesh in instrument coordinates for verification.

### Coordinate Operations
- Compute a similarity transform with user-defined linear scale: `p_instr = s * R * p_sample + t`.
- Require at least 4 non-coplanar feature points to resolve mirror ambiguity.
- Allow 4 to 6 feature points for solving; warn that larger sets are not supported initially.
- Support handedness control:
  - Users specify whether each coordinate system is left- or right-handed.
  - If handedness differs, require an explicit axis flip (X/Y/Z).
  - Provide an auto-suggest option that selects the axis flip with lowest fit error and clearly displays the choice.
- Support optional auto-estimation of scale from the point pairs.
- Report fit error (RMS and max); warn when RMS > 0.1 mm or max > 0.2 mm.

## Transform Math
Goal: solve for a similarity transform that maps sample points to instrument points.

Definitions:
- Sample feature points: `P_i` in sample coordinates.
- Instrument feature points: `Q_i` in instrument coordinates.
- Transform: `Q_i ≈ s * R * P_i + t`
- `s` is a unitless scale (user-provided or auto-estimated).
- `R` is 3x3 rotation (proper if handedness matches; may include reflection when handedness differs).
- `t` is 3x1 translation.

Handedness handling:
1. User selects handedness of sample and instrument.
2. If handedness differs, require explicit axis flip (X/Y/Z) on the sample points before solving.
3. Offer auto-suggest that tries X/Y/Z flips and chooses the lowest RMS error, then displays the chosen axis.

Solve steps (fixed scale `s`):
1. Apply axis flip (if any) to all `P_i`.
2. Compute centroids `P_c` and `Q_c`.
3. Center points: `P'_i = P_i - P_c`, `Q'_i = Q_i - Q_c`.
4. Apply scale to sample points: `P''_i = s * P'_i`.
5. Compute covariance `H = sum(P''_i * Q'_i^T)`.
6. SVD: `H = U * S * V^T`.
7. Compute rotation `R = V * U^T`. If handedness must be preserved, enforce `det(R) = +1`.
8. Compute translation `t = Q_c - R * (s * P_c)`.

Optional auto-scale:
- Estimate `s` from the point pairs using a similarity-fit method (Umeyama-style).
- Use estimated `s` in the same solve steps above.

Error metrics:
- Residuals: `e_i = Q_i - (s * R * P_i + t)`.
- RMS error: `sqrt(mean(||e_i||^2))`.
- Max error: `max(||e_i||)`.
- Warn if RMS > 0.1 mm or max > 0.2 mm.

### Import/Export
- Import point coordinates from CSV/TXT/XLSX.
- Export point coordinates to CSV/TXT/XLSX.
- Export transformed points in instrument coordinates.
- Export the 4x4 transform matrix as CSV.

## Data Formats and Units
- Supported file types: STL (model), CSV/TXT/XLSX (points).
- Units: millimeters (mm) only. No unit switching in the UI.
- Scale coefficient: unitless; used for linear scaling between coordinate systems.
- Precision / rounding rules: preserve full precision in computations; format for display only.

## Project File Schema
Project file format: JSON (single file).
STL storage: embedded (base64) for self-contained project files.

Proposed schema:
```json
{
  "version": "1.0",
  "units": "mm",
  "stl_embedded": {
    "filename": "sample.stl",
    "encoding": "base64",
    "data": "..."
  },
  "handedness": {
    "sample": "right",
    "instrument": "right",
    "axis_flip": "none"
  },
  "points": {
    "feature_sample": [
      {"id": 1, "label": "F1", "x": 0.0, "y": 0.0, "z": 0.0}
    ],
    "feature_instrument": [
      {"id": 1, "label": "F1", "x": 0.0, "y": 0.0, "z": 0.0}
    ],
    "target_sample": [
      {"id": 1, "label": "P1", "x": 0.0, "y": 0.0, "z": 0.0}
    ]
  },
  "transform": {
    "scale": 1.0,
    "rotation": [[1,0,0],[0,1,0],[0,0,1]],
    "translation": [0.0, 0.0, 0.0],
    "matrix_4x4": [
      [1,0,0,0],
      [0,1,0,0],
      [0,0,1,0],
      [0,0,0,1]
    ],
    "fit_error": {
      "rms_mm": 0.0,
      "max_mm": 0.0
    }
  },
  "settings": {
    "model_color": "#4c78a8",
    "model_opacity": 1.0,
    "symbol_size": 6,
    "symbol_color": "#e45756"
  }
}
```

## CSV Schemas
Points CSV:
- Columns: `id`, `label`, `x`, `y`, `z`
- Separate files for:
  - Feature points in sample coordinates
  - Feature points in instrument coordinates
  - Target points in sample coordinates
  - Transformed target points in instrument coordinates

Transform matrix CSV:
- 4 rows x 4 columns.
- Optional header row: `m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33`

## UX and Interaction
- Layout (sidebar, main canvas, panels): sidebar for inputs and settings; main canvas for 3D model.
- Editing behavior: table-based editing for point coordinates with immediate visualization updates.
- Point selection options:
  - Manual entry or file import of coordinates.
  - Click-select points on a user-defined plane that intersects the STL geometry (Abaqus-style plane selection).
- Validation and error handling:
  - Warn if feature points are fewer than 4 or coplanar (coplanar threshold: 1.0 mm max distance from best-fit plane).
  - Enforce 4 to 6 feature points for solving; warn otherwise.
  - Warn if handedness mismatch is unresolved.
  - Warn if RMS > 0.1 mm or max > 0.2 mm.

## Non-Functional Requirements
- Performance targets: interactive rendering for typical STL sizes used in beamline work (TBD).
- Accuracy/tolerance: fit error reporting in micrometers; warnings at RMS > 0.1 mm and max > 0.2 mm.
- Accessibility considerations: keyboard navigation for tables; readable contrast for labels.
- Security/privacy: all processing local; no external upload required.

## Dependencies
- Runtime:
- Python packages: Streamlit and current visualization stack.
- Optional services:

## Risks and Assumptions
- Feature points must be chosen consistently between the model and theodolite measurements.
- Handedness mismatch, if not handled explicitly, can cause mirrored transforms.

## Acceptance Criteria
- User can compute a transform from ≥4 feature points and apply it to points and STL mesh.
- App reports RMS and max fit error and warns above 100 μm.

## Open Questions
- How should project files be stored (single file vs folder, JSON vs CSV bundle)?
- What are typical STL sizes (triangles) and point counts for performance sizing?
- Should click-selection on a plane be MVP or phase 2?
