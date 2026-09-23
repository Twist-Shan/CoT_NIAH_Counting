#!/usr/bin/env python3
"""Build the restrained Aurora main figure and its provenance caption.

The editable output is a native draw.io document.  Text excerpts, the
non-thinking attention strip, the fixed-camera PCA comparison, and the
native-thinking endpoint matrix are generated directly from local research
artifacts so that the figure can be rebuilt deterministically.
"""

from __future__ import annotations

import csv
import gzip
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET

import numpy as np
from tokenizers import Tokenizer


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
REPO = WORKSPACE / "realistic"

OUT_DRAWIO = HERE / "main_figure_v1.drawio"
OUT_CAPTION = HERE / "caption_v1.md"
OUT_SETTINGS = HERE / "main_figure_v1_settings.json"

REPORT = REPO / "reports" / "NiaH_Geometry_Comparison.html"
NATIVE_ROWS = (
    REPO
    / "work"
    / "remote_native_traces_snapshot"
    / "Qwen3-8B"
    / "generations.jsonl"
)
TOKENIZER_JSON = (
    REPO
    / "work"
    / "hf_tokenizers"
    / "models--Qwen--Qwen3-8B"
    / "snapshots"
    / "b968826d9c46dd6066d109eabc6255188de91218"
    / "tokenizer.json"
)
NONTHINK_ATTN = (
    REPO
    / "exports"
    / "run_20260731_v4_numeric_presentation_v3"
    / "Qwen3-8B"
    / "numeric"
    / "attention"
    / "capture"
    / "raw_shards"
    / "v4.4"
    / "V4_4_T10000_N10_seed1255.npz"
)
NONTHINK_BEHAVIOR = (
    REPO
    / "exports"
    / "run_20260731_v4_numeric_presentation_v3"
    / "Qwen3-8B"
    / "numeric"
    / "behavior"
    / "capture"
    / "shards"
    / "v4.4"
    / "V4_4_T10000_N10_seed1255.json"
)
NONTHINK_HEAD_SUMMARY = (
    REPO
    / "exports"
    / "run_20260731_v4_numeric_presentation_v3"
    / "Qwen3-8B"
    / "numeric"
    / "attention"
    / "capture"
    / "shards"
    / "v4.4"
    / "V4_4_T10000_N10_seed1255.csv.gz"
)
NATIVE_TRANSITIONS = HERE / "seed1255_native_transitions.json"
NATIVE_TOKEN_ATTN = HERE / "seed1255_native_token_attention.npz"
NATIVE_TOKEN_ATTN_META = HERE / "seed1255_native_token_attention.json"

SELECTED_SEED = 1255
NONTHINK_LAYER = 28
NONTHINK_HEAD = 19
NATIVE_LAYER = 24
NATIVE_HEAD = 29
PCA_NONTHINK_LAYER = 12
PCA_NATIVE_LAYER = 30
# Both prompt demos show the same three records.  They are the targets of the
# three consecutive high-mass native retrievals 2->3, 3->4, and 4->5.
DISPLAYED_OCCURRENCES = (3, 4, 5)
NATIVE_DISPLAYED_OCCURRENCES = (3, 4, 5)

FULL_NONTHINK_QUESTION = """How many city-score audit records are in the passage?
Do not explain, reason aloud, quote, or list any records.
Your entire response must be exactly one line:
Total: <integer>"""

FULL_NATIVE_QUESTION = """How many city-score audit records are in the passage?
Reason concisely without repeating or restarting.
Stop as soon as you determine the count, then output exactly one line:
Total: <integer>"""


AURORA = {
    "red": "#FF5FA2",       # Sunset Pink, used as the restrained red channel.
    "yellow": "#F6E36A",
    "blue": "#00C2FF",
    "green": "#39E58C",
    "teal": "#00D4B4",
    "violet": "#6750E8",
    "magenta": "#C04DFF",
    "indigo": "#23165C",
    "black": "#161923",
    "white": "#FFFFFF",
    "gray": "#8190A5",
    "brown": "#765347",
}

# Slightly darker than Frost Gray for small explanatory text.  Structural
# rules and the PCA perspective grid keep the original gray so this change
# improves paper-scale legibility without visually increasing chart weight.
MUTED_TEXT = "#4F6176"

# Original count-wise Aurora gradient used in the source PCA report.  These
# intermediate colours are calibrated between the Aurora anchors so adjacent
# counts change hue smoothly.  The mapping is intentionally identical across
# both modes; geometry, not hue, carries the non-thinking/native-thinking
# comparison.
COUNT_COLORS = {
    1: "#6750E8",
    2: "#00A9D8",
    3: "#00A88F",
    4: "#2DBE77",
    5: "#A7C957",
    6: "#D6B52C",
    7: "#F29E4C",
    8: "#E76F51",
    9: "#D94B86",
    10: "#8E5DB7",
}

FONT_SCALE = 1.075

# Keep the broad non-thinking read visually subordinate to the much more
# concentrated targeted-retrieval examples.  This is a uniform display-only
# lightness multiplier: it preserves the ordering and relative variation of
# every raw token mass and is applied to the matching Panel A legend as well.
NONTHINK_PROMPT_ALPHA_SCALE = 0.62

# Match the restrained point treatment in the source report: every ordinary
# state uses one small fixed marker and one cross-panel opacity.  Centroid
# nodes likewise use one compact fixed diameter.  Depth is conveyed by the
# shared projection, painter order, and trajectory rather than marker alpha or
# size, so the two modes remain directly comparable.
PCA_STATE_OPACITY = 58
PCA_STATE_DIAMETER = 5.6
PCA_CENTROID_DIAMETER = 15.5

# PCA component signs are arbitrary.  Use the equivalent (-PC1, +PC2, -PC3)
# display convention so the two horizontal axes point down and away from the
# vertical PC2 axis.  Both modes use the same convention; point geometry and
# all distance-based statistics remain unchanged.
PCA_DISPLAY_AXIS_SIGNS = (-1.0, 1.0, -1.0)

# Match the two PCA subplots' screen-space footprint without changing either
# manifold's internal geometry.  The previous non-thinking diagonal was
# 1.129x the native diagonal under the shared camera, so 0.89 equalizes them.
# Axis lengths remain fixed and identical across panels.
PCA_NONTHINK_DISPLAY_SCALE = 0.75

CANVAS_W = 1864
CANVAS_H = 1194
MARGIN = 0
GAP = 16
COL_W = 924
LEFT_X = 0
RIGHT_X = 940
LEFT_A_Y = 0
LEFT_A_H = 402
LEFT_B_Y = 396
LEFT_B_H = 180
LEFT_C_Y = 582
LEFT_C_H = 612
RIGHT_D_Y = 0
RIGHT_D_H = 600
RIGHT_E_Y = 594
RIGHT_E_H = 600


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def token_attention_alpha(normalized: float) -> float:
    """One shared nonlinear print transform for every displayed prompt token.

    A deliberately early logarithmic knee makes moderately elevated mass
    visible after paper-scale reduction while near-zero background remains
    nearly white.  The transform has no knowledge of record spans: needle and
    haystack tokens are mapped identically and raw-attention ordering is kept.
    """
    value = clamp(normalized)
    if value <= 0.0:
        return 0.0
    # A high-k logarithmic knee makes moderately elevated raw mass legible in
    # print while leaving the near-zero haystack distribution close to white.
    # This remains one span-blind mapping: no needle membership enters here.
    expanded = math.log1p(1000.0 * value) / math.log(1001.0)
    return 0.004 + 0.946 * expanded ** 0.62


def hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def blend(base: str, alpha: float, background: str = AURORA["white"]) -> str:
    alpha = clamp(alpha)
    foreground = hex_rgb(base)
    backdrop = hex_rgb(background)
    mixed = tuple(
        round((1.0 - alpha) * backdrop[i] + alpha * foreground[i])
        for i in range(3)
    )
    return "#" + "".join(f"{channel:02X}" for channel in mixed)


def interpolate_color(stops: Sequence[tuple[float, str]], value: float) -> str:
    value = clamp(value)
    for (left_x, left_color), (right_x, right_color) in zip(stops, stops[1:]):
        if value <= right_x:
            ratio = 0.0 if right_x == left_x else (value - left_x) / (right_x - left_x)
            left = hex_rgb(left_color)
            right = hex_rgb(right_color)
            rgb = tuple(round(left[i] + ratio * (right[i] - left[i])) for i in range(3))
            return "#" + "".join(f"{channel:02X}" for channel in rgb)
    return stops[-1][1]


def load_jsonl_row(path: Path, marker: str) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if marker in line:
                return json.loads(line)
    raise KeyError(f"Could not locate {marker!r} in {path}")


def load_pca_payload() -> dict:
    source = REPORT.read_text(encoding="utf-8")
    match = re.search(
        r"const INDEXED_NUMERIC_N10=(.*?);\s*const INDEXED_NUMERIC_N10_VIEWS=",
        source,
        flags=re.S,
    )
    if match is None:
        raise RuntimeError("Could not locate INDEXED_NUMERIC_N10 in the report")
    return json.loads(match.group(1))["models"]["Qwen3-8B"]


def points_for_layer(
    payload: dict,
    mode: str,
    layer: str,
    *,
    split: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    all_points = payload[mode]["layers"][layer]["points"]
    observed_splits = {row["split"] for row in all_points}
    if len(all_points) != 300 or observed_splits != {"discovery", "confirmation"}:
        raise RuntimeError(
            f"Unexpected PCA source cohort for {mode} L{layer}: "
            f"{len(all_points)} points, splits={observed_splits}"
        )
    points = (
        [row for row in all_points if row["split"] == split]
        if split is not None else all_points
    )
    xyz = np.asarray([[row["x"], row["y"], row["z"]] for row in points], dtype=float)
    labels = np.asarray([row["occurrence"] for row in points], dtype=int)
    expected_size = 100 if split is not None else 300
    if xyz.shape != (expected_size, 3) or set(labels.tolist()) != set(range(1, 11)):
        raise RuntimeError(
            f"Unexpected PCA display cohort for {mode} L{layer}: "
            f"{xyz.shape}, split={split}"
        )
    return xyz, labels


def orient_component_signs(xyz: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, list[int]]:
    oriented = xyz.copy()
    signs: list[int] = []
    for component in range(3):
        correlation = float(np.corrcoef(oriented[:, component], labels)[0, 1])
        sign = -1 if np.isfinite(correlation) and correlation < 0 else 1
        oriented[:, component] *= sign
        signs.append(sign)
    return oriented, signs


def rotate_xyz(xyz: np.ndarray, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    x, y, z = xyz.T
    x1 = cy * x + sy * z
    z1 = -sy * x + cy * z
    y1 = cp * y - sp * z1
    z2 = sp * y + cp * z1
    return np.column_stack([x1, y1, z2])


def view_quality(xyz: np.ndarray, labels: np.ndarray, yaw: float, pitch: float) -> float:
    xy = rotate_xyz(xyz, yaw, pitch)[:, :2]
    centroids = np.asarray([xy[labels == k].mean(axis=0) for k in range(1, 11)])
    residual = xy - centroids[labels - 1]
    within = float(np.mean(np.sum(residual * residual, axis=1))) + 1e-12
    pairs = [
        float(np.sum((centroids[i] - centroids[j]) ** 2))
        for i in range(10)
        for j in range(i + 1, 10)
    ]
    return float(np.mean(pairs)) / within


def silhouette_score_numpy(points: np.ndarray, labels: np.ndarray) -> float:
    """Exact Euclidean silhouette score without a SciPy/sklearn dependency."""
    deltas = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    values: list[float] = []
    for index, label in enumerate(labels):
        same = labels == label
        same[index] = False
        a = float(distances[index, same].mean()) if np.any(same) else 0.0
        b = min(
            float(distances[index, labels == other].mean())
            for other in np.unique(labels)
            if other != label
        )
        denominator = max(a, b)
        values.append(0.0 if denominator == 0 else (b - a) / denominator)
    return float(np.mean(values))


def search_shared_view(non_xyz: np.ndarray, non_labels: np.ndarray,
                       native_xyz: np.ndarray, native_labels: np.ndarray) -> dict:
    # Rotate both modes together.  This is the clean shared view used in the
    # source report: PC2 is upright, while PC1 and PC3 recede symmetrically.
    # Camera choice changes only the rendering, never the held-out NCC values
    # or layer selection.
    yaw = math.degrees(-0.72)
    # Rotate the shared camera vertically in the direction that separates the
    # compact native centroids.  At this pitch only counts 6 and 7 retain a
    # sub-pixel edge contact; all other centroid disks are disjoint.
    pitch = math.degrees(-0.45)
    non_score = view_quality(non_xyz, non_labels, yaw, pitch)
    native_score = view_quality(native_xyz, native_labels, yaw, pitch)
    non_xy = rotate_xyz(non_xyz, yaw, pitch)[:, :2]
    native_xy = rotate_xyz(native_xyz, yaw, pitch)[:, :2]
    return {
        "yaw_deg": yaw,
        "pitch_deg": pitch,
        "camera_source": (
            "shared adjusted manifold view: yaw=-0.72 rad, pitch=-0.45 rad"
        ),
        "non_fisher_ratio": non_score,
        "native_fisher_ratio": native_score,
        "non_2d_silhouette": silhouette_score_numpy(non_xy, non_labels),
        "native_2d_silhouette": silhouette_score_numpy(native_xy, native_labels),
    }


@dataclass
class ProjectedCloud:
    points: np.ndarray
    centroids: np.ndarray
    depth: np.ndarray
    centroid_depth: np.ndarray
    origin: np.ndarray
    bounds: np.ndarray


def project_cloud(xyz: np.ndarray, labels: np.ndarray, *, yaw: float, pitch: float,
                  x: float, y: float, width: float, height: float,
                  display_scale: float = 1.0) -> ProjectedCloud:
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1)))) or 1.0
    rotated = rotate_xyz(centered / scale, yaw, pitch)
    centroids_3d = np.asarray([rotated[labels == k].mean(axis=0) for k in range(1, 11)])
    depth = rotated[:, 2]
    depth_scale = max(float(np.std(depth)), 1e-8)
    perspective = 1.0 / np.clip(1.0 + 0.090 * depth / depth_scale, 0.78, 1.22)
    projected_xy = rotated[:, :2] * perspective[:, None]
    centroid_perspective = 1.0 / np.clip(
        1.0 + 0.090 * centroids_3d[:, 2] / depth_scale, 0.78, 1.22
    )
    centroid_xy = centroids_3d[:, :2] * centroid_perspective[:, None]

    combined = np.vstack([projected_xy, centroid_xy])
    minimum = combined.min(axis=0)
    maximum = combined.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-8)
    available_w = width - 42
    available_h = height - 42
    fit = min(available_w / span[0], available_h / span[1])
    origin_x = x + (width - span[0] * fit) / 2 - minimum[0] * fit
    origin_y = y + (height - span[1] * fit) / 2 + maximum[1] * fit

    screen_points = np.column_stack(
        [origin_x + projected_xy[:, 0] * fit, origin_y - projected_xy[:, 1] * fit]
    )
    screen_centroids = np.column_stack(
        [origin_x + centroid_xy[:, 0] * fit, origin_y - centroid_xy[:, 1] * fit]
    )
    if display_scale <= 0.0:
        raise ValueError("display_scale must be positive")
    display_center = np.asarray([x + width / 2.0, y + height / 2.0], dtype=float)
    screen_points = display_center + display_scale * (screen_points - display_center)
    screen_centroids = display_center + display_scale * (
        screen_centroids - display_center
    )
    # Positive camera-z is farther away in the perspective equation above.
    # Store *nearness* so larger/darker marks are consistently drawn closer.
    normalized_depth = 1.0 - (
        (depth - depth.min()) / max(float(depth.max() - depth.min()), 1e-8)
    )
    centroid_depth = 1.0 - (
        (centroids_3d[:, 2] - depth.min()) / max(
        float(depth.max() - depth.min()), 1e-8
        )
    )
    return ProjectedCloud(
        screen_points,
        screen_centroids,
        normalized_depth,
        centroid_depth,
        np.asarray([origin_x, origin_y], dtype=float),
        np.asarray([x, y, width, height], dtype=float),
    )


class Drawio:
    def __init__(self) -> None:
        self._next_id = 2
        self.mxfile = ET.Element("mxfile", {"host": "Electron", "agent": "Codex"})
        self.diagram = ET.SubElement(
            self.mxfile,
            "diagram",
            {"id": "restrained-aurora-main", "name": "Page-1"},
        )
        self.model = ET.SubElement(
            self.diagram,
            "mxGraphModel",
            {
                "dx": "1900",
                "dy": "1200",
                "grid": "0",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "0",
                "page": "1",
                "pageScale": "1",
                "pageWidth": str(CANVAS_W),
                "pageHeight": str(CANVAS_H),
                "background": AURORA["white"],
                "math": "0",
                "shadow": "0",
            },
        )
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})

    def _id(self) -> str:
        value = str(self._next_id)
        self._next_id += 1
        return value

    def vertex(self, value: str, style: str, x: float, y: float,
               width: float, height: float) -> str:
        cell_id = self._id()
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": cell_id,
                "value": value,
                "style": style,
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": f"{x:.3f}",
                "y": f"{y:.3f}",
                "width": f"{width:.3f}",
                "height": f"{height:.3f}",
                "as": "geometry",
            },
        )
        return cell_id

    def line(self, x1: float, y1: float, x2: float, y2: float, *,
             color: str = AURORA["gray"], width: float = 1.0,
             opacity: int = 100, dashed: bool = False) -> str:
        cell_id = self._id()
        style = (
            f"edgeStyle=none;html=1;rounded=0;endArrow=none;startArrow=none;"
            f"strokeColor={color};strokeWidth={width};opacity={opacity};"
            f"dashed={1 if dashed else 0};"
        )
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {"id": cell_id, "value": "", "style": style, "edge": "1", "parent": "1"},
        )
        geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        ET.SubElement(geometry, "mxPoint", {"x": f"{x1:.3f}", "y": f"{y1:.3f}", "as": "sourcePoint"})
        ET.SubElement(geometry, "mxPoint", {"x": f"{x2:.3f}", "y": f"{y2:.3f}", "as": "targetPoint"})
        return cell_id

    def curve(self, points: Sequence[tuple[float, float]], *,
              color: str, width: float = 1.7, opacity: int = 92,
              arrow: bool = True) -> str:
        """Native draw.io curved edge through a small set of control points."""
        if len(points) < 2:
            raise ValueError("A curve needs at least a source and target point")
        cell_id = self._id()
        style = (
            "edgeStyle=none;html=1;curved=1;rounded=1;"
            f"endArrow={'classic' if arrow else 'none'};endFill=1;endSize=8;"
            f"startArrow=none;strokeColor={color};strokeWidth={width};"
            f"opacity={opacity};"
        )
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {"id": cell_id, "value": "", "style": style, "edge": "1", "parent": "1"},
        )
        geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        ET.SubElement(
            geometry, "mxPoint",
            {"x": f"{points[0][0]:.3f}", "y": f"{points[0][1]:.3f}", "as": "sourcePoint"},
        )
        ET.SubElement(
            geometry, "mxPoint",
            {"x": f"{points[-1][0]:.3f}", "y": f"{points[-1][1]:.3f}", "as": "targetPoint"},
        )
        if len(points) > 2:
            array = ET.SubElement(geometry, "Array", {"as": "points"})
            for px, py in points[1:-1]:
                ET.SubElement(array, "mxPoint", {"x": f"{px:.3f}", "y": f"{py:.3f}"})
        return cell_id

    def rect(self, x: float, y: float, width: float, height: float, *,
             fill: str = "none", stroke: str = "none", stroke_width: float = 1.0,
             rounded: bool = False, arc: float = 3, opacity: int = 100,
             dashed: bool = False) -> str:
        style = (
            f"rounded={1 if rounded else 0};arcSize={arc};html=1;whiteSpace=wrap;"
            f"fillColor={fill};strokeColor={stroke};strokeWidth={stroke_width};"
            f"opacity={opacity};dashed={1 if dashed else 0};"
        )
        return self.vertex("", style, x, y, width, height)

    def ellipse(self, x: float, y: float, width: float, height: float, *,
                fill: str, stroke: str = "none", stroke_width: float = 1.0,
                opacity: int = 100) -> str:
        style = (
            f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor={fill};"
            f"strokeColor={stroke};strokeWidth={stroke_width};opacity={opacity};"
        )
        return self.vertex("", style, x, y, width, height)

    def text(self, value: str, x: float, y: float, width: float, height: float, *,
             size: float = 14, color: str = AURORA["black"], bold: bool = False,
             align: str = "left", valign: str = "middle", html_label: bool = False,
             overflow: str = "hidden") -> str:
        rendered_size = size * FONT_SCALE
        style = (
            f"text;html={1 if html_label else 0};strokeColor=none;fillColor=none;"
            f"whiteSpace=wrap;overflow={overflow};rounded=0;align={align};"
            f"verticalAlign={valign};fontFamily=Times New Roman;fontSize={rendered_size:.3f};"
            f"fontColor={color};fontStyle={1 if bold else 0};spacing=0;"
        )
        return self.vertex(value, style, x, y, width, height)

    def write(self, path: Path) -> None:
        ET.indent(self.mxfile, space="  ")
        path.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            + ET.tostring(self.mxfile, encoding="unicode")
            + "\n",
            encoding="utf-8",
        )


def panel(draw: Drawio, label: str, title: str, x: float, y: float,
          width: float, height: float) -> None:
    # Keep the full 0.8-unit outer stroke inside the page while letting its
    # outer edge meet the crop boundary exactly.  This preserves a visible
    # four-sided frame without reintroducing decorative whitespace.
    border_inset = 0.4
    frame_x, frame_y = x, y
    frame_w, frame_h = width, height
    if abs(frame_x) < 1e-9:
        frame_x += border_inset
        frame_w -= border_inset
    if abs(frame_y) < 1e-9:
        frame_y += border_inset
        frame_h -= border_inset
    if abs(x + width - CANVAS_W) < 1e-9:
        frame_w -= border_inset
    if abs(y + height - CANVAS_H) < 1e-9:
        frame_h -= border_inset
    draw.rect(
        frame_x, frame_y, frame_w, frame_h,
        fill="none", stroke=blend(AURORA["gray"], 0.28), stroke_width=0.8,
        rounded=False,
    )
    draw.rect(x + 16, y + 12, 31, 31,
              fill=blend(AURORA["gray"], 0.12), stroke="none")
    draw.text(label, x + 16, y + 12, 31, 31, size=18.0, color=AURORA["black"],
              bold=True, align="center")
    draw.text(title, x + 59, y + 6, width - 78, 42, size=29.0, bold=True)


def clean_token_piece(piece: str) -> str:
    piece = (
        piece.replace("\u2029", " ")
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("\t", " ")
        .replace("\ufffd", "")
    )
    escaped = html.escape(piece)
    while escaped.startswith(" "):
        escaped = "&nbsp;" + escaped[1:]
    return escaped or "&#8203;"


def token_line(
    tokenizer: Tokenizer,
    input_ids: Sequence[int],
    indices: Iterable[int],
    colors: dict[int, str],
    *,
    default_fill: str,
    prefix: str = "… ",
    suffix: str = " …",
) -> str:
    spans = [html.escape(prefix)]
    for index in indices:
        piece = tokenizer.decode([int(input_ids[index])], skip_special_tokens=False)
        fill = colors.get(index, default_fill)
        spans.append(
            f'<span style="background-color:{fill};border-radius:2px;'
            f'padding:0px 1px;border-right:0.45px solid {AURORA["white"]};'
            f'box-decoration-break:clone;">{clean_token_piece(piece)}</span>'
        )
    spans.append(html.escape(suffix))
    return "".join(spans)


def axis_directions(yaw: float, pitch: float) -> np.ndarray:
    basis = np.eye(3)
    projected = rotate_xyz(basis, yaw, pitch)[:, :2]
    lengths = np.linalg.norm(projected, axis=1)
    lengths[lengths == 0] = 1.0
    return projected / lengths[:, None]


def clip_segment_to_rect(
    start: np.ndarray,
    end: np.ndarray,
    bounds: Sequence[float],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip a two-dimensional segment to an axis-aligned rectangle."""
    x, y, width, height = map(float, bounds)
    xmin, xmax = x, x + width
    ymin, ymax = y, y + height
    delta = end - start
    t_min, t_max = 0.0, 1.0
    for p, q in (
        (-delta[0], start[0] - xmin),
        (delta[0], xmax - start[0]),
        (-delta[1], start[1] - ymin),
        (delta[1], ymax - start[1]),
    ):
        if abs(float(p)) < 1e-12:
            if q < 0.0:
                return None
            continue
        ratio = float(q / p)
        if p < 0.0:
            t_min = max(t_min, ratio)
        else:
            t_max = min(t_max, ratio)
        if t_min > t_max:
            return None
    return start + t_min * delta, start + t_max * delta


def draw_3d_grid_behind_manifold(
    draw: Drawio, cloud: ProjectedCloud, yaw: float, pitch: float
) -> None:
    """Draw a sparse symmetric weak-perspective grid behind the data.

    Each family is exactly parallel in screen space to one labeled PC axis.
    Offset lines occur in matched positive/negative pairs at equal projected
    spacing, which is the correct construction for the shared axonometric
    (weak-perspective) view used here.  The central member of every family
    passes through the visible origin underneath the dark axis.
    """
    grid_color = blend(AURORA["gray"], 0.30)
    plot_x, plot_y, plot_w, plot_h = map(float, cloud.bounds)
    origin = np.asarray(
        [plot_x + plot_w / 2.0, plot_y + plot_h / 2.0], dtype=float
    )
    directions = axis_directions(yaw, pitch)
    screen_directions = np.column_stack([directions[:, 0], -directions[:, 1]])
    line_reach = math.hypot(plot_w, plot_h) * 0.78
    base_spacing = min(plot_w, plot_h) * 0.125
    max_step = 3

    for direction in screen_directions:
        direction = direction / max(float(np.linalg.norm(direction)), 1e-8)
        normal = np.asarray([-direction[1], direction[0]], dtype=float)
        if normal[1] > 0.0:
            normal *= -1.0
        for step in range(-max_step, max_step + 1):
            anchor = origin + (base_spacing * step) * normal
            clipped = clip_segment_to_rect(
                anchor - line_reach * direction,
                anchor + line_reach * direction,
                cloud.bounds,
            )
            if clipped is None:
                continue
            start_2d, end_2d = clipped
            centrality = 1.0 - abs(step) / (max_step + 0.5)
            draw.line(
                float(start_2d[0]), float(start_2d[1]),
                float(end_2d[0]), float(end_2d[1]),
                color=grid_color,
                width=0.40 + 0.08 * centrality,
                opacity=round(20 + 14 * centrality),
            )


def draw_axes_behind_manifold(
    draw: Drawio, cloud: ProjectedCloud, yaw: float, pitch: float
) -> None:
    """Draw a shared-camera PCA triad underneath the marks, screen-centred."""
    draw_3d_grid_behind_manifold(draw, cloud, yaw, pitch)
    directions = axis_directions(yaw, pitch) * np.asarray(
        PCA_DISPLAY_AXIS_SIGNS, dtype=float
    )[:, None]
    axis_color = MUTED_TEXT
    labels = ["PC1", "PC2", "PC3"]
    plot_x, plot_y, plot_w, plot_h = map(float, cloud.bounds)
    origin_x, origin_y = plot_x + plot_w / 2.0, plot_y + plot_h / 2.0
    axis_lengths = [170.0, 178.0, 170.0]
    for direction, label, length in zip(
        directions, labels, axis_lengths
    ):
        dx = float(direction[0] * length)
        dy = float(-direction[1] * length)
        endpoint = (origin_x + dx, origin_y + dy)
        draw.curve(
            [(origin_x, origin_y), endpoint],
            color=axis_color, width=1.45, opacity=74, arrow=True,
        )
        label_x = endpoint[0] + (7 if dx >= 0 else -43)
        label_y = endpoint[1] + (4 if dy >= 0 else -21)
        if label == "PC2":
            # Place the vertical-axis label just below/right of the arrowhead;
            # this keeps it inside the plot instead of colliding with NCC.
            label_x = endpoint[0] + 7
            label_y = endpoint[1] + 5
        draw.text(label, label_x, label_y, 46, 25, size=16.0,
                  color=axis_color, bold=True,
                  align="left" if dx >= 0 or label == "PC2" else "right")
    draw.ellipse(origin_x - 3.5, origin_y - 3.5, 7.0, 7.0,
                 fill=axis_color, opacity=80)


def draw_cloud(draw: Drawio, cloud: ProjectedCloud, labels: np.ndarray, *,
               yaw: float, pitch: float) -> None:
    draw_axes_behind_manifold(draw, cloud, yaw, pitch)

    # The centroid path is part of the manifold geometry.  Paint its far
    # segments first, then the state clouds and centroid nodes from far to near,
    # so foreground marks naturally occlude background structure.
    path_segments: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    for index, (first, second) in enumerate(zip(cloud.centroids[:-1], cloud.centroids[1:])):
        segment_depth = clamp(float(
            0.5 * (cloud.centroid_depth[index] + cloud.centroid_depth[index + 1])
        ))
        path_segments.append((segment_depth, index, first, second))
    for segment_depth, _index, first, second in sorted(
        path_segments, key=lambda item: item[0]
    ):
        draw.line(first[0], first[1], second[0], second[1],
                  color=AURORA["black"],
                  width=0.90 + 1.95 * segment_depth,
                  opacity=round(28 + 67 * segment_depth))

    for index in np.argsort(cloud.depth):
        px, py = cloud.points[index]
        diameter = PCA_STATE_DIAMETER
        color = COUNT_COLORS[int(labels[index])]
        draw.ellipse(px - diameter / 2, py - diameter / 2, diameter, diameter,
                     fill=color, opacity=PCA_STATE_OPACITY)

    centroid_order = np.argsort(cloud.centroid_depth)
    for centroid_index in centroid_order:
        count = int(centroid_index) + 1
        point = cloud.centroids[centroid_index]
        px, py = map(float, point)
        diameter = PCA_CENTROID_DIAMETER
        draw.ellipse(px - diameter / 2, py - diameter / 2,
                     diameter, diameter, fill=COUNT_COLORS[count],
                     stroke=AURORA["black"],
                     stroke_width=1.0)
        label_size = 10.4
        # Use exactly the same bounding box as the circle.  With zero spacing,
        # horizontal centering, and verticalAlign=middle, the numeral centre
        # is geometrically identical to the centroid marker centre.
        draw.text(str(count), px - diameter / 2, py - diameter / 2,
                  diameter, diameter, size=label_size,
                  color="#FFFFFF", bold=True,
                  align="center", valign="middle")

def load_native_example() -> tuple[dict, Tokenizer]:
    marker = f"T10000_N10_seed{SELECTED_SEED}"
    row = load_jsonl_row(NATIVE_ROWS, marker)
    if int(row["gold_count"]) != 10:
        raise RuntimeError("The selected native example is not an N=10 row")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_JSON))
    return row, tokenizer


def trace_item_text(row: dict, occurrence: int) -> str:
    """Return the exact primary item span from the stored model trace."""
    site = next(
        site for site in row["trace_parse"]["char_sites"]
        if site["site_id"] == f"item_end:{occurrence}" and site.get("primary")
    )
    return row["raw_output_text"][int(site["char_start"]):int(site["char_end"])].strip()


def load_nonthinking_attention(native_row: dict) -> dict:
    with np.load(NONTHINK_ATTN) as payload:
        key = f"layer_{NONTHINK_LAYER:03d}"
        attention = np.asarray(payload[key][NONTHINK_HEAD], dtype=float)
        key_start = int(np.asarray(payload["key_starts"]).reshape(-1)[NONTHINK_LAYER])
        query_position = int(np.asarray(payload["query_position"]).reshape(-1)[0])
        sequence_length = int(np.asarray(payload["sequence_length"]).reshape(-1)[0])

    spans = native_row["prompt_record_spans"]
    registered_spans = [
        [int(span["start"]), int(span["end"])] for span in spans
    ]
    if len(registered_spans) != 10:
        raise RuntimeError("Expected ten registered prompt-record spans")
    if sequence_length != int(attention.shape[0]) or query_position + 1 != sequence_length:
        raise RuntimeError("Non-thinking capture length metadata is inconsistent")
    if any(start < key_start or end > key_start + len(attention) or start >= end
           for start, end in registered_spans):
        raise RuntimeError("Registered prompt-record spans fall outside the capture")
    capture_manifest = {
        "source": str(NONTHINK_ATTN),
        "seed": SELECTED_SEED,
        "prompt_token_count": len(native_row["input_ids"]),
        "sequence_length": sequence_length,
        "needle_spans": registered_spans,
        "span_registry": "native generation prompt_record_spans",
    }
    masses: list[float] = []
    for span in spans:
        local_start = max(0, int(span["start"]) - key_start)
        local_end = min(attention.shape[0], int(span["end"]) - key_start)
        masses.append(float(attention[local_start:local_end].sum()))

    needle_total = float(sum(masses))
    official_row: dict[str, str] | None = None
    with gzip.open(NONTHINK_HEAD_SUMMARY, "rt", encoding="utf-8", newline="") as handle:
        for candidate in csv.DictReader(handle):
            if (
                int(candidate["layer"]) == NONTHINK_LAYER
                and int(candidate["head"]) == NONTHINK_HEAD
            ):
                official_row = candidate
                break
    if official_row is None:
        raise RuntimeError("Selected non-thinking head is absent from the summary shard")
    official_masses = [
        float(value) for value in json.loads(official_row["needle_span_masses"])
    ]
    official_max_abs_delta = max(
        abs(observed - expected)
        for observed, expected in zip(masses, official_masses)
    )
    if official_max_abs_delta > 5e-6:
        raise RuntimeError(
            "Raw non-thinking attention disagrees with the archived span summary: "
            f"max |Δ|={official_max_abs_delta:.3e}"
        )
    proportions = np.asarray(masses, dtype=float) / max(needle_total, 1e-12)
    entropy = float(
        -(proportions * np.log(np.maximum(proportions, 1e-12))).sum() / math.log(10)
    )
    behavior = json.loads(NONTHINK_BEHAVIOR.read_text(encoding="utf-8"))
    return {
        "attention": attention,
        "key_start": key_start,
        "query_position": query_position,
        "record_masses": masses,
        "needle_total_mass": needle_total,
        "normalized_record_entropy": entropy,
        "answer": behavior["full_answer_text"].replace("Total:", "Total: "),
        "answer_prefix": str(behavior["answer_prefix"]),
        "is_correct": bool(behavior["is_correct"]),
        "capture_manifest": capture_manifest,
        "official_span_mass_max_abs_delta": official_max_abs_delta,
    }


def load_fixed_head_events() -> list[dict]:
    capture = json.loads(NATIVE_TRANSITIONS.read_text(encoding="utf-8"))
    if capture.get("schema_version") != "aurora_native_fixed_head_transitions_v1":
        raise RuntimeError("Unexpected native fixed-head transition schema")
    if (
        int(capture["seed"]) != SELECTED_SEED
        or int(capture["layer"]) != NATIVE_LAYER
        or int(capture["head"]) != NATIVE_HEAD
    ):
        raise RuntimeError("Native fixed-head transition metadata does not match the figure")
    by_source = {
        int(event["from_occurrence"]): event for event in capture["events"]
    }

    events: list[dict] = []
    for occurrence in range(0, 10):
        if occurrence not in by_source:
            raise RuntimeError(f"Missing native endpoint row {occurrence}→{occurrence + 1}")
        event = dict(by_source[occurrence])
        record_masses = np.asarray([float(row["mass"]) for row in event["records"]])
        needle_total = float(record_masses.sum())
        relative = record_masses / max(needle_total, 1e-12)
        target_index = int(event["to_occurrence"]) - 1
        event["needle_total_mass"] = needle_total
        event["relative_record_masses"] = relative.tolist()
        event["target_raw_mass"] = float(record_masses[target_index])
        event["target_share"] = float(relative[target_index])
        events.append(event)
    return events


def load_native_token_attention(native_row: dict) -> dict:
    metadata = json.loads(NATIVE_TOKEN_ATTN_META.read_text(encoding="utf-8"))
    with np.load(NATIVE_TOKEN_ATTN) as payload:
        prompt_input_ids = np.asarray(payload["prompt_input_ids"], dtype=np.int32)
        prompt_attention = np.asarray(payload["prompt_attention"], dtype=float)
        from_occurrences = np.asarray(payload["from_occurrences"], dtype=int)
        to_occurrences = np.asarray(payload["to_occurrences"], dtype=int)
        query_full = np.asarray(payload["query_full_sequence_tokens"], dtype=int)
        query_output = np.asarray(payload["query_output_token_indices"], dtype=int)
        record_spans = np.asarray(payload["record_spans"], dtype=int)

    if metadata.get("schema_version") != "aurora_native_token_attention_v1":
        raise RuntimeError("Unexpected native token-attention schema")
    if (
        int(metadata["seed"]) != SELECTED_SEED
        or int(metadata["layer"]) != NATIVE_LAYER
        or int(metadata["head"]) != NATIVE_HEAD
    ):
        raise RuntimeError("Native token-attention metadata does not match the figure")
    if prompt_attention.shape != (3, int(metadata["prompt_token_count"])):
        raise RuntimeError(
            f"Unexpected native token-attention shape: {prompt_attention.shape}"
        )
    if prompt_input_ids.shape != (int(metadata["prompt_token_count"]),):
        raise RuntimeError("Native prompt-token array length mismatch")
    if not np.array_equal(
        prompt_input_ids,
        np.asarray(native_row["input_ids"], dtype=np.int32),
    ):
        raise RuntimeError("Native token capture does not match the displayed prompt IDs")
    if record_spans.shape != (10, 2):
        raise RuntimeError(f"Unexpected native record-span shape: {record_spans.shape}")

    events = metadata["events"]
    for row_index, event in enumerate(events):
        source = int(event["from_occurrence"])
        target = int(event["to_occurrence"])
        if (
            source != int(from_occurrences[row_index])
            or target != int(to_occurrences[row_index])
            or int(event["query_full_sequence_token"]) != int(query_full[row_index])
            or int(event["query_output_token_index"]) != int(query_output[row_index])
        ):
            raise RuntimeError(f"Native token-attention event {row_index} is misaligned")
        start, end = map(int, record_spans[target - 1])
        observed = float(prompt_attention[row_index, start:end].sum())
        expected = float(event["target_mass"])
        if abs(observed - expected) > 5e-6:
            raise RuntimeError(
                f"Native token-attention target mass mismatch for {source}→{target}: "
                f"{observed:.8f} vs {expected:.8f}"
            )
    return {
        "metadata": metadata,
        "input_ids": prompt_input_ids,
        "attention": prompt_attention,
        "events": events,
        "record_spans": metadata["record_spans"],
    }


def excerpt_html(
    tokenizer: Tokenizer,
    input_ids: Sequence[int],
    spans: Sequence[dict],
    occurrences: Sequence[int],
    colors: dict[int, str],
    *,
    default_fill: str,
    context_tokens: int = 8,
) -> str:
    def decoded_units(indices: Sequence[int]) -> list[tuple[list[int], str]]:
        units: list[tuple[list[int], str]] = []
        buffered: list[int] = []
        for index in indices:
            buffered.append(index)
            piece = tokenizer.decode(
                [int(input_ids[item]) for item in buffered],
                skip_special_tokens=False,
            )
            if "\ufffd" in piece and len(buffered) < 4:
                continue
            units.append((list(buffered), piece))
            buffered = []
        if buffered:
            piece = tokenizer.decode(
                [int(input_ids[item]) for item in buffered],
                skip_special_tokens=False,
            ).replace("\ufffd", "")
            if piece:
                units.append((list(buffered), piece))
        return units

    def render_indices(indices: Sequence[int]) -> str:
        rendered: list[str] = []
        for unit_indices, piece in decoded_units(indices):
            fill = next(
                (colors[index] for index in unit_indices if index in colors),
                default_fill,
            )
            rendered.append(
                f'<wbr/><span style="background-color:{fill};border-radius:2px;'
                f'padding:1px 2px 1px 1px;border-right:0.7px solid {AURORA["white"]};'
                f'box-decoration-break:clone;">'
                f"{clean_token_piece(piece)}</span>"
            )
        return "".join(rendered)

    parts: list[str] = []
    for offset, occurrence in enumerate(occurrences):
        span = spans[occurrence - 1]
        span_start, span_end = int(span["start"]), int(span["end"])
        before = list(range(max(0, span_start - context_tokens), span_start))
        after = list(range(span_end, min(len(input_ids), span_end + context_tokens)))

        sentence = (
            f'In the 2024 city score audit, {span["city"]} received a score of '
            f'{span["score"]}.'
        )
        canonical = tokenizer.encode(sentence, add_special_tokens=False).ids
        # The stored span ends the sentence with a period+newline token, whereas
        # stand-alone encoding uses a bare period. Match the stable prefix and
        # retain the original final token to preserve exact attention indices.
        prefix = canonical[:-1]
        span_ids = list(input_ids[span_start:span_end])
        relative_start = -1
        for candidate in range(len(span_ids) - len(prefix) + 1):
            if span_ids[candidate:candidate + len(prefix)] == prefix:
                relative_start = candidate
                break
        if relative_start < 0:
            raise RuntimeError(f'Could not locate canonical sentence for N{occurrence}')
        sentence_start = span_start + relative_start
        sentence_indices = list(
            range(sentence_start, min(span_end, sentence_start + len(prefix) + 1))
        )

        if offset == 0:
            parts.append("… ")
        else:
            parts.append(
                f'<span style="color:{MUTED_TEXT};background-color:transparent;">'
                "<br/>…&nbsp;</span>"
            )
        parts.append(render_indices(before))
        parts.append(render_indices(sentence_indices))
        parts.append(render_indices(after))
    parts.append(" …")
    return "".join(parts)


def draw_rule(draw: Drawio, x: float, y: float, width: float) -> None:
    draw.line(x, y, x + width, y, color=blend(AURORA["gray"], 0.29), width=0.75)


def draw_attention_scale(
    draw: Drawio,
    x: float,
    y: float,
    *,
    label: str,
    hues: Sequence[str],
    opacity_scale: float = 1.0,
) -> None:
    """Compact shared low-to-high opacity key for the two prompt demos."""
    draw.text(label, x, y, 190, 22, size=14.5,
              color=MUTED_TEXT, align="right")
    draw.text("low", x + 196, y, 25, 22, size=13.0,
              color=MUTED_TEXT, align="center")
    swatch_x = x + 226
    opacities = tuple(
        opacity_scale * opacity for opacity in (0.004, 0.16, 0.50, 0.95)
    )
    for column, opacity in enumerate(opacities):
        cell_x = swatch_x + column * 15
        band_height = 14 / len(hues)
        for band, hue in enumerate(hues):
            draw.rect(
                cell_x, y + 4 + band * band_height, 12, band_height + 0.25,
                fill=blend(hue, opacity), stroke="none",
            )
    draw.text("high", x + 287, y, 29, 22, size=13.0,
              color=MUTED_TEXT, align="center")


def question_block(native: bool) -> str:
    del native
    return "How many city-score audit records are in the passage?"


def draw_prompt_demo_a(draw: Drawio, row: dict, tokenizer: Tokenizer,
                       nonthinking: dict) -> None:
    x, y, width, height = LEFT_X, LEFT_A_Y, COL_W, LEFT_A_H
    panel(draw, "A", "Non-thinking · broad retrieval at final query",
          x, y, width, height)
    draw.text("INPUT · PROMPT", x + 24, y + 54, 180, 24, size=18.0,
              color=AURORA["black"], bold=True)
    draw_attention_scale(
        draw, x + 545, y + 54,
        label="opacity = raw attention mass", hues=[AURORA["red"]],
        opacity_scale=NONTHINK_PROMPT_ALPHA_SCALE,
    )
    draw_rule(draw, x + 24, y + 84, width - 48)

    attention = nonthinking["attention"]
    key_start = int(nonthinking["key_start"])
    positive = attention[attention > 0]
    robust_high = float(np.quantile(positive, 0.997)) if positive.size else 1.0
    colors: dict[int, str] = {}
    spans = row["prompt_record_spans"]
    for occurrence in DISPLAYED_OCCURRENCES:
        span = spans[occurrence - 1]
        for index in range(
            max(0, int(span["start"]) - 65),
            min(len(row["input_ids"]), int(span["end"]) + 65),
        ):
            local = index - key_start
            value = attention[local] if 0 <= local < attention.shape[0] else 0.0
            normalized = clamp(float(value) / max(robust_high, 1e-12))
            alpha = NONTHINK_PROMPT_ALPHA_SCALE * token_attention_alpha(normalized)
            colors[index] = blend(AURORA["red"], alpha)

    label_width = (width - 48) / len(DISPLAYED_OCCURRENCES)
    for label_index, occurrence in enumerate(DISPLAYED_OCCURRENCES):
        record = row["gold_records"][occurrence - 1]
        label_x = x + 24 + label_index * label_width
        draw.ellipse(label_x, y + 101, 10, 10, fill=AURORA["red"])
        draw.text(
            f'N{occurrence} · {record["city"]} · '
            f'mass {float(nonthinking["record_masses"][occurrence - 1]) * 100:.2f}%',
            label_x + 15, y + 92, label_width - 19, 28,
            size=17.0, color=AURORA["black"], bold=True,
        )

    passage = excerpt_html(
        tokenizer, row["input_ids"], spans, DISPLAYED_OCCURRENCES, colors,
        default_fill=AURORA["white"], context_tokens=11,
    )
    draw.text(
        f'<div style="font-family:Times New Roman,serif;font-size:22px;line-height:1.13;'
        f'color:{AURORA["black"]};">{passage}</div>',
        x + 24, y + 124, width - 48, 182, size=20.5,
        html_label=True, valign="top", overflow="hidden",
    )
    draw_rule(draw, x + 24, y + 314, width - 48)
    draw.text("QUESTION", x + 24, y + 314, 112, 38, size=18.0,
              color=AURORA["black"], bold=True)
    draw.text(question_block(False), x + 144, y + 314, width - 168, 38,
              size=21.5, color=AURORA["black"], valign="middle")
    draw_rule(draw, x + 24, y + 352, width - 48)
    draw.text("OUTPUT · ANSWER", x + 24, y + 352, 190, 38, size=18.0,
              color=AURORA["black"], bold=True)
    answer_prefix = str(nonthinking["answer_prefix"])
    answer_suffix = str(nonthinking["answer"])
    if not answer_suffix.startswith(answer_prefix):
        raise RuntimeError("Non-thinking answer does not begin with its captured prefix")
    answer_suffix = answer_suffix[len(answer_prefix):].strip()
    # The captured attention row is emitted by the final token of the forced
    # answer prefix.  Put the readout rule immediately after that exact prefix,
    # before the generated integer, rather than at an approximate x coordinate.
    answer_text = (
        f'<span style="color:{AURORA["black"]};border-right:3px solid '
        f'{AURORA["red"]};padding-right:4px;">'
        f'{html.escape(answer_prefix)}</span>&nbsp;&nbsp;'
        f'<span style="color:{AURORA["black"]};">'
        f'{html.escape(answer_suffix)}</span>'
    )
    if not nonthinking["is_correct"]:
        answer_text += (
            f'&nbsp;&nbsp;<span style="color:{AURORA["red"]};">(Incorrect)</span>'
        )
    draw.text(answer_text, x + 224, y + 352, 330, 38, size=21.5,
              color=AURORA["black"], bold=False, html_label=True)
    draw.text("prefix end = head readout", x + width - 290, y + 352,
              266, 38, size=13.5, color=MUTED_TEXT, align="right")


def draw_attention_strip_c(draw: Drawio, row: dict, nonthinking: dict) -> None:
    x, y, width, height = LEFT_X, LEFT_B_Y, COL_W, LEFT_B_H
    panel(
        draw, "C", "Non-thinking · final-query attention",
        x, y, width, height,
    )
    # Report-style full-prompt strip: one shallow attention row, not a second
    # text demo or a tall heatmap.  The quiet vertical margins preserve row
    # alignment with panel D while keeping the quantitative mark visually thin.
    plot_x, plot_y, plot_w, plot_h = x + 54, y + 98, width - 92, 44
    draw.rect(plot_x, plot_y, plot_w, plot_h, fill=blend(AURORA["gray"], 0.08),
              stroke=blend(AURORA["gray"], 0.35), stroke_width=0.7)
    attention = np.asarray(nonthinking["attention"], dtype=float)
    n_bins = 170
    edges = np.linspace(0, attention.shape[0], n_bins + 1, dtype=int)
    binned = np.asarray([attention[edges[i]:edges[i + 1]].sum() for i in range(n_bins)])
    high = float(np.quantile(binned, 0.985)) or 1.0
    cell_width = plot_w / n_bins
    for index, value in enumerate(binned):
        intensity = math.sqrt(clamp(float(value) / high))
        draw.rect(
            plot_x + index * cell_width, plot_y, cell_width + 0.18, plot_h,
            fill=blend(AURORA["red"], 0.035 + 0.87 * intensity), opacity=100,
        )

    sequence_length = attention.shape[0]
    for span in row["prompt_record_spans"]:
        center = 0.5 * (int(span["start"]) + int(span["end"]))
        px = plot_x + plot_w * center / sequence_length
        draw.line(px, plot_y - 8, px, plot_y + plot_h + 8,
                  color=AURORA["black"], width=1.05, opacity=78)
        draw.text(f'N{span["slot_index"]}', px - 16, plot_y - 31, 32, 20,
                  size=18.0, color=AURORA["black"], bold=True, align="center")

    draw.text("0", plot_x - 8, plot_y + plot_h + 8, 30, 22, size=15.5,
              color=MUTED_TEXT)
    draw.text("prompt position", plot_x, plot_y + plot_h + 8, plot_w, 20,
              size=16.5, color=MUTED_TEXT, align="center")
    draw.text("10k tokens", plot_x + plot_w - 88, plot_y + plot_h + 8, 94, 20,
              size=15.5, color=MUTED_TEXT, align="right")


def draw_pca_e(
    draw: Drawio,
    non_xyz: np.ndarray,
    non_labels: np.ndarray,
    native_xyz: np.ndarray,
    native_labels: np.ndarray,
    view: dict,
    non_ncc: float,
    native_ncc: float,
) -> None:
    x, y, width, height = LEFT_X, LEFT_C_Y, COL_W, LEFT_C_H
    panel(draw, "E", "Count-state geometry under a shared 3D view",
          x, y, width, height)
    # Leave a deliberate header-to-plot gutter so PC2 never collides with the
    # left/right subtitle and NCC row at paper scale.
    left_plot = (x + 10, y + 92, 442, 486)
    right_plot = (x + 472, y + 92, 442, 486)
    draw.text("Non-thinking", left_plot[0] + 8, y + 52, 250, 30,
              size=22.0, color=AURORA["black"], bold=True, align="left")
    draw.text("Native-thinking", right_plot[0] + 8, y + 52, 250, 30,
              size=22.0, color=AURORA["black"], bold=True, align="left")
    draw.text(
        f"NCC Accuracy: {non_ncc * 100:.0f}%", left_plot[0] + 236, y + 52,
        left_plot[2] - 244, 30,
        size=16.5, color=AURORA["black"], bold=True, align="right",
    )
    draw.text(
        f"NCC Accuracy: {native_ncc * 100:.0f}%", right_plot[0] + 236, y + 52,
        right_plot[2] - 244, 30,
        size=16.5, color=AURORA["black"], bold=True, align="right",
    )
    draw.line(x + width / 2, y + 92, x + width / 2, y + height - 14,
              color=blend(AURORA["gray"], 0.28), width=0.75)

    yaw, pitch = float(view["yaw_deg"]), float(view["pitch_deg"])
    non_cloud = project_cloud(
        non_xyz, non_labels, yaw=yaw, pitch=pitch,
        x=left_plot[0], y=left_plot[1], width=left_plot[2], height=left_plot[3],
        display_scale=PCA_NONTHINK_DISPLAY_SCALE,
    )
    native_cloud = project_cloud(
        native_xyz, native_labels, yaw=yaw, pitch=pitch,
        x=right_plot[0], y=right_plot[1], width=right_plot[2], height=right_plot[3],
    )
    draw_cloud(draw, non_cloud, non_labels, yaw=yaw, pitch=pitch)
    draw_cloud(draw, native_cloud, native_labels, yaw=yaw, pitch=pitch)


def draw_prompt_demo_b(draw: Drawio, row: dict, tokenizer: Tokenizer,
                       token_attention: dict) -> None:
    x, y, width, height = RIGHT_X, RIGHT_D_Y, COL_W, RIGHT_D_H
    panel(draw, "B", "Native-thinking · targeted retrieval loops in thinking trace",
          x, y, width, height)
    draw.text("INPUT · PROMPT", x + 24, y + 54, 180, 24, size=18.0,
              color=AURORA["black"], bold=True)
    draw_attention_scale(
        draw, x + 545, y + 54,
        label="opacity = raw attention mass",
        hues=[AURORA["blue"], AURORA["green"], AURORA["yellow"]],
    )
    draw_rule(draw, x + 24, y + 84, width - 48)

    event_by_target = {
        int(event["to_occurrence"]): (row_index, event)
        for row_index, event in enumerate(token_attention["events"])
    }
    hues = dict(zip(
        NATIVE_DISPLAYED_OCCURRENCES,
        (AURORA["blue"], AURORA["green"], AURORA["yellow"]),
    ))
    spans = token_attention["record_spans"]
    input_ids = token_attention["input_ids"]
    attention_rows = token_attention["attention"]
    display_values: list[float] = []
    for occurrence in NATIVE_DISPLAYED_OCCURRENCES:
        row_index, _event = event_by_target[occurrence]
        span = spans[occurrence - 1]
        start = max(0, int(span["start"]) - 14)
        end = min(len(input_ids), int(span["end"]) + 14)
        display_values.extend(float(value) for value in attention_rows[row_index, start:end])
    positive_values = np.asarray([value for value in display_values if value > 0], dtype=float)
    robust_high = (
        float(np.quantile(positive_values, 0.98)) if positive_values.size else 1.0
    )
    line_ys = [y + 94, y + 168, y + 242]
    for occurrence, line_y in zip(NATIVE_DISPLAYED_OCCURRENCES, line_ys):
        span = spans[occurrence - 1]
        row_index, _event = event_by_target[occurrence]
        colors: dict[int, str] = {}
        display_start = max(0, int(span["start"]) - 14)
        display_end = min(len(input_ids), int(span["end"]) + 14)
        for index in range(display_start, display_end):
            value = float(attention_rows[row_index, index])
            normalized = clamp(value / max(robust_high, 1e-12))
            alpha = token_attention_alpha(normalized)
            colors[index] = blend(hues[occurrence], alpha)
        draw.ellipse(x + 24, line_y + 28, 12, 12, fill=hues[occurrence])
        draw.text(
            f"{int(_event['from_occurrence'])}→{int(_event['to_occurrence'])}"
            f"  ·  N{occurrence}<br>"
            f"mass {float(_event['target_mass']) * 100:.1f}%",
            x + 44, line_y + 4, 92, 60,
            size=16.5, color=AURORA["black"], bold=True,
            html_label=True, valign="middle",
        )
        passage = excerpt_html(
            tokenizer, input_ids, spans, [occurrence], colors,
            default_fill=AURORA["white"], context_tokens=8,
        )
        draw.text(
            f'<div style="font-family:Times New Roman,serif;font-size:22px;line-height:1.13;'
            f'color:{AURORA["black"]};">{passage}</div>',
            x + 142, line_y, width - 172, 72, size=20.5,
            html_label=True, valign="top", overflow="hidden",
        )

    draw_rule(draw, x + 24, y + 321, width - 48)
    draw.text("QUESTION", x + 24, y + 321, 112, 42, size=18.0,
              color=AURORA["black"], bold=True)
    draw.text(question_block(True), x + 144, y + 321, width - 168, 42,
              size=21.5, color=AURORA["black"], valign="middle")
    draw_rule(draw, x + 24, y + 363, width - 48)
    draw.text("OUTPUT · THINKING", x + 24, y + 363, 220, 39, size=18.0,
              color=AURORA["black"], bold=True)
    draw.text("item end = head readout", x + width - 240, y + 363, 216, 39,
              size=13.5, color=MUTED_TEXT, align="right")

    trace_rows: list[tuple[str, str]] = []
    for trace_index, target_occurrence in enumerate(NATIVE_DISPLAYED_OCCURRENCES):
        source_occurrence = target_occurrence - 1
        text_value = trace_item_text(row, source_occurrence)
        if trace_index == 0:
            text_value = f"<think>  …  {text_value}"
        trace_rows.append((text_value, hues[target_occurrence]))
    for row_index, (text_value, hue) in enumerate(trace_rows):
        row_y = y + 402 + 40 * row_index
        marker_x = x + width - 54
        draw.rect(x + 24, row_y, width - 78, 36,
                  fill=blend(hue, 0.065), stroke="none")
        draw.text(html.escape(text_value), x + 34, row_y, width - 104, 36,
                  size=20.5, color=AURORA["black"], html_label=True)
        draw.rect(marker_x, row_y - 2, 4, 40, fill=hue, stroke="none")
        draw.text("end", marker_x + 7, row_y, 34, 36,
                  size=12.5, color=hue, bold=True)

    draw.text(
              f"…  {html.escape(trace_item_text(row, NATIVE_DISPLAYED_OCCURRENCES[-1]))}"
              f"  ·  …  ·  "
              f"{html.escape(trace_item_text(row, 10))}  …  </think>",
              x + 24, y + 526, width - 48, 22, size=17.5,
              color=MUTED_TEXT, html_label=True)
    draw_rule(draw, x + 24, y + 552, width - 48)
    draw.text("OUTPUT · ANSWER", x + 24, y + 552, 190, 48, size=18.0,
              color=AURORA["black"], bold=True)
    draw.text("Total: 10", x + 224, y + 552, 230, 48, size=21.5,
              color=AURORA["black"], bold=False)


def draw_matrix_d(
    draw: Drawio,
    row: dict,
    events: Sequence[dict],
) -> None:
    x, y, width, height = RIGHT_X, RIGHT_E_Y, COL_W, RIGHT_E_H
    panel(
        draw, "D", "Native-thinking · targeted-retrieval attention",
        x, y, width, height,
    )
    row_label_x = x + 24
    row_label_w = 160
    draw.text("prompt records", row_label_x, y + 76, row_label_w, 28, size=18.0,
              color=MUTED_TEXT, align="right")
    cell_size = 48.0
    matrix_x, matrix_y = x + 244, y + 104
    matrix_w = 10 * cell_size
    matrix_h = 10 * cell_size
    cell_w = cell_h = cell_size
    transitions = [(k, k + 1) for k in range(0, 10)]
    draw.text("Thinking-trace transition (k→k+1)", matrix_x,
              y + 50, matrix_w, 22,
              size=18.5, color=AURORA["black"], bold=True,
              align="center")
    for column, (source, target) in enumerate(transitions):
        label = f"{source}→{target}"
        draw.text(label, matrix_x + column * cell_w, y + 76, cell_w, 28,
                  size=17.0, color=AURORA["black"], bold=True, align="center")

    cities = [record["city"] for record in row["gold_records"]]
    for row_index, city in enumerate(cities):
        cy = matrix_y + row_index * cell_h
        draw.text(f"N{row_index + 1} · {city}", row_label_x, cy, row_label_w, cell_h,
                  size=18.0, color=AURORA["black"], align="right")

    # Restore the original high-contrast Aurora attention scale used in the
    # report: near-zero mass is midnight indigo rather than white.
    matrix_stops = (
        (0.00, AURORA["indigo"]),
        (0.22, AURORA["violet"]),
        (0.50, AURORA["blue"]),
        (0.76, AURORA["green"]),
        (1.00, AURORA["yellow"]),
    )
    for event_index, event in enumerate(events):
        values = event["relative_record_masses"]
        for row_index, value in enumerate(values):
            fill = interpolate_color(matrix_stops, float(value))
            draw.rect(
                matrix_x + event_index * cell_w,
                matrix_y + row_index * cell_h,
                cell_w, cell_h,
                fill=fill, stroke=AURORA["white"], stroke_width=0.9,
            )
        target_row = int(event["to_occurrence"]) - 1
        draw.rect(
            matrix_x + event_index * cell_w,
            matrix_y + target_row * cell_h,
            cell_w, cell_h,
            fill="none", stroke=AURORA["black"], stroke_width=1.15,
        )
        draw.ellipse(
            matrix_x + (event_index + 1) * cell_w - 14,
            matrix_y + target_row * cell_h + 8,
            7, 7, fill=AURORA["black"],
        )

    draw.rect(matrix_x, matrix_y, matrix_w, matrix_h, fill="none",
              stroke=blend(AURORA["gray"], 0.55), stroke_width=0.9)
    legend_x = x + width - 140
    legend_w, legend_h = 24, 360
    legend_y = matrix_y + (matrix_h - legend_h) / 2
    for index in range(50):
        value = 1.0 - index / 49
        draw.rect(
            legend_x, legend_y + index * legend_h / 50,
            legend_w, legend_h / 50 + 0.2,
            fill=interpolate_color(matrix_stops, value),
        )
    draw.rect(legend_x, legend_y, legend_w, legend_h, fill="none",
              stroke=blend(AURORA["gray"], 0.55), stroke_width=0.7)
    for label, fraction in (("1.0", 0.0), ("0.5", 0.5), ("0.0", 1.0)):
        draw.text(label, legend_x + legend_w + 8,
                  legend_y + fraction * legend_h - 10,
                  42, 20, size=14.5, color=MUTED_TEXT)
    draw.text("relative<br>needle mass", legend_x - 36,
              legend_y + legend_h + 8, 90, 38,
              size=15.0, color=MUTED_TEXT, html_label=True,
              align="center")


def build_caption(settings: dict, row: dict, nonthinking: dict,
                  events: Sequence[dict]) -> str:
    displayed = [row["gold_records"][index - 1] for index in DISPLAYED_OCCURRENCES]
    native_displayed = [
        row["gold_records"][index - 1] for index in NATIVE_DISPLAYED_OCCURRENCES
    ]
    displayed_text = ", ".join(
        f'N{record["slot_index"]} {record["city"]} ({record["score"]})'
        for record in displayed
    )
    native_displayed_text = ", ".join(
        f'N{record["slot_index"]} {record["city"]} ({record["score"]})'
        for record in native_displayed
    )
    mass_text = ", ".join(
        f'N{index + 1}={mass * 100:.2f}%'
        for index, mass in enumerate(nonthinking["record_masses"])
    )
    target_text = ", ".join(
        f'{event["from_occurrence"]}→{event["to_occurrence"]}: '
        f'{event["target_share"] * 100:.1f}%'
        for event in events
    )
    view = settings["pca_shared_view"]
    non_ncc = settings["pca_ncc"]["non_thinking_confirmation_balanced_accuracy"]
    native_ncc = settings["pca_ncc"]["native_thinking_confirmation_balanced_accuracy"]
    registered_spans = nonthinking["capture_manifest"]["needle_spans"]
    span_start = min(int(span[0]) for span in registered_spans)
    span_end = max(int(span[1]) for span in registered_spans)
    event_by_target = {
        int(event["to_occurrence"]): event for event in events
    }
    displayed_events = [
        event_by_target[occurrence] for occurrence in NATIVE_DISPLAYED_OCCURRENCES
    ]
    displayed_target_masses = ", ".join(
        f'{event["from_occurrence"]}→{event["to_occurrence"]} '
        f'{float(event["target_mass"]) * 100:.2f}%'
        for event in displayed_events
    )
    displayed_transitions = ", ".join(
        f'{event["from_occurrence"]}→{event["to_occurrence"]}'
        for event in displayed_events
    )
    displayed_trace_items = ", ".join(
        f'`{occurrence - 1}. city - score`'
        for occurrence in NATIVE_DISPLAYED_OCCURRENCES
    )
    return f"""# Main figure caption and settings

## Main-text caption (LaTeX-ready)

```latex
Non-thinking uses broad final-query retrieval, whereas native thinking repeatedly retrieves the next record. \\textbf{{A,}} On a matched $N=10$ prompt, token shading shows attention from a non-thinking final-answer readout, which attends to multiple records but returns an incorrect count. \\textbf{{B,}} On the same prompt, native-thinking item-end queries target the next record and support the correct count. \\textbf{{C,}} The full-prompt non-thinking profile contains distributed attention peaks. \\textbf{{D,}} Native-thinking endpoint-to-record attention from $0\\!\\to\\!1$ through $9\\!\\to\\!10$ forms a near-diagonal targeted-retrieval pattern. \\textbf{{E,}} Under a shared 3D view, native-thinking states form a more compact, ordered count manifold and achieve 98\\% held-out nearest-centroid classification accuracy, versus 46\\% for non-thinking. Head and layer selection, normalization, exact prompts, and provenance are provided in the appendix.
```

## Appendix note: methods, provenance, and display definitions

### Reproducibility status

- Every visible prompt excerpt, answer, token-attention value, and transition column is now drawn from seed {SELECTED_SEED}; no evidence is mixed across seeds.
- The remote capture was run without leaving a model process on the rented GPU; the downloaded JSON/NPZ artifacts below are the immutable inputs to this draft.

### Shared example

- Model: Qwen3-8B.
- Stimulus: `V4_4_T10000_N10_seed{SELECTED_SEED}`; gold N=10.
- The non-thinking and native-thinking prompt demos use the same seed and passage.
- Non-thinking prompt windows: {displayed_text}. Native-thinking prompt windows: {native_displayed_text}. Ellipses mark compressed intervals of the original haystack.
- Non-thinking output: `{nonthinking["answer"]}` (incorrect; gold answer is `Total: 10`).
- Native-thinking output: `Total: 10`.

### Full question wording

The main figure displays only the shared first-sentence question. The exact experimental wording is:

### Non-thinking

```text
{FULL_NONTHINK_QUESTION}
```

### Native-thinking

```text
{FULL_NATIVE_QUESTION}
```

### Panel-specific provenance

#### A and C · Non-thinking prompt demo and full-prompt attention

- Query: the final token of the forced answer prefix, immediately before answer generation.
- The thin pink rule in Panel A marks the exact end of the forced answer prefix `{nonthinking["answer_prefix"]}`. This is token position {int(nonthinking["query_position"]):,}, the captured head readout that produces the first unconstrained answer token; the generated integer is shown immediately to its right.
- Selected head: L{NONTHINK_LAYER}H{NONTHINK_HEAD}.
- The ten passage-span coordinates are taken from the matched native-generation span registry before any mass is computed; they occupy prompt-token positions {span_start:,}–{span_end:,} and all lie inside the non-thinking capture.
- Re-summing the raw L{NONTHINK_LAYER}H{NONTHINK_HEAD} vector reproduces the archived ten-span summary with maximum absolute deviation {nonthinking["official_span_mass_max_abs_delta"]:.2e}.
- This is an illustrative head selected to make the multi-needle pattern legible: the screen favored high total needle-span mass together with dispersion across several records. It should not be read as an unbiased estimate of average-head behavior.
- Needle-span attention mass: {nonthinking["needle_total_mass"] * 100:.2f}% total.
- Normalized entropy of the ten needle-span masses: {nonthinking["normalized_record_entropy"]:.3f}.
- Per-record raw masses: {mass_text}.
- Panel A uses raw per-token attention clipped at the 99.7th percentile of positive values. Every displayed token—needle and haystack alike—uses the same span-independent monotone nonlinear print transform (`{NONTHINK_PROMPT_ALPHA_SCALE:.2f} × [0.004 + 0.946(log(1+1000v)/log(1001))^0.62]` for positive `v`), where `v` is the clipped normalized raw token mass. The uniform {NONTHINK_PROMPT_ALPHA_SCALE:.2f} lightness multiplier keeps the broad non-thinking read visually subordinate to the concentrated targeted-retrieval examples without changing token ordering or relative variation; the Panel A legend uses the same multiplier.
- The three direct labels in Panel A report raw attention summed over each complete registered needle span; they are not transformed display intensities.
- Panel C sums attention within 170 equal-width prompt-position bins and applies a monotone square-root display transform.

#### B and D · Native-thinking targeted retrieval

- Panel B shows verbatim primary item spans from the stored seed-{SELECTED_SEED} trace ({displayed_trace_items}); ellipses mark omitted surrounding reasoning and later list items.
- Fixed attention head for all quantitative transitions: L{NATIVE_LAYER}H{NATIVE_HEAD}. It was selected over 1→2 through 9→10 by maximizing geometric-mean next-record raw mass multiplied by the square root of mean next-record share; the selection criterion never uses the 0→1 initialization row.
- Panel B uses exact key-wise attention vectors at item-end rows {displayed_transitions}. Values use one positive 98th-percentile clip and the same base token-wise transform as A, without A's display-only lightness multiplier. Hue identifies the transition; opacity monotonically encodes measured raw attention without using span membership.
- Panel D is a complete 10×10 map containing the exact stored rows from the initial 0→1 lookup through 9→10. The horizontal axis is the thinking-trace transition `k→k+1`. Each column normalizes the ten prompt-record masses to sum to one. Square cells preserve equal visual area across rows and transitions; the continuous Aurora scale is shown vertically at right. Target shares are: {target_text}.
- Re-summing the three displayed raw token vectors over their registered target spans matches the stored target masses to better than 5×10⁻⁶ absolute: {displayed_target_masses}.

#### E · Shared-camera PCA

- Display cohort: 100 held-out confirmation states per mode (10 seeds × 10 count states), projected into the discovery-fitted PCA basis. Restricting the artwork to the confirmation split reduces overplotting and aligns the displayed geometry with the held-out NCC comparison.
- Display layers are selected independently for each mode by the highest discovery-set NCC balanced accuracy (ties: higher discovery logistic balanced accuracy, then earlier layer). Non-thinking: L{PCA_NONTHINK_LAYER} (discovery NCC {settings["pca_layer_selection"]["non_thinking"]["discovery_ncc_balanced_accuracy"] * 100:.1f}%); native-thinking: L{PCA_NATIVE_LAYER} (discovery NCC {settings["pca_layer_selection"]["native_thinking"]["discovery_ncc_balanced_accuracy"] * 100:.1f}%). The selected layers are frozen before confirmation evaluation.
- Both modes use one camera: yaw {view["yaw_deg"]:.2f}°, pitch {view["pitch_deg"]:.2f}°. Because PCA signs are arbitrary, the orientation triad uses the equivalent display convention `(-PC1, +PC2, -PC3)`, making PC1 and PC3 point down and away from PC2. This shared sign reparameterization changes neither plotted point geometry nor any distance-based statistic.
- To balance the two subplots at paper scale, the complete non-thinking cloud (states, centroid path, and centroid nodes) receives one uniform {PCA_NONTHINK_DISPLAY_SCALE:.2f}× screen-space display scale about the subplot centre; native-thinking uses 1.00×. This matches their visible bounding-box diagonals while preserving all within-panel relative distances. The PC1/PC2/PC3 orientation triads remain identical in length and are not quantitative scale bars.
- Two-dimensional silhouette under that shared view: non-thinking {view["non_2d_silhouette"]:.3f}; native-thinking {view["native_2d_silhouette"]:.3f}.
- Perspective foreshortening, near-to-far painter ordering, and path weight encode projected depth. Ordinary states use one fixed {PCA_STATE_DIAMETER:.1f}-unit diameter and {PCA_STATE_OPACITY}% opacity in both modes; centroid nodes use one fixed {PCA_CENTROID_DIAMETER:.1f}-unit diameter. Thus marker size and chromatic strength do not vary with depth, and count colours remain directly comparable. The background is a sparse symmetric axonometric orientation grid: each of its three line families is exactly parallel to one projected PC axis, with three equally spaced positive/negative pairs around a central line through the visible origin. Both subplots share the same camera, origin, spacing, and styling. The grid indicates orientation under the shared weak-perspective view and is not a quantitative tick system.
- Held-out confirmation nearest-centroid balanced accuracy is shown inside each subplot: non-thinking {non_ncc * 100:.0f}%; native-thinking {native_ncc * 100:.0f}%. NCC is computed from the report's discovery-fitted representation and is independent of the selected display camera.

### Color and layout

- Aurora Sunset Pink (`{AURORA["red"]}`): non-thinking attention.
- Ice Cyan (`{AURORA["blue"]}`), Aurora Green (`{AURORA["green"]}`), and Aurora Yellow (`{AURORA["yellow"]}`): the three consecutive native-thinking retrievals. Within each hue, opacity reflects raw per-token attention under the documented monotone display transform.
- Panel D uses the report's Aurora mass scale: Midnight Indigo → Polar Violet → Ice Cyan → Aurora Green → Aurora Yellow.
- Panel E restores the source report's smoothly ordered count-wise Aurora sequence (violet → cyan → teal → green → yellow → orange → pink → violet), shared identically by both modes. Fixed-opacity state points and saturated centroid nodes use the same count colour; geometry, rather than hue, carries the mode comparison.
- Neutral text, target outlines, dividers, and panel borders use Night Black and Frost Gray on a pure paper-white (`#FFFFFF`) background.
- The exported canvas is cropped to the outer panel borders with zero decorative padding; spacing inside panels is retained for labels and data marks.
- The main figure deliberately omits model, seed, layer, head, and normalization metadata; this file is the authoritative detail record.
- The composition uses direct labels, hairline structural dividers, restrained token tint, and no decorative chrome. Color is reserved for attention magnitude, retrieval identity, and count identity.

### Source artifacts

- Non-thinking attention: `{NONTHINK_ATTN}`
- Non-thinking behavior: `{NONTHINK_BEHAVIOR}`
- Non-thinking archived head summary: `{NONTHINK_HEAD_SUMMARY}`
- Original native generation, prompt IDs, and prompt spans: `{NATIVE_ROWS}`
- Exact native fixed-head transition archive (Panel D displays 0→1…9→10): `{NATIVE_TRANSITIONS}`
- Exact native item-end token vectors: `{NATIVE_TOKEN_ATTN}`
- Native item-end capture metadata: `{NATIVE_TOKEN_ATTN_META}`
- Reproducible remote capture script: `{HERE / "extract_seed_attention_remote.py"}`
- PCA report payload: `{REPORT}`
"""


def audit_drawio_geometry(draw: Drawio) -> dict:
    """Fail fast on off-canvas cells and inconsistent text-box alignment."""
    vertices = 0
    text_cells = 0
    top_aligned_text = 0
    for cell in draw.root.findall("mxCell"):
        if cell.get("vertex") != "1":
            continue
        geometry = cell.find("mxGeometry")
        if geometry is None:
            continue
        vertices += 1
        x = float(geometry.get("x", "0"))
        y = float(geometry.get("y", "0"))
        width = float(geometry.get("width", "0"))
        height = float(geometry.get("height", "0"))
        if x < -1e-6 or y < -1e-6 or x + width > CANVAS_W + 1e-6 or y + height > CANVAS_H + 1e-6:
            raise RuntimeError(
                f"Off-canvas cell {cell.get('id')}: {(x, y, width, height)}"
            )
        style = cell.get("style", "")
        if style.startswith("text;"):
            text_cells += 1
            if "spacing=0;" not in style:
                raise RuntimeError(f"Text cell {cell.get('id')} has nonzero spacing")
            if "verticalAlign=middle;" in style:
                continue
            if "verticalAlign=top;" in style:
                top_aligned_text += 1
                continue
            raise RuntimeError(f"Text cell {cell.get('id')} has no vertical alignment")
    return {
        "vertices_checked": vertices,
        "text_cells_checked": text_cells,
        "intentional_top_aligned_multiline_cells": top_aligned_text,
        "off_canvas_cells": 0,
        "single_line_vertical_alignment": "middle",
        "text_spacing": 0,
    }


def main() -> None:
    row, tokenizer = load_native_example()
    nonthinking = load_nonthinking_attention(row)
    events = load_fixed_head_events()
    native_token_attention = load_native_token_attention(row)

    payload = load_pca_payload()
    non_layer = str(PCA_NONTHINK_LAYER)
    native_layer = str(PCA_NATIVE_LAYER)
    non_xyz, non_labels = points_for_layer(
        payload, "non_thinking", non_layer, split="confirmation"
    )
    native_xyz, native_labels = points_for_layer(
        payload, "native_thinking", native_layer, split="confirmation"
    )
    non_ncc = float(
        payload["non_thinking"]["layers"][non_layer]["metrics"]
        ["confirmation_ncc_balanced_accuracy"]
    )
    native_ncc = float(
        payload["native_thinking"]["layers"][native_layer]["metrics"]
        ["confirmation_ncc_balanced_accuracy"]
    )
    non_discovery_ncc = float(
        payload["non_thinking"]["layers"][non_layer]["metrics"]
        ["discovery_oof_ncc_balanced_accuracy"]
    )
    native_discovery_ncc = float(
        payload["native_thinking"]["layers"][native_layer]["metrics"]
        ["discovery_oof_ncc_balanced_accuracy"]
    )
    # Use one equivalent PCA sign convention in both panels.  The screen-space
    # manifold geometry is kept fixed; only the positive orientation rays are
    # reparameterized so PC1/PC3 point down and away from PC2.
    non_signs = [int(value) for value in PCA_DISPLAY_AXIS_SIGNS]
    native_signs = [int(value) for value in PCA_DISPLAY_AXIS_SIGNS]
    view = search_shared_view(non_xyz, non_labels, native_xyz, native_labels)

    settings = {
        "schema_version": "restrained_aurora_main_figure_v1",
        "font_family": "Times New Roman",
        "font_scale": FONT_SCALE,
        "background_color": AURORA["white"],
        "muted_text_color": MUTED_TEXT,
        "main_question_display": "shared first sentence only; full prompts are in caption_v1.md",
        "design_audit": (
            "Anthropic research-blog inspired: direct labels, hairline dividers, "
            "restrained tint, minimal visible metadata, no decorative chrome"
        ),
        "layout_audit": {
            "column_width": COL_W,
            "inter_column_gap": GAP,
            "outer_margin": MARGIN,
            "panel_edges": {
                "left": [LEFT_X, LEFT_X + COL_W],
                "right": [RIGHT_X, RIGHT_X + COL_W],
                "top": min(LEFT_A_Y, RIGHT_D_Y),
                "bottom": max(LEFT_C_Y + LEFT_C_H, RIGHT_E_Y + RIGHT_E_H),
            },
            "asymmetric_columns": {
                "left": {
                    "A_prompt_demo": [LEFT_A_Y, LEFT_A_Y + LEFT_A_H],
                    "C_attention_strip": [LEFT_B_Y, LEFT_B_Y + LEFT_B_H],
                    "E_pca_comparison": [LEFT_C_Y, LEFT_C_Y + LEFT_C_H],
                },
                "right": {
                    "B_prompt_demo": [RIGHT_D_Y, RIGHT_D_Y + RIGHT_D_H],
                    "D_attention_matrix": [RIGHT_E_Y, RIGHT_E_Y + RIGHT_E_H],
                },
            },
            "text_alignment": (
                "all single-line labels use verticalAlign=middle and spacing=0; "
                "boxed trace items use centered text before the colored end marker"
            ),
        },
        "selected_seed": SELECTED_SEED,
        "gold_count": 10,
        "displayed_occurrences": list(DISPLAYED_OCCURRENCES),
        "native_displayed_occurrences": list(NATIVE_DISPLAYED_OCCURRENCES),
        "nonthinking": {
            "layer": NONTHINK_LAYER,
            "head": NONTHINK_HEAD,
            "query_position": int(nonthinking["query_position"]),
            "answer_prefix": str(nonthinking["answer_prefix"]),
            "head_readout_marker": (
                "thin Sunset Pink rule at the exact end of the forced answer "
                "prefix, immediately before the generated integer"
            ),
            "needle_total_mass": nonthinking["needle_total_mass"],
            "normalized_record_entropy": nonthinking["normalized_record_entropy"],
            "record_masses": nonthinking["record_masses"],
            "official_span_mass_max_abs_delta": (
                nonthinking["official_span_mass_max_abs_delta"]
            ),
            "answer": nonthinking["answer"],
            "prompt_demo_display": (
                "raw per-token attention clipped at positive q99.7; every token "
                "uses one span-independent nonlinear transform, followed by "
                f"a uniform {NONTHINK_PROMPT_ALPHA_SCALE:.2f} display-lightness multiplier: "
                f"{NONTHINK_PROMPT_ALPHA_SCALE:.2f}*"
                "[0.004+0.946*(log(1+1000*v)/log(1001))^0.62]"
            ),
        },
        "native_thinking": {
            "layer": NATIVE_LAYER,
            "head": NATIVE_HEAD,
            "trace_display": (
                "verbatim primary item spans from the stored numbered-list trace; "
                "ellipses mark omitted surrounding reasoning and later items"
            ),
            "target_shares": [
                event["target_share"] for event in events
            ],
            "prompt_demo_resolution": (
                "exact key-wise token attention captured at output endpoints "
                + "/".join(
                    str(int(event["query_output_token_index"]))
                    for event in native_token_attention["events"]
                )
                + "; "
                + "one shared q98 clip and the documented span-independent monotone "
                + "token transform are applied to all three shown windows; "
                + "colored rules mark item-end queries"
            ),
            "token_capture": {
                "array_file": str(NATIVE_TOKEN_ATTN),
                "metadata_file": str(NATIVE_TOKEN_ATTN_META),
                "query_output_token_indices": [
                    int(event["query_output_token_index"])
                    for event in native_token_attention["events"]
                ],
                "query_full_sequence_tokens": [
                    int(event["query_full_sequence_token"])
                    for event in native_token_attention["events"]
                ],
                "target_masses": [
                    float(event["target_mass"])
                    for event in native_token_attention["events"]
                ],
                "attention_total_masses": [
                    float(event["attention_total_mass"])
                    for event in native_token_attention["events"]
                ],
            },
        },
        "pca_shared_view": view,
        "pca_display_cohort": "confirmation only: 10 seeds × 10 count states per mode",
        "pca_display_scales": {
            "non_thinking": PCA_NONTHINK_DISPLAY_SCALE,
            "native_thinking": 1.0,
            "note": (
                "uniform screen-space footprint balancing only; axes retain "
                "identical fixed lengths and are orientation cues, not scale bars"
            ),
        },
        "pca_layer_selection": {
            "rule": (
                "each mode independently maximizes discovery-set grouped-OOF "
                "nearest-centroid balanced accuracy; ties are resolved by higher "
                "discovery logistic balanced accuracy and then the earlier layer"
            ),
            "non_thinking": {
                "layer": PCA_NONTHINK_LAYER,
                "discovery_ncc_balanced_accuracy": non_discovery_ncc,
            },
            "native_thinking": {
                "layer": PCA_NATIVE_LAYER,
                "discovery_ncc_balanced_accuracy": native_discovery_ncc,
            },
        },
        "pca_ncc": {
            "definition": (
                "held-out confirmation nearest-centroid balanced accuracy; "
                "computed independently of the display camera"
            ),
            "non_thinking_confirmation_balanced_accuracy": non_ncc,
            "native_thinking_confirmation_balanced_accuracy": native_ncc,
        },
        "pca_axis_embedding": (
            "pure dark-gray PC1/PC2/PC3 axes originate at each subplot's screen centre, "
            "extend behind each manifold, and share one camera; three sparse background "
            "line families remain exactly parallel to the projected PC axes and use three "
            "matched, equally spaced positive/negative offset pairs around the visible origin; "
            "their central lines align with the labeled axes; PC1 and PC3 use the equivalent "
            "negative display sign so their positive rays point down and away from PC2; "
            f"the non-thinking cloud uses a uniform {PCA_NONTHINK_DISPLAY_SCALE:.2f}x "
            "screen-space footprint scale while native-thinking uses 1.00x; the axis "
            "lengths remain identical and non-quantitative; "
            "ordinary state opacity is fixed "
            f"at {PCA_STATE_OPACITY}% with a fixed {PCA_STATE_DIAMETER:.1f}-unit diameter "
            f"across both modes; centroid nodes use a fixed {PCA_CENTROID_DIAMETER:.1f}-unit "
            "diameter; path weight, foreshortening, and far-to-near painter ordering provide "
            "the depth cues"
        ),
        "pca_component_signs": {
            "non_thinking": non_signs,
            "native_thinking": native_signs,
        },
    }

    draw = Drawio()
    draw_prompt_demo_a(draw, row, tokenizer, nonthinking)
    draw_attention_strip_c(draw, row, nonthinking)
    draw_pca_e(
        draw, non_xyz, non_labels, native_xyz, native_labels,
        view, non_ncc, native_ncc,
    )
    draw_prompt_demo_b(draw, row, tokenizer, native_token_attention)
    draw_matrix_d(draw, row, events)
    settings["render_audit"] = audit_drawio_geometry(draw)
    draw.write(OUT_DRAWIO)

    OUT_SETTINGS.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    OUT_CAPTION.write_text(
        build_caption(settings, row, nonthinking, events),
        encoding="utf-8",
    )
    print(json.dumps({
        "drawio": str(OUT_DRAWIO),
        "caption": str(OUT_CAPTION),
        "settings": str(OUT_SETTINGS),
        "seed": SELECTED_SEED,
        "nonthinking_head": f"L{NONTHINK_LAYER}H{NONTHINK_HEAD}",
        "needle_total_mass": nonthinking["needle_total_mass"],
        "shared_view": view,
    }, indent=2))


if __name__ == "__main__":
    main()
