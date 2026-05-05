#!/usr/bin/env python3
"""Analyze action trajectories in a RoboTwin/Diffusion Policy zarr dataset."""

import argparse
import json
import math
import pathlib
import webbrowser
from html import escape
from typing import Dict, List, Tuple

import numpy as np
import zarr


DEFAULT_ACTION_NAMES_14 = [
    "left_joint_0",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_gripper",
    "right_joint_0",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_gripper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an HTML dashboard for per-episode action changes in a zarr dataset."
    )
    parser.add_argument("zarr_path", help="Path to a .zarr dataset directory.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output HTML path. Defaults to <zarr_path>/action_dashboard.html.",
    )
    parser.add_argument(
        "--max-points-per-episode",
        type=int,
        default=600,
        help="Downsample each episode to at most this many points in the HTML. Default: 600.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Initially selected episode index. Default: 0.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated dashboard in the default browser.",
    )
    return parser.parse_args()


def downsample_indices(length: int, max_points: int) -> np.ndarray:
    if max_points <= 0 or length <= max_points:
        return np.arange(length)
    indices = np.linspace(0, length - 1, num=max_points, dtype=np.int64)
    return np.unique(indices)


def get_episode_slices(episode_ends: np.ndarray) -> List[Tuple[int, int]]:
    starts = np.concatenate([[0], episode_ends[:-1]])
    return [(int(start), int(end)) for start, end in zip(starts, episode_ends)]


def get_action_names(action_dim: int) -> List[str]:
    if action_dim == 14:
        return DEFAULT_ACTION_NAMES_14
    return [f"action_{idx}" for idx in range(action_dim)]


def get_arm_for_dim(dim_idx: int, action_dim: int, name: str) -> str:
    if name.startswith("left_"):
        return "left"
    if name.startswith("right_"):
        return "right"
    return "left" if dim_idx < action_dim // 2 else "right"


def get_arm_indices(action_names: List[str]) -> Dict[str, List[int]]:
    action_dim = len(action_names)
    arms = {"left": [], "right": []}
    for dim_idx, name in enumerate(action_names):
        arms[get_arm_for_dim(dim_idx, action_dim, name)].append(dim_idx)
    return arms


def finite_float(value: float) -> float:
    value = float(value)
    if math.isfinite(value):
        return value
    return 0.0


def summarize_dim(values: np.ndarray) -> Dict[str, float]:
    diffs = np.diff(values, axis=0)
    abs_diffs = np.abs(diffs)
    return {
        "min": finite_float(np.min(values)),
        "max": finite_float(np.max(values)),
        "mean": finite_float(np.mean(values)),
        "std": finite_float(np.std(values)),
        "mean_abs_delta": finite_float(np.mean(abs_diffs)) if len(values) > 1 else 0.0,
        "max_abs_delta": finite_float(np.max(abs_diffs)) if len(values) > 1 else 0.0,
    }


def build_dashboard_data(
    action: np.ndarray,
    state: np.ndarray,
    episode_ends: np.ndarray,
    max_points_per_episode: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    episode_slices = get_episode_slices(episode_ends)
    action_dim = action.shape[1]
    action_names = get_action_names(action_dim)
    arm_indices = get_arm_indices(action_names)

    episodes: List[Dict] = []
    episode_summary: List[Dict] = []
    dim_summary: List[Dict] = []

    for ep_idx, (start, end) in enumerate(episode_slices):
        ep_action = action[start:end]
        indices = downsample_indices(len(ep_action), max_points_per_episode)
        sampled = ep_action[indices]
        local_x = indices.astype(int).tolist()
        global_x = (indices + start).astype(int).tolist()
        deltas = np.diff(sampled, axis=0, prepend=sampled[:1])
        abs_delta_norm = np.linalg.norm(np.diff(ep_action, axis=0), axis=1) if len(ep_action) > 1 else np.zeros(0)
        left_abs_delta_norm = (
            np.linalg.norm(np.diff(ep_action[:, arm_indices["left"]], axis=0), axis=1)
            if len(ep_action) > 1 and arm_indices["left"]
            else np.zeros(0)
        )
        right_abs_delta_norm = (
            np.linalg.norm(np.diff(ep_action[:, arm_indices["right"]], axis=0), axis=1)
            if len(ep_action) > 1 and arm_indices["right"]
            else np.zeros(0)
        )

        episodes.append(
            {
                "index": ep_idx,
                "start": start,
                "end": end,
                "length": end - start,
                "x": local_x,
                "global_x": global_x,
                "actions": sampled.astype(float).round(6).tolist(),
                "deltas": deltas.astype(float).round(6).tolist(),
            }
        )
        episode_summary.append(
            {
                "episode": ep_idx,
                "start": start,
                "end": end,
                "length": end - start,
                "mean_abs_delta_norm": finite_float(np.mean(abs_delta_norm)) if len(abs_delta_norm) else 0.0,
                "max_abs_delta_norm": finite_float(np.max(abs_delta_norm)) if len(abs_delta_norm) else 0.0,
                "left_mean_delta_norm": finite_float(np.mean(left_abs_delta_norm)) if len(left_abs_delta_norm) else 0.0,
                "left_max_delta_norm": finite_float(np.max(left_abs_delta_norm)) if len(left_abs_delta_norm) else 0.0,
                "right_mean_delta_norm": finite_float(np.mean(right_abs_delta_norm)) if len(right_abs_delta_norm) else 0.0,
                "right_max_delta_norm": finite_float(np.max(right_abs_delta_norm)) if len(right_abs_delta_norm) else 0.0,
                "action_min": finite_float(np.min(ep_action)),
                "action_max": finite_float(np.max(ep_action)),
            }
        )

    for dim_idx, name in enumerate(action_names):
        stats = summarize_dim(action[:, dim_idx])
        dim_summary.append({"dim": dim_idx, "arm": get_arm_for_dim(dim_idx, action_dim, name), "name": name, **stats})

    return episodes, episode_summary, dim_summary


def fmt(value: float) -> str:
    if abs(value) >= 10000 or (0 < abs(value) < 0.001):
        return f"{value:.4e}"
    return f"{value:.6g}"


def render_rows(rows: List[Dict], columns: List[str]) -> str:
    html_rows = []
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = fmt(value)
            cells.append(f"<td>{escape(str(value))}</td>")
        html_rows.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(html_rows)


def render_html(
    *,
    title: str,
    zarr_path: pathlib.Path,
    action_shape: Tuple[int, int],
    state_shape,
    camera_shape,
    episode_count: int,
    initial_episode: int,
    action_names: List[str],
    episodes: List[Dict],
    episode_summary: List[Dict],
    dim_summary: List[Dict],
) -> str:
    data_json = json.dumps(
        {
            "actionNames": action_names,
            "episodes": episodes,
            "episodeSummary": episode_summary,
            "dimSummary": dim_summary,
            "initialEpisode": initial_episode,
        },
        ensure_ascii=False,
    )
    dim_rows = render_rows(
        dim_summary,
        ["dim", "arm", "name", "min", "max", "mean", "std", "mean_abs_delta", "max_abs_delta"],
    )
    episode_rows = render_rows(
        episode_summary,
        [
            "episode",
            "start",
            "end",
            "length",
            "left_mean_delta_norm",
            "left_max_delta_norm",
            "right_mean_delta_norm",
            "right_max_delta_norm",
            "mean_abs_delta_norm",
            "max_abs_delta_norm",
        ],
    )

    camera_text = "not found" if camera_shape is None else str(tuple(camera_shape))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #0f172a;
      --card: #111827;
      --muted: #94a3b8;
      --text: #e5e7eb;
      --border: #1f2937;
      --accent: #38bdf8;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 24px 28px 12px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 26px;
    }}
    .sub {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.65;
      word-break: break-all;
    }}
    .toolbar {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      padding: 12px 28px 18px;
    }}
    input, select, button {{
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #020617;
      color: var(--text);
      padding: 9px 12px;
      font-size: 14px;
    }}
    button {{
      cursor: pointer;
      border-color: #334155;
    }}
    button:hover {{
      border-color: var(--accent);
    }}
    main {{
      display: grid;
      grid-template-columns: 340px minmax(0, 1fr);
      gap: 18px;
      padding: 0 28px 28px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      box-shadow: 0 8px 28px rgba(0, 0, 0, 0.25);
    }}
    .panel {{
      padding: 16px;
      max-height: 78vh;
      overflow: auto;
    }}
    .metric {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 7px 4px;
      font-size: 14px;
    }}
    #leftActionChart, #rightActionChart, #leftDeltaChart, #rightDeltaChart {{
      height: 38vh;
      min-height: 320px;
    }}
    .charts {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      margin-top: 12px;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 7px 6px;
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
    }}
    h2 {{
      font-size: 15px;
      margin: 18px 0 8px;
    }}
    h2:first-child {{
      margin-top: 0;
    }}
    @media (max-width: 1200px) {{
      .charts {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 980px) {{
      main {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(title)}</h1>
    <div class="sub">
      source: {escape(str(zarr_path))}<br>
      action: {escape(str(tuple(action_shape)))}, state: {escape(str(tuple(state_shape)))}, head_camera: {escape(camera_text)}, episodes: {episode_count}
    </div>
  </header>
  <div class="toolbar">
    <label>episode <select id="episodeSelect" onchange="drawCharts()"></select></label>
    <input id="filter" placeholder="filter action dims..." oninput="renderDimList()">
    <button onclick="selectArm('left')">only left</button>
    <button onclick="selectArm('right')">only right</button>
    <button onclick="selectAll()">all dims</button>
    <button onclick="selectNone()">none</button>
  </div>
  <main>
    <section class="card panel">
      <h2>Action Dims</h2>
      <div id="dimList"></div>
      <h2>Per-Dim Summary</h2>
      <table>
        <thead>
          <tr><th>dim</th><th>arm</th><th>name</th><th>min</th><th>max</th><th>mean</th><th>std</th><th>mean |delta|</th><th>max |delta|</th></tr>
        </thead>
        <tbody>{dim_rows}</tbody>
      </table>
      <h2>Episode Summary</h2>
      <table>
        <thead>
          <tr><th>ep</th><th>start</th><th>end</th><th>len</th><th>left mean</th><th>left max</th><th>right mean</th><th>right max</th><th>all mean</th><th>all max</th></tr>
        </thead>
        <tbody>{episode_rows}</tbody>
      </table>
    </section>
    <section class="charts">
      <div class="card"><div id="leftActionChart"></div></div>
      <div class="card"><div id="rightActionChart"></div></div>
      <div class="card"><div id="leftDeltaChart"></div></div>
      <div class="card"><div id="rightDeltaChart"></div></div>
    </section>
  </main>
  <script>
    const data = {data_json};
    let selected = new Set(data.actionNames.map((_, idx) => idx));
    const arms = {{
      left: data.dimSummary.filter(item => item.arm === "left").map(item => item.dim),
      right: data.dimSummary.filter(item => item.arm === "right").map(item => item.dim)
    }};

    function colorFor(index) {{
      const palette = ["#38bdf8", "#fb7185", "#a78bfa", "#34d399", "#fbbf24", "#f472b6", "#60a5fa", "#c084fc", "#f97316", "#22d3ee", "#84cc16", "#e879f9", "#2dd4bf", "#facc15"];
      return palette[index % palette.length];
    }}

    function setupEpisodes() {{
      const select = document.getElementById("episodeSelect");
      data.episodes.forEach(ep => {{
        const option = document.createElement("option");
        option.value = ep.index;
        option.textContent = `episode ${{ep.index}} (${{ep.length}} steps)`;
        if (ep.index === data.initialEpisode) option.selected = true;
        select.appendChild(option);
      }});
    }}

    function renderDimList() {{
      const filter = document.getElementById("filter").value.toLowerCase();
      const root = document.getElementById("dimList");
      root.innerHTML = "";
      ["left", "right"].forEach(arm => {{
        const title = document.createElement("h2");
        title.textContent = arm === "left" ? "Left Arm" : "Right Arm";
        root.appendChild(title);
        arms[arm].forEach(idx => {{
          const name = data.actionNames[idx];
          if (!name.toLowerCase().includes(filter) && !String(idx).includes(filter) && !arm.includes(filter)) return;
          const label = document.createElement("label");
          label.className = "metric";
          const checkbox = document.createElement("input");
          checkbox.type = "checkbox";
          checkbox.checked = selected.has(idx);
          checkbox.onchange = () => {{
            checkbox.checked ? selected.add(idx) : selected.delete(idx);
            drawCharts();
          }};
          const span = document.createElement("span");
          span.textContent = `${{idx}}: ${{name}}`;
          label.appendChild(checkbox);
          label.appendChild(span);
          root.appendChild(label);
        }});
      }});
    }}

    function currentEpisode() {{
      const idx = Number(document.getElementById("episodeSelect").value);
      return data.episodes.find(ep => ep.index === idx) || data.episodes[0];
    }}

    function makeTraces(ep, key, arm) {{
      return arms[arm].filter(dim => selected.has(dim)).sort((a, b) => a - b).map(dim => ({{
        x: ep.x,
        y: ep[key].map(row => row[dim]),
        name: `${{dim}}: ${{data.actionNames[dim]}}`,
        mode: "lines",
        line: {{ color: colorFor(dim), width: 1.8 }},
        hovertemplate: "step=%{{x}}<br>value=%{{y:.6g}}<extra>" + data.actionNames[dim] + "</extra>"
      }}));
    }}

    function drawCharts() {{
      const ep = currentEpisode();
      const baseLayout = {{
        paper_bgcolor: "#111827",
        plot_bgcolor: "#111827",
        font: {{ color: "#e5e7eb" }},
        margin: {{ l: 58, r: 24, t: 38, b: 48 }},
        hovermode: "x unified",
        legend: {{ orientation: "h", y: -0.24 }},
        xaxis: {{ title: "episode step", gridcolor: "#1f2937", zerolinecolor: "#334155" }},
        yaxis: {{ gridcolor: "#1f2937", zerolinecolor: "#334155" }}
      }};
      Plotly.newPlot("leftActionChart", makeTraces(ep, "actions", "left"), {{
        ...baseLayout,
        title: `Episode ${{ep.index}} Left Action Values`,
        yaxis: {{ ...baseLayout.yaxis, title: "action" }}
      }}, {{ responsive: true, displaylogo: false }});
      Plotly.newPlot("rightActionChart", makeTraces(ep, "actions", "right"), {{
        ...baseLayout,
        title: `Episode ${{ep.index}} Right Action Values`,
        yaxis: {{ ...baseLayout.yaxis, title: "action" }}
      }}, {{ responsive: true, displaylogo: false }});
      Plotly.newPlot("leftDeltaChart", makeTraces(ep, "deltas", "left"), {{
        ...baseLayout,
        title: `Episode ${{ep.index}} Left Step-to-Step Delta`,
        yaxis: {{ ...baseLayout.yaxis, title: "delta action" }}
      }}, {{ responsive: true, displaylogo: false }});
      Plotly.newPlot("rightDeltaChart", makeTraces(ep, "deltas", "right"), {{
        ...baseLayout,
        title: `Episode ${{ep.index}} Right Step-to-Step Delta`,
        yaxis: {{ ...baseLayout.yaxis, title: "delta action" }}
      }}, {{ responsive: true, displaylogo: false }});
    }}

    function selectArm(which) {{
      selected = new Set(arms[which]);
      renderDimList();
      drawCharts();
    }}

    function selectAll() {{
      selected = new Set(data.actionNames.map((_, idx) => idx));
      renderDimList();
      drawCharts();
    }}

    function selectNone() {{
      selected = new Set();
      renderDimList();
      drawCharts();
    }}

    setupEpisodes();
    renderDimList();
    drawCharts();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    zarr_path = pathlib.Path(args.zarr_path).expanduser().resolve()
    if not zarr_path.is_dir():
        raise FileNotFoundError(f"Cannot find zarr directory: {zarr_path}")

    root = zarr.open(str(zarr_path), mode="r")
    action = np.asarray(root["data/action"][:], dtype=np.float32)
    state = np.asarray(root["data/state"][:], dtype=np.float32)
    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    camera_shape = tuple(root["data/head_camera"].shape) if "head_camera" in root["data"] else None

    if action.ndim != 2:
        raise ValueError(f"Expected action shape (T, D), got {action.shape}")
    if len(episode_ends) == 0:
        raise ValueError("meta/episode_ends is empty")

    initial_episode = min(max(args.episode, 0), len(episode_ends) - 1)
    action_names = get_action_names(action.shape[1])
    episodes, episode_summary, dim_summary = build_dashboard_data(
        action=action,
        state=state,
        episode_ends=episode_ends,
        max_points_per_episode=args.max_points_per_episode,
    )

    out_path = pathlib.Path(args.out).expanduser().resolve() if args.out else zarr_path / "action_dashboard.html"
    html = render_html(
        title=f"{zarr_path.name} action analysis",
        zarr_path=zarr_path,
        action_shape=action.shape,
        state_shape=state.shape,
        camera_shape=camera_shape,
        episode_count=len(episode_ends),
        initial_episode=initial_episode,
        action_names=action_names,
        episodes=episodes,
        episode_summary=episode_summary,
        dim_summary=dim_summary,
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote action dashboard: {out_path}")
    print(f"action shape: {action.shape}, state shape: {state.shape}, episodes: {len(episode_ends)}")
    if args.open:
        webbrowser.open(f"file://{out_path}")


if __name__ == "__main__":
    main()
