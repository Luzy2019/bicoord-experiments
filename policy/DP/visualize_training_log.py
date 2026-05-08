import argparse
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOG = Path(
    "data/outputs/2026.05.08/12.58.18_factorized_robot_stack_bowls_stack_bowls/logs.json.txt"
)


PLOT_GROUPS = [
    (
        "01_losses",
        "Training and validation losses",
        ["train_loss", "val_loss", "diffusion_loss", "speed_modulation_loss"],
        False,
    ),
    (
        "02_diffusion_losses",
        "Left/right diffusion losses",
        ["left_diffusion_loss", "right_diffusion_loss", "diffusion_loss"],
        False,
    ),
    (
        "03_factorized_aux_losses",
        "Factorized auxiliary losses",
        [
            "factorized_aux_loss",
            "left_marginal_aux_loss",
            "right_marginal_aux_loss",
            "left_cond_aux_loss",
            "right_cond_aux_loss",
        ],
        False,
    ),
    (
        "04_factorized_weights",
        "Factorized weights",
        ["factorized_w", "factorized_u"],
        False,
    ),
    (
        "05_speed_target_losses",
        "Speed target losses",
        ["left_speed_target_loss", "right_speed_target_loss", "speed_modulation_loss"],
        False,
    ),
    (
        "06_speed_regularizers",
        "Speed regularizer losses",
        [
            "left_speed_smooth_loss",
            "right_speed_smooth_loss",
            "left_speed_fast_loss",
            "right_speed_fast_loss",
            "left_speed_risk_loss",
            "right_speed_risk_loss",
        ],
        True,
    ),
    (
        "07_speed_alpha_mean",
        "Speed alpha mean",
        ["left_speed_alpha_mean", "right_speed_alpha_mean"],
        False,
    ),
    (
        "08_speed_alpha_range",
        "Speed alpha min/max",
        [
            "left_speed_alpha_min",
            "left_speed_alpha_max",
            "right_speed_alpha_min",
            "right_speed_alpha_max",
        ],
        False,
    ),
    (
        "09_coupling_risk",
        "Coupling MSE risk",
        ["left_coupling_mse_risk", "right_coupling_mse_risk"],
        False,
    ),
    (
        "10_train_action_mse",
        "Train action MSE error",
        ["train_action_mse_error"],
        False,
    ),
    (
        "11_learning_rate",
        "Learning rate",
        ["lr"],
        False,
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize JSONL training logs from DP runs.")
    parser.add_argument(
        "log_path",
        nargs="?",
        type=Path,
        default=DEFAULT_LOG,
        help=f"Path to logs.json.txt. Default: {DEFAULT_LOG}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated figures. Default: <log_dir>/log_plots",
    )
    parser.add_argument("--x-key", default="global_step", help="Metric used as x axis.")
    parser.add_argument(
        "--smooth",
        type=int,
        default=25,
        help="Moving average window. Use 1 to disable smoothing.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output figure DPI.",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Only save PNG/CSV outputs; skip the HTML dashboard.",
    )
    return parser.parse_args()


def load_jsonl(path: Path):
    rows = []
    skipped = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
    if not rows:
        raise ValueError(f"No valid JSON rows found in {path}")
    return rows, skipped


def series_from_rows(rows, key: str):
    values = np.full(len(rows), np.nan, dtype=np.float64)
    for idx, row in enumerate(rows):
        value = row.get(key)
        if isinstance(value, (int, float)):
            values[idx] = float(value)
    return values


def smooth_series(values: np.ndarray, window: int):
    if window <= 1:
        return values.copy()
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return values.copy()

    filled = values.copy()
    x = np.arange(len(values))
    filled[~valid] = np.interp(x[~valid], x[valid], values[valid])
    kernel = np.ones(window, dtype=np.float64)
    numerator = np.convolve(filled, kernel, mode="same")
    denominator = np.convolve(np.ones_like(filled), kernel, mode="same")
    smoothed = numerator / denominator
    smoothed[~valid] = np.nan
    return smoothed


def plot_group(output_dir: Path, name: str, title: str, x, series, log_y: bool, smooth: int, dpi: int):
    present = [(key, values) for key, values in series.items() if np.isfinite(values).any()]
    if not present:
        return None

    plt.figure(figsize=(12, 5))
    for key, values in present:
        valid = np.isfinite(values)
        if valid.sum() == 0:
            continue
        plt.plot(x[valid], values[valid], alpha=0.25, linewidth=0.8)
        y_plot = smooth_series(values, smooth)
        label = f"{key} (smooth {smooth})" if smooth > 1 else key
        plt.plot(x[valid], y_plot[valid], label=label, linewidth=1.8)

    plt.title(title)
    plt.xlabel("global step")
    plt.ylabel("value")
    if log_y:
        plt.yscale("symlog", linthresh=1e-8)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = output_dir / f"{name}.png"
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    return out_path


def plot_epoch_overview(output_dir: Path, x, epoch, dpi: int):
    if not np.isfinite(epoch).any():
        return None
    plt.figure(figsize=(12, 3))
    plt.step(x, epoch, where="post")
    plt.title("Epoch over global steps")
    plt.xlabel("global step")
    plt.ylabel("epoch")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = output_dir / "00_epoch_overview.png"
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    return out_path


def write_summary(output_dir: Path, rows, keys, skipped: int):
    lines = [
        f"records: {len(rows)}",
        f"skipped_invalid_lines: {skipped}",
        f"first_global_step: {rows[0].get('global_step')}",
        f"last_global_step: {rows[-1].get('global_step')}",
        f"first_epoch: {rows[0].get('epoch')}",
        f"last_epoch: {rows[-1].get('epoch')}",
        "",
        "metric,last,mean,min,max",
    ]
    for key in keys:
        values = series_from_rows(rows, key)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            continue
        lines.append(
            f"{key},{finite[-1]:.10g},{finite.mean():.10g},{finite.min():.10g},{finite.max():.10g}"
        )
    summary_path = output_dir / "summary.csv"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def format_metric(value):
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        return ""
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.4e}"
    return f"{value:.6g}"


def metric_stats(rows, key: str):
    values = series_from_rows(rows, key)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return None
    return {
        "last": float(finite[-1]),
        "mean": float(finite.mean()),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def render_metric_cards(rows, keys):
    cards = []
    for key in keys:
        stats = metric_stats(rows, key)
        if stats is None:
            continue
        cards.append(
            f"""
            <div class="metric-card">
              <div class="metric-name">{html.escape(key)}</div>
              <div class="metric-value">{format_metric(stats["last"])}</div>
              <div class="metric-sub">mean {format_metric(stats["mean"])}</div>
            </div>
            """
        )
    return "\n".join(cards)


def render_metric_table(rows, keys):
    body = []
    for key in keys:
        stats = metric_stats(rows, key)
        if stats is None:
            continue
        body.append(
            "<tr>"
            f"<td>{html.escape(key)}</td>"
            f"<td>{format_metric(stats['last'])}</td>"
            f"<td>{format_metric(stats['mean'])}</td>"
            f"<td>{format_metric(stats['min'])}</td>"
            f"<td>{format_metric(stats['max'])}</td>"
            "</tr>"
        )
    return "\n".join(body)


def write_html_dashboard(
    output_dir: Path,
    log_path: Path,
    rows,
    skipped: int,
    saved_plots,
    all_keys,
    smooth: int,
):
    key_cards = [
        "train_loss",
        "val_loss",
        "diffusion_loss",
        "speed_modulation_loss",
        "left_speed_alpha_mean",
        "right_speed_alpha_mean",
        "train_action_mse_error",
        "lr",
    ]
    plot_cards = []
    for plot_path in saved_plots:
        rel_path = plot_path.relative_to(output_dir).as_posix()
        title = plot_path.stem.replace("_", " ")
        plot_cards.append(
            f"""
            <section class="plot-card" id="{html.escape(plot_path.stem)}">
              <div class="plot-title">{html.escape(title)}</div>
              <a href="{html.escape(rel_path)}" target="_blank">
                <img src="{html.escape(rel_path)}" alt="{html.escape(title)}">
              </a>
            </section>
            """
        )

    nav_links = "\n".join(
        f'<a href="#{html.escape(plot_path.stem)}">{html.escape(plot_path.stem)}</a>'
        for plot_path in saved_plots
    )
    dashboard = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DP Training Dashboard</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #111827;
      --panel-soft: #1f2937;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --line: #334155;
      --accent: #60a5fa;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      padding: 18px 28px;
      background: rgba(15, 23, 42, 0.94);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(8px);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 22px;
      letter-spacing: 0.2px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
      word-break: break-all;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 220px minmax(0, 1fr);
      gap: 18px;
      padding: 18px 24px 36px;
    }}
    nav {{
      position: sticky;
      top: 92px;
      align-self: start;
      max-height: calc(100vh - 110px);
      overflow: auto;
      padding: 14px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
    }}
    nav a {{
      display: block;
      padding: 7px 8px;
      color: var(--muted);
      text-decoration: none;
      border-radius: 8px;
      font-size: 13px;
    }}
    nav a:hover {{
      color: var(--text);
      background: var(--panel-soft);
    }}
    main {{
      min-width: 0;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .metric-card, .plot-card, .table-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.20);
    }}
    .metric-card {{
      padding: 14px 16px;
    }}
    .metric-name {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }}
    .metric-value {{
      color: var(--accent);
      font-size: 25px;
      font-weight: 700;
    }}
    .metric-sub {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
    }}
    .plots {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
      gap: 18px;
    }}
    .plot-card {{
      padding: 12px;
      scroll-margin-top: 105px;
    }}
    .plot-title {{
      margin: 2px 4px 10px;
      color: var(--muted);
      font-size: 14px;
      text-transform: capitalize;
    }}
    img {{
      width: 100%;
      display: block;
      border-radius: 10px;
      background: white;
    }}
    .table-card {{
      margin-top: 18px;
      padding: 14px;
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
    }}
    @media (max-width: 900px) {{
      .layout {{ grid-template-columns: 1fr; }}
      nav {{ position: static; max-height: none; }}
      .plots {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>DP Training Dashboard</h1>
    <div class="meta">
      log: {html.escape(str(log_path))}<br>
      records: {len(rows)} · skipped invalid lines: {skipped} ·
      global step: {html.escape(str(rows[0].get("global_step")))} → {html.escape(str(rows[-1].get("global_step")))} ·
      epoch: {html.escape(str(rows[0].get("epoch")))} → {html.escape(str(rows[-1].get("epoch")))} ·
      smooth window: {smooth}
    </div>
  </header>
  <div class="layout">
    <nav>
      {nav_links}
    </nav>
    <main>
      <section class="metrics">
        {render_metric_cards(rows, key_cards)}
      </section>
      <section class="plots">
        {"".join(plot_cards)}
      </section>
      <section class="table-card">
        <table>
          <thead>
            <tr><th>metric</th><th>last</th><th>mean</th><th>min</th><th>max</th></tr>
          </thead>
          <tbody>
            {render_metric_table(rows, all_keys)}
          </tbody>
        </table>
      </section>
    </main>
  </div>
</body>
</html>
"""
    html_path = output_dir / "index.html"
    html_path.write_text(dashboard, encoding="utf-8")
    return html_path


def main():
    args = parse_args()
    log_path = args.log_path.expanduser().resolve()
    output_dir = args.output_dir or log_path.parent / "log_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, skipped = load_jsonl(log_path)
    all_keys = sorted({key for row in rows for key in row})
    x = series_from_rows(rows, args.x_key)
    if not np.isfinite(x).any():
        x = np.arange(len(rows), dtype=np.float64)

    saved = []
    epoch_path = plot_epoch_overview(output_dir, x, series_from_rows(rows, "epoch"), args.dpi)
    if epoch_path is not None:
        saved.append(epoch_path)

    for name, title, keys, log_y in PLOT_GROUPS:
        series = {key: series_from_rows(rows, key) for key in keys}
        out_path = plot_group(output_dir, name, title, x, series, log_y, args.smooth, args.dpi)
        if out_path is not None:
            saved.append(out_path)

    summary_path = write_summary(output_dir, rows, all_keys, skipped)
    html_path = None
    if not args.no_html:
        html_path = write_html_dashboard(
            output_dir=output_dir,
            log_path=log_path,
            rows=rows,
            skipped=skipped,
            saved_plots=saved,
            all_keys=all_keys,
            smooth=args.smooth,
        )
    print(f"Loaded {len(rows)} valid rows from {log_path}")
    if skipped:
        print(f"Skipped {skipped} invalid/incomplete lines")
    print(f"Saved {len(saved)} figures to {output_dir}")
    print(f"Saved summary to {summary_path}")
    if html_path is not None:
        print(f"Saved HTML dashboard to {html_path}")


if __name__ == "__main__":
    main()
