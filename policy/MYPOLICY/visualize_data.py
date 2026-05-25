import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


LEFT_DIM = 7
RIGHT_DIM = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize MYPOLICY trajectory data as PNG files.")
    parser.add_argument("--input", type=Path, default=Path("outputs/mypolicy_samples_stage1.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/data_vis"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument(
        "--keys",
        nargs="+",
        default=None,
        help="Trajectory keys to draw. Default: source target and prediction if present.",
    )
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height-per-panel", type=int, default=330)
    return parser.parse_args()


def load_font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def load_payload(path: Path) -> Dict[str, np.ndarray]:
    payload = np.load(path)
    return {
        key: payload[key]
        for key in payload.files
        if payload[key].ndim == 3 and payload[key].shape[-1] >= LEFT_DIM + RIGHT_DIM
    }


def active_mask(trajectory: np.ndarray, eps: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    left = np.linalg.norm(trajectory[:, :LEFT_DIM], axis=-1) > eps
    right = np.linalg.norm(trajectory[:, LEFT_DIM : LEFT_DIM + RIGHT_DIM], axis=-1) > eps
    return left, right


def active_overlap(trajectory: np.ndarray, eps: float = 1e-4) -> float:
    left, right = active_mask(trajectory, eps=eps)
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(left, right).sum() / union)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    panel_box: Tuple[int, int, int, int],
    title: str,
    trajectory: np.ndarray,
    title_font,
    small_font,
) -> None:
    x0, y0, x1, y1 = panel_box
    draw.rectangle(panel_box, fill=(255, 255, 255), outline=(203, 213, 225), width=1)
    draw.text((x0 + 14, y0 + 10), title, fill=(15, 23, 42), font=title_font)

    plot_left = x0 + 70
    plot_right = x1 - 20
    plot_top = y0 + 48
    plot_bottom = y1 - 58
    plot_height = plot_bottom - plot_top
    split_y = plot_top + plot_height // 2

    left_values = trajectory[:, :LEFT_DIM]
    right_values = trajectory[:, LEFT_DIM : LEFT_DIM + RIGHT_DIM]
    max_abs = float(np.nanmax(np.abs(trajectory)))
    max_abs = max(max_abs, 1e-6)
    horizon = trajectory.shape[0]

    def sx(t: int) -> float:
        return plot_left + t / max(horizon - 1, 1) * (plot_right - plot_left)

    def sy(value: float, center: float, half_height: float) -> float:
        return center - value / max_abs * half_height

    # grid
    for i in range(6):
        x = plot_left + i * (plot_right - plot_left) / 5
        draw.line([(x, plot_top), (x, plot_bottom)], fill=(241, 245, 249), width=1)
        draw.text((x - 10, plot_bottom + 8), str(round(i * (horizon - 1) / 5)), fill=(71, 85, 105), font=small_font)
    for center, label in ((plot_top + plot_height * 0.25, "left"), (plot_top + plot_height * 0.75, "right")):
        draw.line([(plot_left, center), (plot_right, center)], fill=(148, 163, 184), width=1)
        draw.text((x0 + 18, center - 8), label, fill=(51, 65, 85), font=small_font)

    left_center = plot_top + plot_height * 0.25
    right_center = plot_top + plot_height * 0.75
    half = plot_height * 0.20
    colors = [
        (37, 99, 235),
        (220, 38, 38),
        (22, 163, 74),
        (147, 51, 234),
        (234, 88, 12),
        (8, 145, 178),
        (190, 18, 60),
    ]

    for dim in range(LEFT_DIM):
        pts = [(sx(t), sy(float(left_values[t, dim]), left_center, half)) for t in range(horizon)]
        draw.line(pts, fill=colors[dim], width=2)
    for dim in range(RIGHT_DIM):
        pts = [(sx(t), sy(float(right_values[t, dim]), right_center, half)) for t in range(horizon)]
        draw.line(pts, fill=colors[dim], width=2)

    left_active, right_active = active_mask(trajectory)
    bar_top = y1 - 34
    bar_h = 9
    for t in range(horizon):
        xa = sx(t)
        xb = sx(min(t + 1, horizon - 1))
        if left_active[t]:
            draw.rectangle((xa, bar_top, max(xa + 1, xb), bar_top + bar_h), fill=(59, 130, 246))
        if right_active[t]:
            draw.rectangle((xa, bar_top + bar_h + 3, max(xa + 1, xb), bar_top + 2 * bar_h + 3), fill=(239, 68, 68))
    draw.text((x0 + 14, bar_top - 2), "active", fill=(71, 85, 105), font=small_font)
    draw.text(
        (plot_left, y1 - 20),
        f"max_abs={max_abs:.4f} | active_overlap={active_overlap(trajectory):.3f}",
        fill=(71, 85, 105),
        font=small_font,
    )


def draw_sample(
    output_path: Path,
    sample_index: int,
    trajectories: List[Tuple[str, np.ndarray]],
    width: int,
    height_per_panel: int,
) -> None:
    title_font = load_font(16)
    small_font = load_font(11)
    margin = 18
    gap = 14
    height = margin * 2 + len(trajectories) * height_per_panel + (len(trajectories) - 1) * gap
    image = Image.new("RGB", (width, height), (15, 23, 42))
    draw = ImageDraw.Draw(image)
    draw.text((margin, 8), f"MYPOLICY trajectory sample {sample_index}", fill=(226, 232, 240), font=title_font)

    y = margin + 22
    for name, trajectory in trajectories:
        draw_panel(
            draw=draw,
            panel_box=(margin, y, width - margin, y + height_per_panel),
            title=name,
            trajectory=trajectory,
            title_font=title_font,
            small_font=small_font,
        )
        y += height_per_panel + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data = load_payload(args.input)
    if not data:
        raise ValueError(f"No trajectory arrays found in {args.input}")

    keys = args.keys or [
        key for key in ("source", "target", "prediction_raw", "prediction") if key in data
    ]
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"Missing keys in {args.input}: {', '.join(missing)}")

    total = min(data[keys[0]].shape[0], args.start + args.num_samples)
    rows = []
    for sample_index in range(args.start, total):
        trajectories = [(key, data[key][sample_index]) for key in keys]
        output_path = args.output_dir / f"sample_{sample_index:04d}.png"
        draw_sample(
            output_path=output_path,
            sample_index=sample_index,
            trajectories=trajectories,
            width=args.width,
            height_per_panel=args.height_per_panel,
        )
        row = {"sample": sample_index}
        for key, trajectory in trajectories:
            row[f"{key}_active_overlap"] = active_overlap(trajectory)
            row[f"{key}_max_abs"] = float(np.nanmax(np.abs(trajectory)))
        rows.append(row)

    write_summary(args.output_dir / "summary.csv", rows)
    print(f"Saved {len(rows)} PNG files to {args.output_dir}")
    print(f"Saved summary to {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
