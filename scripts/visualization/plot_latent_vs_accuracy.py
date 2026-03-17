#!/usr/bin/env python3
"""
Plot line chart: X-axis = latent dimension (1, 2, 4, 8, 16), Y-axis = max accuracy.
Three lines: BLINK, MMVP, HRBench-4K.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "evaluation/results"

FOLDERS = [
    "SFT_box_resampler_steps2500_b4_resampler0.1_acc8_latent1",
    "SFT_box_resampler_steps2500_b4_resampler0.1_acc8_latent2",
    "SFT_box_resampler_steps2500_b4_resampler0.1_acc8_latent4",
    "SFT_box_resampler_steps2500_b4_resampler0.1_acc8_latent8",
    "SFT_box_resampler_steps2500_b4_resampler0.1_acc8_latent16",
]

LATENT_DIMS = [1, 2, 4, 8, 16]

# =============================================================================
# Custom line chart data points
# Override auto-loaded values when a dataset has no folder under results or when you want explicit values
# Format: { "dataset_name": [y1, y2, y3, y4, y5] } for x=[1, 2, 4, 8, 16]
# Use None or omit to auto-load from JSON under results
# =============================================================================
CUSTOM_LINE_POINTS = {
    "BLINK": [15.6, 35.2, 46.4, 54.2, 53.37],       # None = auto-load from blink/decoding_by_steps/box_resampler
    "MMVP": [24.3, 42.3, 41.3, 70.9, 69.6],        # None = auto-load from MMVP/decoding_by_steps/box_resampler
    "HRBench-4K": [25.0, 41.4, 45.6, 71.1, 69.9],  # None = auto-load; if no data, fill here e.g. [15.0, 35.0, 45.0, 52.0, 51.0]
}


def get_max_accuracy(folder_path: Path) -> float | None:
    """Find the highest overall_accuracy in a folder."""
    max_acc = None
    if not folder_path.exists():
        return None
    for f in folder_path.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
            if isinstance(data, list) and len(data) > 0:
                first = data[0]
                if "overall_accuracy" in first:
                    acc = first["overall_accuracy"]
                    if max_acc is None or acc > max_acc:
                        max_acc = acc
        except Exception:
            continue
    return max_acc


def load_series_from_results(dataset_key: str) -> list[float] | None:
    """Load accuracy series from evaluation results. dataset_key: 'blink', 'MMVP', 'HRBench4K'."""
    base = RESULTS_ROOT / dataset_key / "decoding_by_steps" / "box_resampler"
    if not base.exists() and dataset_key == "HRBench4K":
        # HRBench4K may only have stage2_distillation, not box_resampler
        base = RESULTS_ROOT / "HRBench4K" / "decoding_by_steps"
        for sub in base.iterdir():
            if sub.is_dir() and (sub / FOLDERS[0]).exists():
                base = sub
                break
        else:
            return None

    ys = []
    for folder_name in FOLDERS:
        folder_path = base / folder_name
        acc = get_max_accuracy(folder_path)
        ys.append(acc)
    if all(y is None for y in ys):
        return None
    return [y if y is not None else float('nan') for y in ys]


def main():
    datasets_config = [
        ("BLINK", "blink"),
        ("MMVP", "MMVP"),
        ("HRBench-4K", "HRBench4K"),
    ]

    series = {}  # legend_name -> (xs, ys)
    colors = ["#2563eb", "#dc2626", "#16a34a"]
    markers = ["o", "s", "^"]

    for i, (legend_name, result_key) in enumerate(datasets_config):
        custom = CUSTOM_LINE_POINTS.get(legend_name)
        if custom is not None and len(custom) == len(LATENT_DIMS):
            ys = custom
            print(f"[Custom] {legend_name}: {ys}")
        else:
            ys = load_series_from_results(result_key)
            if ys is None:
                print(f"[Skip] {legend_name}: no data found")
                continue
            print(f"[Auto]  {legend_name}: {[f'{y:.2f}' if not (y != y) else 'N/A' for y in ys]}")

        xs = LATENT_DIMS
        series[legend_name] = (xs, ys)
        # Filter out NaN for plotting
        valid = [(x, y) for x, y in zip(xs, ys) if y == y]
        if not valid:
            continue

    if not series:
        print("No data to plot.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#fafafa")
    ax.set_facecolor("#ffffff")

    # Annotate accuracy only at x=8 and x=16; MMVP above, HRBench-4K below, BLINK above
    annotate_at_x = {8, 16}
    # BLINK, MMVP, HRBench-4K: positive=above, negative=below
    y_offsets = {"BLINK": 10, "MMVP": 5, "HRBench-4K": -25}

    for i, (legend_name, (xs, ys)) in enumerate(series.items()):
        valid_xs = [x for x, y in zip(xs, ys) if y == y]
        valid_ys = [y for y in ys if y == y]
        if not valid_xs:
            continue
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        y_off = y_offsets.get(legend_name, 10)
        ax.plot(valid_xs, valid_ys, marker=marker, linewidth=2.5, markersize=10,
                label=legend_name, color=color, markeredgecolor="white", markeredgewidth=1.5)
        for x, y in zip(valid_xs, valid_ys):
            if x in annotate_at_x:
                ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, y_off),
                           ha="center", fontsize=22, fontweight="medium", color=color)

    ax.set_xlabel("Latent Steps", fontsize=24, fontweight="medium")
    ax.set_ylabel("Accuracy (%)", fontsize=24, fontweight="medium")
    ax.set_xticks(LATENT_DIMS)
    ax.set_xticklabels([str(d) for d in LATENT_DIMS], fontsize=22)
    ax.tick_params(axis="y", labelsize=22)
    ax.legend(loc="lower right", fontsize=24, frameon=True, fancybox=True,
              shadow=True, framealpha=0.95, edgecolor="#e5e7eb")
    ax.grid(True, alpha=0.4, linestyle="-")
    y_min = min(y for _, ys in series.values() for y in ys if y == y)
    ax.set_ylim(bottom=max(0, y_min - 8))
    for spine in ax.spines.values():
        spine.set_color("#e5e7eb")
        spine.set_linewidth(0.8)

    plt.subplots_adjust(left=0.12, right=0.96, top=0.95, bottom=0.12)
    out_path = PROJECT_ROOT / "evaluation/results/latent_step_vs_accuracy.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
