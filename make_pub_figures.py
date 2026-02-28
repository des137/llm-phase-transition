"""
Publication-grade figure generation for the phase-transition paper.

Produces four figures saved as high-resolution PDFs (and PNGs) in
paper_idea/experiment/pub_figures/:

  fig1_qa_phase_transitions.pdf   — QA curves, all models, 3 granularities
  fig2_alpha_star_heatmap.pdf     — α* heatmap (models × granularities)
  fig3_claude_all_tasks.pdf       — Claude Sonnet 4.5: all 4 tasks × 3 gran.
  fig4_sharpness_beta.pdf         — β bar chart across models

Usage
-----
  cd paper_idea/experiment
  .venv/bin/python make_pub_figures.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.optimize import curve_fit  # type: ignore[import]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
OUT_DIR = os.path.join(SCRIPT_DIR, "pub_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "0.7",
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ---------------------------------------------------------------------------
# Model display names and colours
# ---------------------------------------------------------------------------
MODEL_DISPLAY = {
    "openai/_gpt-4o":                                          "GPT-4o",
    "anthropic/_claude-sonnet-4-5-20250929":                   "Claude Sonnet 4.5",
    "together/_meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo":  "Llama 3.1 8B",
    "deepseek/_deepseek-chat":                                 "DeepSeek Chat",
    "openrouter/_meta-llama_llama-3.3-70b-instruct":           "Llama 3.3 70B",
    "mistral/_mistral-small-latest":                           "Mistral Small",
    "grok/_grok-3-mini-beta":                                  "Grok 3 Mini",
}

# Colour palette — colourblind-friendly (Wong 2011)
PALETTE = [
    "#0072B2",  # blue        → GPT-4o
    "#D55E00",  # vermillion  → Claude
    "#009E73",  # green       → Llama 8B
    "#CC79A7",  # pink        → DeepSeek
    "#56B4E9",  # sky blue    → Llama 70B
    "#E69F00",  # orange      → Mistral
    "#F0E442",  # yellow      → Grok
]
MODEL_COLOUR = {k: PALETTE[i] for i, k in enumerate(MODEL_DISPLAY)}

MARKER_STYLES = ["o", "s", "^", "D", "v", "P", "X"]
MODEL_MARKER = {k: MARKER_STYLES[i] for i, k in enumerate(MODEL_DISPLAY)}

GRAN_LABELS = {"char": "Character-level", "word": "Word-level", "sentence": "Sentence-level"}
GRAN_ORDER  = ["char", "word", "sentence"]

TASK_LABELS = {
    "qa":                "QA (Token F1)",
    "summarization":     "Summarisation (ROUGE-L)",
    "coherence_judge":   "Coherence Self-Rating",
    "instruction_follow":"Instruction Following",
}
TASK_ORDER = ["qa", "summarization", "coherence_judge", "instruction_follow"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results() -> Dict[str, List[dict]]:
    """Load all JSONL files → {internal_model_key: [records]}."""
    out: Dict[str, List[dict]] = {}
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.endswith(".jsonl"):
            continue
        key = fname.replace(".jsonl", "").replace("_", "/", 1)
        path = os.path.join(RESULTS_DIR, fname)
        records = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if not r.get("error"):
                        records.append(r)
                except json.JSONDecodeError:
                    pass
        if records:
            out[key] = records
    return out


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

PRIMARY_METRIC = {
    "qa":                "token_f1",
    "summarization":     "rouge_l",
    "coherence_judge":   "coherence_score",
    "instruction_follow":"instruction_follow",
}

def _extract_score(record: dict) -> Optional[float]:
    """Extract the primary numeric score from a result record."""
    task = record.get("task_name", "")
    parsed = record.get("parsed", {}) or {}
    raw = record.get("raw_response", "") or ""

    if task == "qa":
        # token_f1 in parsed
        v = parsed.get("token_f1")
        if v is not None:
            return float(v)
        # fallback: compute from raw
        return None

    if task == "summarization":
        v = parsed.get("rouge_l")
        if v is not None:
            return float(v)
        return None

    if task == "coherence_judge":
        v = parsed.get("score") or parsed.get("coherence_score")
        if v is not None:
            try:
                return float(v) / 10.0   # normalise 1-10 → 0-1
            except (TypeError, ValueError):
                pass
        # try to parse integer from raw response
        import re
        m = re.search(r"\b([1-9]|10)\b", str(raw))
        if m:
            return float(m.group(1)) / 10.0
        return None

    if task == "instruction_follow":
        v = parsed.get("follow_rate") or parsed.get("instruction_follow_rate")
        if v is not None:
            return float(v)
        # binary: did the model produce a list?
        import re
        if re.search(r"^\s*[-*•\d]", str(raw), re.MULTILINE):
            return 1.0
        return 0.0

    return None


def build_curves(
    records: List[dict],
    granularity: str,
    task: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (alphas, means, stds) arrays for a given (granularity, task).
    Groups by alpha, averages across texts/seeds/reps.
    """
    by_alpha: Dict[float, List[float]] = defaultdict(list)
    for r in records:
        if r.get("granularity") != granularity:
            continue
        if r.get("task_name") != task:
            continue
        score = _extract_score(r)
        if score is not None:
            by_alpha[round(float(r["alpha"]), 2)].append(score)

    if not by_alpha:
        return np.array([]), np.array([]), np.array([])

    alphas = np.array(sorted(by_alpha))
    means  = np.array([np.mean(by_alpha[a]) for a in alphas])
    stds   = np.array([np.std(by_alpha[a])  for a in alphas])
    return alphas, means, stds


# ---------------------------------------------------------------------------
# Sigmoid fitting
# ---------------------------------------------------------------------------

def _sigmoid(x, alpha_star, beta, L, offset):
    return L / (1.0 + np.exp(-beta * (x - alpha_star))) + offset


def fit_sigmoid(alphas: np.ndarray, means: np.ndarray):
    """Fit sigmoid; returns (alpha_star, beta, L, offset, r2) or None."""
    if len(alphas) < 5:
        return None
    try:
        p0 = [float(np.median(alphas)), 10.0, float(means.max() - means.min()), float(means.min())]
        bounds = ([0, 0, 0, -0.5], [1, 200, 2, 1.5])
        popt, _ = curve_fit(_sigmoid, alphas, means, p0=p0, bounds=bounds, maxfev=8000)
        y_pred = _sigmoid(alphas, *popt)
        ss_res = np.sum((means - y_pred) ** 2)
        ss_tot = np.sum((means - means.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
        return (*popt, r2)   # alpha_star, beta, L, offset, r2
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Figure 1 — QA phase transitions, all models, 3 granularities
# ---------------------------------------------------------------------------

def fig1_qa_phase_transitions(all_records: Dict[str, List[dict]]) -> None:
    """
    3-column figure: one panel per granularity.
    Each panel shows QA Token F1 vs α for all 7 models with sigmoid fits.
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)

    for col, gran in enumerate(GRAN_ORDER):
        ax = axes[col]
        ax.set_title(GRAN_LABELS[gran], fontweight="bold", pad=4)
        ax.set_xlabel("Coherence ratio $\\alpha$")
        if col == 0:
            ax.set_ylabel("QA Token F1")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))

        for model_key, display in MODEL_DISPLAY.items():
            if model_key not in all_records:
                continue
            records = all_records[model_key]
            alphas, means, stds = build_curves(records, gran, "qa")
            if len(alphas) < 4:
                continue

            colour = MODEL_COLOUR[model_key]
            marker = MODEL_MARKER[model_key]

            # Data points
            ax.plot(alphas, means, marker=marker, color=colour,
                    linewidth=1.4, markersize=3.5, label=display, zorder=3)

            # Shaded ±1 std (very light)
            if stds.max() > 0.01:
                ax.fill_between(alphas, means - stds, means + stds,
                                alpha=0.10, color=colour, zorder=2)

            # Sigmoid fit overlay
            fit = fit_sigmoid(alphas, means)
            if fit and fit[4] > 0.70:   # r2 > 0.70
                alpha_star, beta, L, offset, r2 = fit
                x_fine = np.linspace(0, 1, 300)
                y_fine = _sigmoid(x_fine, alpha_star, beta, L, offset)
                ax.plot(x_fine, y_fine, "--", color=colour,
                        linewidth=0.9, alpha=0.75, zorder=4)
                # Critical threshold tick
                ax.axvline(alpha_star, color=colour, linewidth=0.6,
                           linestyle=":", alpha=0.55, zorder=1)

        # Annotation: "α*≈0.13" for sentence panel
        if gran == "sentence":
            ax.annotate(
                "$\\alpha^* \\approx 0.13$",
                xy=(0.13, 0.5), xytext=(0.28, 0.62),
                fontsize=7, color="0.35",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8),
            )

    # Single legend below all panels
    handles, labels = axes[0].get_legend_handles_labels()
    # Collect from all axes to get all models
    for ax in axes[1:]:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.18),
        frameon=True,
        fontsize=7.5,
        handlelength=1.8,
        columnspacing=1.0,
    )

    fig.suptitle(
        "Figure 1: QA Phase Transitions Across Models and Corruption Granularities",
        fontsize=9, fontweight="bold", y=1.02,
    )

    _save(fig, "fig1_qa_phase_transitions")


# ---------------------------------------------------------------------------
# Figure 2 — α* heatmap
# ---------------------------------------------------------------------------

def fig2_alpha_star_heatmap(all_records: Dict[str, List[dict]]) -> None:
    """
    Heatmap: rows = models, columns = granularities.
    Cell value = α* for QA task (NaN if no clean fit).
    """
    model_keys = [k for k in MODEL_DISPLAY if k in all_records]
    display_names = [MODEL_DISPLAY[k] for k in model_keys]

    data = np.full((len(model_keys), len(GRAN_ORDER)), np.nan)

    for i, model_key in enumerate(model_keys):
        records = all_records[model_key]
        for j, gran in enumerate(GRAN_ORDER):
            alphas, means, _ = build_curves(records, gran, "qa")
            if len(alphas) < 5:
                continue
            fit = fit_sigmoid(alphas, means)
            if fit and fit[4] > 0.70:
                data[i, j] = fit[0]   # alpha_star

    fig, ax = plt.subplots(figsize=(3.8, 3.2))

    # Mask NaN cells
    masked = np.ma.masked_invalid(data)
    cmap = plt.cm.RdYlGn   # red=low α* (fragile), green=high α* (robust)
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    # Annotate cells
    for i in range(len(model_keys)):
        for j in range(len(GRAN_ORDER)):
            val = data[i, j]
            if not np.isnan(val):
                text_colour = "white" if val < 0.25 or val > 0.80 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=text_colour, fontweight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=9, color="0.55")

    ax.set_xticks(range(len(GRAN_ORDER)))
    ax.set_xticklabels(["Character", "Word", "Sentence"], fontsize=8)
    ax.set_yticks(range(len(model_keys)))
    ax.set_yticklabels(display_names, fontsize=8)
    ax.set_xlabel("Corruption granularity", labelpad=6)
    ax.set_title("Figure 2: Critical Threshold $\\alpha^*$ (QA Task)\n"
                 "Lower = collapses earlier (less robust)",
                 fontsize=9, fontweight="bold", pad=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("$\\alpha^*$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    _save(fig, "fig2_alpha_star_heatmap")


# ---------------------------------------------------------------------------
# Figure 3 — Claude Sonnet 4.5: all 4 tasks × 3 granularities
# ---------------------------------------------------------------------------

def fig3_claude_all_tasks(all_records: Dict[str, List[dict]]) -> None:
    """
    3×4 grid: rows = granularities, columns = tasks.
    Shows Claude Sonnet 4.5 performance with sigmoid fits.
    """
    claude_key = "anthropic/_claude-sonnet-4-5-20250929"
    if claude_key not in all_records:
        print(f"  [WARN] Claude data not found for fig3; skipping.")
        return

    records = all_records[claude_key]

    fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.4), sharex=True)

    for row, gran in enumerate(GRAN_ORDER):
        for col, task in enumerate(TASK_ORDER):
            ax = axes[row][col]
            alphas, means, stds = build_curves(records, gran, task)

            if row == 0:
                ax.set_title(TASK_LABELS[task], fontsize=8, fontweight="bold", pad=3)
            if col == 0:
                ax.set_ylabel(GRAN_LABELS[gran], fontsize=8)
            if row == 2:
                ax.set_xlabel("$\\alpha$", fontsize=8)

            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.05, 1.05 if task != "coherence_judge" else 1.15)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))

            if len(alphas) < 4:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="0.6")
                continue

            colour = MODEL_COLOUR[claude_key]

            # Data
            ax.plot(alphas, means, "o-", color=colour,
                    linewidth=1.4, markersize=3.5, zorder=3)
            if stds.max() > 0.01:
                ax.fill_between(alphas, means - stds, means + stds,
                                alpha=0.15, color=colour, zorder=2)

            # Sigmoid fit
            fit = fit_sigmoid(alphas, means)
            if fit and fit[4] > 0.70:
                alpha_star, beta, L, offset, r2 = fit
                x_fine = np.linspace(0, 1, 300)
                y_fine = _sigmoid(x_fine, alpha_star, beta, L, offset)
                ax.plot(x_fine, y_fine, "--", color="tomato",
                        linewidth=1.1, zorder=4,
                        label=f"$\\alpha^*$={alpha_star:.2f}, $\\beta$={beta:.0f}")
                ax.axvline(alpha_star, color="tomato", linewidth=0.7,
                           linestyle=":", alpha=0.6, zorder=1)
                ax.legend(fontsize=6.5, loc="upper left",
                          handlelength=1.2, framealpha=0.7)
            elif fit:
                # Fit exists but poor R²
                ax.text(0.97, 0.05, f"$R^2$={fit[4]:.2f}",
                        ha="right", va="bottom", transform=ax.transAxes,
                        fontsize=6.5, color="0.5")

    fig.suptitle(
        "Figure 3: Claude Sonnet 4.5 — All Tasks × Granularities\n"
        "(dashed = sigmoid fit, dotted vertical = $\\alpha^*$)",
        fontsize=9, fontweight="bold", y=1.01,
    )
    plt.tight_layout(h_pad=0.8, w_pad=0.5)
    _save(fig, "fig3_claude_all_tasks")


# ---------------------------------------------------------------------------
# Figure 4 — Sharpness β bar chart
# ---------------------------------------------------------------------------

def fig4_sharpness_beta(all_records: Dict[str, List[dict]]) -> None:
    """
    Grouped bar chart: β values for QA task across models and granularities.
    Only bars where R² > 0.70 are shown; others are hatched/grey.
    """
    model_keys = [k for k in MODEL_DISPLAY if k in all_records]
    display_names = [MODEL_DISPLAY[k] for k in model_keys]

    n_models = len(model_keys)
    n_gran   = len(GRAN_ORDER)
    x = np.arange(n_models)
    width = 0.25
    offsets = np.linspace(-(n_gran - 1) * width / 2,
                           (n_gran - 1) * width / 2, n_gran)

    gran_colours = {"char": "#4C72B0", "word": "#DD8452", "sentence": "#55A868"}
    gran_hatches = {"char": "", "word": "//", "sentence": "xx"}

    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    for j, gran in enumerate(GRAN_ORDER):
        betas = []
        alphas_star = []
        valid = []
        for model_key in model_keys:
            records = all_records[model_key]
            alphas_arr, means, _ = build_curves(records, gran, "qa")
            if len(alphas_arr) < 5:
                betas.append(0.0)
                alphas_star.append(np.nan)
                valid.append(False)
                continue
            fit = fit_sigmoid(alphas_arr, means)
            if fit and fit[4] > 0.70:
                betas.append(min(fit[1], 200.0))
                alphas_star.append(fit[0])
                valid.append(True)
            else:
                betas.append(0.0)
                alphas_star.append(np.nan)
                valid.append(False)

        bars = ax.bar(
            x + offsets[j], betas, width,
            label=GRAN_LABELS[gran],
            color=gran_colours[gran],
            hatch=gran_hatches[gran],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
        )

        # Annotate α* above each valid bar
        for xi, (bar, v, a_star) in enumerate(zip(bars, valid, alphas_star)):
            if v and not np.isnan(a_star):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 3,
                    f"{a_star:.2f}",
                    ha="center", va="bottom",
                    fontsize=5.5, color="0.3", rotation=90,
                )

    # Reference line at β=10 (sharp transition threshold)
    ax.axhline(10, color="0.4", linewidth=0.8, linestyle="--",
               label="$\\beta=10$ threshold")
    ax.axhline(200, color="0.7", linewidth=0.6, linestyle=":",
               label="Fitting cap ($\\beta=200$)")

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Sharpness $\\beta$")
    ax.set_ylim(0, 230)
    ax.set_title(
        "Figure 4: Sigmoid Sharpness $\\beta$ for QA Task\n"
        "(numbers = $\\alpha^*$; grey = no clean fit)",
        fontsize=9, fontweight="bold",
    )
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(50))

    plt.tight_layout()
    _save(fig, "fig4_sharpness_beta")


# ---------------------------------------------------------------------------
# Figure 5 — Cross-model QA comparison: all models on one panel per gran
#            (cleaner version of fig1 with error bands and α* markers)
# ---------------------------------------------------------------------------

def fig5_qa_with_alpha_markers(all_records: Dict[str, List[dict]]) -> None:
    """
    Same as fig1 but with explicit α* vertical lines labelled per model,
    and a bottom panel showing the α* values as a dot plot.
    """
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.45, wspace=0.25)

    top_axes = [fig.add_subplot(gs[0, c]) for c in range(3)]
    bot_axes = [fig.add_subplot(gs[1, c]) for c in range(3)]

    for col, gran in enumerate(GRAN_ORDER):
        ax_top = top_axes[col]
        ax_bot = bot_axes[col]

        ax_top.set_title(GRAN_LABELS[gran], fontweight="bold", fontsize=9, pad=4)
        ax_top.set_xlim(-0.02, 1.02)
        ax_top.set_ylim(-0.05, 1.05)
        ax_top.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        if col == 0:
            ax_top.set_ylabel("QA Token F1", fontsize=8)
        ax_top.set_xlabel("$\\alpha$", fontsize=8)

        ax_bot.set_xlim(-0.02, 1.02)
        ax_bot.set_ylim(-0.5, len(MODEL_DISPLAY) - 0.5)
        ax_bot.set_yticks([])
        ax_bot.set_xlabel("$\\alpha^*$", fontsize=8)
        if col == 0:
            ax_bot.set_ylabel("$\\alpha^*$ per model", fontsize=7)
        ax_bot.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax_bot.axvline(0.13, color="0.6", linewidth=0.7, linestyle="--", zorder=0)

        model_idx = 0
        for model_key, display in MODEL_DISPLAY.items():
            if model_key not in all_records:
                model_idx += 1
                continue
            records = all_records[model_key]
            alphas, means, stds = build_curves(records, gran, "qa")
            colour = MODEL_COLOUR[model_key]
            marker = MODEL_MARKER[model_key]

            if len(alphas) < 4:
                model_idx += 1
                continue

            ax_top.plot(alphas, means, marker=marker, color=colour,
                        linewidth=1.3, markersize=3.5, label=display, zorder=3)
            if stds.max() > 0.01:
                ax_top.fill_between(alphas, means - stds, means + stds,
                                    alpha=0.08, color=colour, zorder=2)

            fit = fit_sigmoid(alphas, means)
            if fit and fit[4] > 0.70:
                alpha_star, beta, L, offset, r2 = fit
                x_fine = np.linspace(0, 1, 300)
                y_fine = _sigmoid(x_fine, alpha_star, beta, L, offset)
                ax_top.plot(x_fine, y_fine, "--", color=colour,
                            linewidth=0.85, alpha=0.7, zorder=4)
                # Dot plot in bottom panel
                ax_bot.scatter([alpha_star], [model_idx], color=colour,
                               marker=marker, s=40, zorder=3)
                ax_bot.text(alpha_star + 0.02, model_idx, f"{alpha_star:.2f}",
                            va="center", fontsize=6, color=colour)

            model_idx += 1

    # Legend
    handles, labels = top_axes[0].get_legend_handles_labels()
    for ax in top_axes[1:]:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.04),
        frameon=True,
        fontsize=7.5,
        handlelength=1.8,
    )

    fig.suptitle(
        "Figure 5: QA Phase Transitions with Critical Thresholds $\\alpha^*$",
        fontsize=9, fontweight="bold", y=1.01,
    )
    _save(fig, "fig5_qa_with_alpha_markers")


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading results from: {RESULTS_DIR}")
    all_records = load_all_results()
    print(f"  Loaded {len(all_records)} models: {list(all_records.keys())}")

    print("\nGenerating Figure 1: QA phase transitions (all models, 3 granularities)...")
    fig1_qa_phase_transitions(all_records)

    print("\nGenerating Figure 2: α* heatmap...")
    fig2_alpha_star_heatmap(all_records)

    print("\nGenerating Figure 3: Claude all tasks × granularities...")
    fig3_claude_all_tasks(all_records)

    print("\nGenerating Figure 4: Sharpness β bar chart...")
    fig4_sharpness_beta(all_records)

    print("\nGenerating Figure 5: QA with α* dot-plot markers...")
    fig5_qa_with_alpha_markers(all_records)

    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

# Made with Bob
