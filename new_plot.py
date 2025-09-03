# plot_policy_grid.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Config (edit this only)
# -----------------------

csv_path = "csvs/section3.2aa.csv"  # <-- your CSV
# betas = [0.0001,0.00025,0.0005,0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.5,5.0]

betas = [0.001,0.01,0.1,1.0]

out_path = "plots/" + csv_path.split("/")[-1].replace(".csv", "_policy_grid.png")
dpi = 300

# YLIM = (0.0, 0.5)  # unified y axis for policy plots; set to None for auto
YLIM = None

# -----------------------
# Helpers
# -----------------------
def _closest_match(value, choices, tol=1e-12):
    for c in choices:
        if abs(value - c) <= tol * max(1.0, abs(value), abs(c)):
            return c
    return None

def _floatify(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _base_dir_from_plot_path(p):
    # e.g., ".../<run_dir>/final_policy_distribution.png" -> "<run_dir>"
    return os.path.dirname(str(p)) if isinstance(p, str) else "."

def _load_vector_txt(path):
    """
    Loads one float per line from a text file (scientific notation OK).
    Returns 1D numpy array or None if missing/invalid.
    """
    if path and os.path.exists(path):
        try:
            arr = np.loadtxt(path, dtype=float)
            arr = np.asarray(arr).flatten()
            if arr.size > 0:
                return arr
        except Exception:
            pass
    return None

def _nice_beta_str(b):
    if b < 1e-3 or b > 1:
        return f"{b:.0e}"
    elif abs(b - round(b, 3)) < 1e-12:
        s = f"{b:.3f}".rstrip('0').rstrip('.')
        return s
    else:
        return f"{b:g}"

# def _plot_vector(ax, vec, title=None, linewidth=1.0, as_bars=False, ylim=None, color=None):
#     ax.cla()
#     if vec is None:
#         ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=9)
#         ax.set_axis_off()
#         return

#     # Normalize if probability-like
#     if np.all(vec >= 0) and vec.sum() > 0:
#         s = vec.sum()
#         if not np.isclose(s, 1.0, rtol=1e-4, atol=1e-6):
#             vec = vec / s

#     x = np.arange(len(vec))
#     if as_bars:
#         ax.bar(x, vec, width=10, color=color)
#     else:
#         ax.plot(x, vec, linewidth=linewidth)

#     # Title
#     if title:
#         ax.set_title(title, fontsize=9, pad=2)

#     # Unified y axis if requested
#     if ylim is not None:
#         ax.set_ylim(*ylim)

#     # Keep ticks (for grid), but hide labels/marks
#     ax.set_axisbelow(True)  # grid under bars/lines
#     ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))   # sensible # of y gridlines
#     ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False, length=0)

#     # --- X ticks with numbers (0, 25, 50, ...) ---
#     # n = len(vec)
#     # step = max(1, n // 4)   # aim for ~4 ticks (adjust if you want more/less)
#     # xticks = np.arange(0, n+1, step)
#     # ax.set_xticks(xticks)
#     # ax.set_xticklabels([str(x) for x in xticks], fontsize=6)

#     # --- Y ticks (optional small numbers) ---
#     # ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
#     # ax.tick_params(axis="y", labelsize=6, length=2)

#     # label
#     ax.tick_params(labelbottom=False, labelleft=False)

#     # Grid
#     ax.grid(True, axis="y", which="major", linestyle="--", linewidth=0.5, alpha=0.6)

#     # Border
#     for s in ax.spines.values():
#         s.set_visible(True)
#         s.set_linewidth(0.8)
#         s.set_color("black")
def _plot_vector(ax, vec, title=None, linewidth=1.0, as_bars=False, ylim=None,
                 color=None, show_yticks=False):
    ax.cla()
    if vec is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=9)
        ax.set_axis_off()
        return

    # Normalize if probability-like
    # if np.all(vec >= 0) and vec.sum() > 0:
    #     s = vec.sum()
    #     if not np.isclose(s, 1.0, rtol=1e-4, atol=1e-6):
    #         vec = vec / s

    x = np.arange(len(vec))
    if as_bars:
        ax.bar(x, vec, width=10, color=color)
    else:
        ax.plot(x, vec, linewidth=linewidth)

    if title:
        ax.set_title(title, fontsize=9, pad=2)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
    ax.grid(True, axis="y", which="major", linestyle="--", linewidth=0.5, alpha=0.6)

    # Default: hide tick labels
    ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False, length=0)

    # Optionally show y-ticks
    if show_yticks:
        from matplotlib.ticker import FormatStrFormatter
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        def no_zero(val, pos):
            return "" if abs(val) < 1e-12 else f"{val:g}"
        ax.yaxis.set_major_formatter(plt.FuncFormatter(no_zero))
        ax.tick_params(axis="y", which="major", labelleft=True, length=2, labelsize=5)
        # pass
        # optional: format as 0.00

    # Border
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color("black")

# -----------------------
# Load and preprocess
# -----------------------
df = pd.read_csv(csv_path)
req_cols = {"kl_coeff", "kl_grad_type", "final_policy_plot_path"}
if not req_cols.issubset(df.columns):
    raise ValueError("CSV must have columns: kl_grad_type, kl_coeff, final_policy_plot_path")

df["kl_coeff_f"] = df["kl_coeff"].apply(_floatify)

# KL type names
rev_name = "rev-vanilla"
# fwd_name = "fwd-simple"
fwd_name = "schulman-lowvar"

# Determine a base folder (the first row’s run dir); adjust if you want a specific folder
first_dir = _base_dir_from_plot_path(df["final_policy_plot_path"].iloc[0])

# NEW FILENAMES (as per your message)
reward_txt_path = os.path.join(first_dir, "reward_values.txt")                    # was reward_values.txt
ref_probs_txt_path = os.path.join(first_dir, "reference_policy_probs.txt")  # unchanged from previous reply

# Ensure betas exist in CSV
avail_betas = set(df["kl_coeff_f"].dropna().tolist())
resolved_betas = []
for b in betas:
    m = _closest_match(b, avail_betas)
    if m is None:
        print(f"[warn] KL coeff {b} not found in CSV; skipping this column.")
    else:
        resolved_betas.append(m)
if not resolved_betas:
    raise ValueError("None of the requested betas were found in the CSV.")

# -----------------------
# Build figure: 2 rows x (1 + N) cols
# row 0: Reward | Rev KL @ betas...
# row 1: RefPol | Fwd KL @ betas...
# -----------------------
ncols = 1 + len(resolved_betas)
nrows = 2

# fig_w = 1.2 + 1.45*len(resolved_betas)
fig_w = 2 + 1.45*len(resolved_betas)
fig_h = 2.9
fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=dpi)
# fig, axes = plt.subplots(
#     nrows, ncols, figsize=(fig_w, fig_h), dpi=dpi,
#     gridspec_kw={'width_ratios': [1.0, 0.1] + [1.0] * (ncols - 2)}
# )
if nrows == 1:
    axes = np.array([axes])
if ncols == 1:
    axes = axes.reshape(nrows, 1)

# Turn off everything initially
for r in range(nrows):
    for c in range(ncols):
        axes[r, c].set_axis_off()

# First column: reward/ref plotted from txt
reward_vec = _load_vector_txt(reward_txt_path)
ref_vec = _load_vector_txt(ref_probs_txt_path)

axes[0, 0].set_axis_on()
_plot_vector(axes[0, 0], reward_vec, title="Reward Function", show_yticks=True)
axes[1, 0].set_axis_on()
_plot_vector(axes[1, 0], ref_vec, title="Reference Policy", show_yticks=True)

# Column titles (betas) along the first row of KL plots
# for j, b in enumerate(resolved_betas, start=1):
#     b_str = _nice_beta_str(b)
#     axes[0, j].set_title(r"$\beta$ = " + b_str, fontsize=9, pad=5)

# Centered row headers (outside the grid)
# fig.text(0.5, 0.965, "Reverse KL", fontsize=12, ha="center", va="top")
# fig.text(0.5, 0.49,  "Forward KL", fontsize=12, ha="center", va="top")

# Optional thin vertical separator between first column and the rest
# make line longer

# sep_x = (1.025 / ncols)
# fig.lines.append(
#     plt.Line2D(
#         [sep_x, sep_x],
#         [0.001, 0.999],   # <-- longer vertical span
#         transform=fig.transFigure,
#         linewidth=0.9,
#         color="gray"
#     )
# )

# Fill policy plots for each (row, beta) by reading final_policy_probs.txt
def _final_probs_path_from_row(row):
    run_dir = _base_dir_from_plot_path(row["final_policy_plot_path"])
    return os.path.join(run_dir, "final_policy_probs.txt")  # NEW source: txt per run

# Make room at the top for figure-level labels

# --- Align beta labels for BOTH rows using shared column centers ---
# Make sure layout is final before measuring positions
plt.tight_layout()
fig.canvas.draw()

# Give some breathing room so labels aren't clipped
plt.subplots_adjust(top=0.93, bottom=0.08)


# --- Add horizontal separator between Reverse KL and Forward KL rows ---
# Get left and right bounds of KL columns
# left = axes[0, 1].get_position().x0
# right = axes[0, ncols-1].get_position().x1

# # Y coordinate: use the bottom of the top row / top of the bottom row
# y_sep = axes[0, 1].get_position().y0   # bottom of Reverse KL row

# fig.lines.append(
#     plt.Line2D([left, right], [y_sep, y_sep],
#                transform=fig.transFigure,
#                linewidth=1.0, color="gray", linestyle="-")
# )
# fig.canvas.draw()
# x_sep = axes[0, 1].get_position().x0 - 0.01  # small nudge if you like
# fig.lines.append(plt.Line2D([x_sep, x_sep], [0.001, 0.999],
#                             transform=fig.transFigure, linewidth=0.9, color="gray"))


### Beta plotting ########################

# # Column centers: use TOP row for all columns so both rows share the same x
# col_centers = []
# top_y = []
# bot_y = []
# for j in range(1, ncols):  # skip first "Reward/Reference" column
#     p_top = axes[0, j].get_position()
#     p_bot = axes[1, j].get_position()
#     col_centers.append(0.5 * (p_top.x0 + p_top.x1))
#     top_y.append(p_top.y1)
#     bot_y.append(p_bot.y1)

# pad_top = +0.01   # space above the axes for labels
# pad_mid = -0.02   # same for the bottom row

# # Labels above the Reverse-KL row
# for idx, b in enumerate(resolved_betas, start=1):
#     fig.text(col_centers[idx-1], top_y[idx-1] + pad_top,
#              r"$\beta$ = " + _nice_beta_str(b),
#              ha="center", va="bottom", fontsize=9)

# # Labels above the Forward-KL row (aligned x with the top row)
# for idx, b in enumerate(resolved_betas, start=1):
#     fig.text(col_centers[idx-1], bot_y[idx-1] + pad_mid,
#              r"$\beta$ = " + _nice_beta_str(b),
#              ha="center", va="bottom", fontsize=9)
# --- put beta labels per-axes so they always align ---

############################################

############### Label Forward KL and Reverse KL ##################

# Horizontal span: from first KL col to last KL col
left = axes[0, 1].get_position().x0
right = axes[0, ncols-1].get_position().x1
center_x = 0.5 * (left + right)

# Reverse KL row header (row 0, orange bars)
top0 = axes[0, 1].get_position().y1
fig.text(center_x, top0 + 0.08, "Reverse KL",
         ha="center", va="bottom", fontsize=11, fontweight="bold")

# Forward KL row header (row 1, green bars)
top1 = axes[1, 1].get_position().y1
fig.text(center_x, top1 + 0.025, "Forward KL",
         ha="center", va="bottom", fontsize=11, fontweight="bold")

############################################

for j, b in enumerate(resolved_betas, start=1):

    # Reverse KL row
    rev_row = df[(df["kl_grad_type"] == rev_name) & (np.isfinite(df["kl_coeff_f"]))]
    rev_row = rev_row[np.isclose(rev_row["kl_coeff_f"], b, rtol=1e-12, atol=0.0)]
    axes[0, j].set_axis_on()
    if len(rev_row) > 0:
        fpath = _final_probs_path_from_row(rev_row.iloc[0])
        vec = _load_vector_txt(fpath)
        _plot_vector(
            axes[0, j], vec, title=None, as_bars=True, ylim=YLIM, color='C1',
            # show_yticks=(j == 1)  # show on first KL column only
            show_yticks=True
        )
    else:
        axes[0, j].text(0.5, 0.5, "not found", ha="center", va="center", fontsize=9)
        for s in axes[0, j].spines.values():
            s.set_visible(False)
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])

    # Forward KL row
    fwd_row = df[(df["kl_grad_type"] == fwd_name) & (np.isfinite(df["kl_coeff_f"]))]
    fwd_row = fwd_row[np.isclose(fwd_row["kl_coeff_f"], b, rtol=1e-12, atol=0.0)]
    axes[1, j].set_axis_on()
    if len(fwd_row) > 0:
        fpath = _final_probs_path_from_row(fwd_row.iloc[0])
        vec = _load_vector_txt(fpath)
        _plot_vector(
            axes[1, j], vec, title=None, as_bars=True, ylim=YLIM, color='C2',
            # show_yticks=(j == 1)  # show on first KL column only
            show_yticks=True
        )
    else:
        axes[1, j].text(0.5, 0.5, "not found", ha="center", va="center", fontsize=9)
        for s in axes[1, j].spines.values():
            s.set_visible(False)
        axes[1, j].set_xticks([]); axes[1, j].set_yticks([])

# Compact spacing (reduced space b/w rows)
# plt.subplots_adjust(
#     top=0.93, bottom=0.08,
#     wspace=0.25,  # spacing between columns
#     hspace=0.45   # <-- increase this (default is ~0.2)
# )
# plt.subplots_adjust(
#     top=0.93, bottom=0.08,
#     left=0.08, right=0.98,  # <-- added padding for y-tick labels
#     wspace=0.25,
#     hspace=0.45
# )
plt.subplots_adjust(
    top=0.93,
    bottom=0.10,   # more space for x-axis padding
    left=0.15,     # more space for y-tick labels
    right=0.98,    # keep a little margin on the right
    wspace=0.40,   # wider spacing between columns
    hspace=0.60    # taller spacing between rows
)
# plt.tight_layout()

# --- Add beta labels tied to subplots (robust) ---
fig.canvas.draw()
for j, b in enumerate(resolved_betas, start=1):
    beta_txt = r"$\beta$ = " + _nice_beta_str(b)
    axes[0, j].text(0.5, 1.08, beta_txt,
                    transform=axes[0, j].transAxes,
                    ha="center", va="bottom", fontsize=9)
    axes[1, j].text(0.5, 1.05, beta_txt,
                    transform=axes[1, j].transAxes,
                    ha="center", va="bottom", fontsize=9)
    
x_sep = axes[0, 1].get_position().x0 - 0.0345  # small nudge if you like
fig.lines.append(plt.Line2D([x_sep, x_sep], [0.0001, 0.9999],
                            transform=fig.transFigure, linewidth=0.9, color="gray"))


# --- Add horizontal separator between Reverse KL and Forward KL rows ---

# Get left/right span of KL columns (skip the first "Reward/Reference" column)
left = axes[0, 1].get_position().x0
right = axes[0, ncols-1].get_position().x1

# Separator y: bottom of the top row (Reverse KL) = top of the gap
y_sep = axes[0, 1].get_position().y0 - 0.05  # small nudge if you like

fig.lines.append(
    plt.Line2D([left, right], [y_sep, y_sep],
               transform=fig.transFigure,
               linewidth=1.0, color="gray", linestyle="-")
)

# Save
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
print(f"Saved figure to: {out_path}")