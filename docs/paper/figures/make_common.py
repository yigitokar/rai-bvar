"""Shared helpers for the figure scripts ``make_fig_*.py`` (not a figure script itself).

Every script writes one PDF, ``docs/paper/figures/<name>.pdf``, which the paper includes as
``\\includegraphics{figures/<name>.pdf}``; ``<name>`` is the script stem of OUTLINE.md §14
(``fig_architecture``, ``fig_lpd_estimators``, ...). A figure whose data does not exist yet is
written as a clearly marked placeholder (:func:`placeholder`) so that the paper compiles.

Colours are the validated reference palette of the dataviz method (light mode, categorical slots
in their fixed order; diverging blue/red around a neutral grey; status colours never reused for
series). Lines are solid; series are told apart by colour *and* marker or direct label.

Run the scripts with the main repository's environment, without writing byte code into it::

    PYTHONDONTWRITEBYTECODE=1 /Users/yigitokar/rai-bvar/.venv/bin/python make_fig_<name>.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent  # docs/paper/figures
PAPER = HERE.parent  # docs/paper
DATA = HERE / "data"
MAIN_REPO = Path(os.environ.get("RAI_BVAR_REPO", "/Users/yigitokar/rai-bvar"))

# text width of the paper (letterpaper, 1in margins)
TEXTWIDTH = 6.5

# --- palette (reference instance, light mode) ---------------------------------------------------
SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#6250d6",  # 7 violet
    "#e34948",  # 8 red
]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
NEUTRAL = "#f0efec"  # diverging midpoint
SURFACE = "#ffffff"  # print: the page itself
# sequential blue ramp (100 .. 700)
BLUE = {100: "#cde2fb", 200: "#9ec5f4", 300: "#6da7ec", 400: "#3987e5", 450: "#2a78d6",
        500: "#256abf", 600: "#184f95", 700: "#0d366b"}  # fmt: skip
# the red arm of the diverging pair, from neutral to the pole
RED_ARM = ["#f0efec", "#f6c9c4", "#f09a92", "#e86a63", "#e34948", "#b8302f", "#8a1f1f"]
BLUE_ARM = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
STATUS_CRITICAL = "#d03b3b"


def style() -> None:
    """Matplotlib defaults for print figures: 8-pt sans, hairline grid, no top/right spines."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "axes.labelcolor": INK2,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "pdf.fonttype": 42,  # embed TrueType, keeps text selectable
            "text.color": INK,
        }
    )


OUTDIR = HERE  # set by --outdir (dry runs write to a scratch directory, never over the paper's PDFs)


def out_path(name: str) -> Path:
    return OUTDIR / f"{name}.pdf"


def save(fig, name: str) -> Path:
    path = out_path(name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {path.relative_to(PAPER) if path.is_relative_to(PAPER) else path}")
    return path


def placeholder(name: str, fig_id: str, title: str, needs: str, command: str,
                width: float = TEXTWIDTH, height: float = 2.4) -> Path:  # fmt: skip
    """A clearly marked placeholder PDF: the figure id, what it will show, what data it waits
    for, and the command that replaces it."""
    style()
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.add_patch(
        plt.Rectangle((0.01, 0.03), 0.98, 0.94, transform=ax.transAxes, facecolor="#f7f6f3",
                      edgecolor=STATUS_CRITICAL, linewidth=1.2)  # fmt: skip
    )
    import textwrap

    def wrapped(s: str, size: float) -> str:  # stay inside the frame (matplotlib's wrap does not)
        return "\n".join(textwrap.wrap(s, width=max(20, int(width * 72 * 0.9 / (0.6 * size)))))

    ax.text(0.5, 0.78, f"PLACEHOLDER  {fig_id}", ha="center", va="center", fontsize=13,
            fontweight="bold", color=STATUS_CRITICAL, transform=ax.transAxes)  # fmt: skip
    ax.text(0.5, 0.58, wrapped(title, 8.5), ha="center", va="center", fontsize=8.5, color=INK,
            transform=ax.transAxes)  # fmt: skip
    ax.text(0.5, 0.38, wrapped(f"Waiting for: {needs}", 7.5), ha="center", va="center",
            fontsize=7.5, color=INK2, transform=ax.transAxes)  # fmt: skip
    ax.text(0.5, 0.17, wrapped(f"Regenerate: {command}", 7), ha="center", va="center", fontsize=7,
            color=INK2, family="monospace", transform=ax.transAxes)  # fmt: skip
    return save(fig, name)


REPO_ROOT = PAPER.parent.parent  # the repository that holds docs/paper (branch paper-draft)
# the release validation records: slow tiers at 7f880e1 on an Apple M5 Max (main 3d5a407, PR #9)
RELEASE_DIR = REPO_ROOT / "benchmarks" / "reports" / "validation" / "7f880e1_apple-m5-max"
RELEASE_RECORD = RELEASE_DIR / "record"


def record_status_note(fig, paths) -> None:
    """A grey footer naming the validation record a figure was drawn from: the release record at
    7f880e1 (commit, machine and date from its ``meta.json``), or, for any other input, a warning
    that it is not the release record."""
    paths = [Path(p).resolve() for p in paths]
    names = ", ".join(p.name for p in paths)
    if all(p.is_relative_to(RELEASE_RECORD.resolve()) for p in paths):
        meta = json.loads((RELEASE_DIR / "meta.json").read_text())
        assert meta["dirty"] is False, meta
        text = (f"Data: {names}, release validation record at commit {meta['commit'][:7]} (clean tree), "
                f"{meta['machine']['cpu']}, {meta['started_utc'][:10]} "
                "(benchmarks/reports/validation/7f880e1_apple-m5-max/record/).")
    else:
        text = f"Draft data: {names}; not the release validation record."
    fig.text(0.0, -0.01, text, ha="left", va="top", fontsize=5.6, color=MUTED, wrap=True)


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main_repo_on_path(*sub: str) -> None:
    """Make the main repository's sources (and, if asked, its test helpers) importable, read-only.
    Byte code is never written (``sys.dont_write_bytecode``)."""
    sys.dont_write_bytecode = True
    for s in ("src", *sub):
        p = str(MAIN_REPO / s)
        if p not in sys.path:
            sys.path.insert(0, p)


def direct_label(ax, x, y, text, color=INK2, **kw):
    """A direct label in text ink (never the series colour) next to a mark."""
    kw.setdefault("fontsize", 7)
    kw.setdefault("va", "center")
    return ax.text(x, y, text, color=color, **kw)


# --- box-and-arrow diagrams (Fig1, Fig3) ---------------------------------------------------------
def box(ax, x, y, w, h, title, body="", *, face="#f4f3ef", edge=BASELINE, title_color=INK,
        fontsize=7.0, body_size=6.2, mono_body=False, round_=0.04, lw=0.8):  # fmt: skip
    """A rounded box centred at (x, y) in data coordinates, with a bold title and a body."""
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={round_}",
                       facecolor=face, edgecolor=edge, linewidth=lw, zorder=2)  # fmt: skip
    )
    if body:
        ax.text(x, y + h / 2 - 0.06, title, ha="center", va="top", fontsize=fontsize,
                fontweight="bold", color=title_color, zorder=3)  # fmt: skip
        ax.text(x, y + h / 2 - 0.06 - fontsize / 72 * 1.5, body, ha="center", va="top",
                fontsize=body_size, color=INK2, zorder=3, linespacing=1.25,
                family="monospace" if mono_body else None)  # fmt: skip
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=fontsize, fontweight="bold",
                color=title_color, zorder=3)  # fmt: skip


def arrow(ax, xy_from, xy_to, *, color=MUTED, lw=0.9, label="", label_xy=None, label_ha="left",
          label_size=6.0, style="-|>", connection="arc3"):  # fmt: skip
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=0, shrinkB=0,
                                mutation_scale=7, connectionstyle=connection))  # fmt: skip
    if label:
        lx, ly = label_xy if label_xy else ((xy_from[0] + xy_to[0]) / 2, (xy_from[1] + xy_to[1]) / 2)
        ax.text(lx, ly, label, ha=label_ha, va="center", fontsize=label_size, color=INK2,
                zorder=4, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.6))  # fmt: skip


# --- the Clark-Ravazzolo acceptance report ----------------------------------------------------------
ACCEPTANCE_JSON = MAIN_REPO / "benchmarks" / "reports" / "clark_ravazzolo_acceptance.json"
FIRST_RUN_JSON = MAIN_REPO / "benchmarks" / "reports" / "clark_ravazzolo.json"  # 551f181, history


# The header test lives in acceptance_check.py (no dependencies), which docs/paper/fill_acceptance.py
# shares, so that the figures and the \acc values apply the same test.
from acceptance_check import acceptance_problems  # noqa: E402,F401


def add_common_args(ap, *, acc: bool = True) -> None:
    """--acc PATH, --allow-non-acceptance (dry runs on another report) and --outdir DIR."""
    if acc:
        ap.add_argument("--acc", type=Path, default=None,
                        help="the acceptance report JSON (benchmarks/reports/"
                        "clark_ravazzolo_acceptance.json in the main repository)")
        ap.add_argument("--allow-non-acceptance", action="store_true",
                        help="plot a report that is not the registered acceptance run (dry runs "
                        "only; use with --outdir outside the paper)")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="write the PDF here instead of docs/paper/figures/")


def apply_outdir(args) -> None:
    global OUTDIR
    if getattr(args, "outdir", None):
        args.outdir.mkdir(parents=True, exist_ok=True)
        OUTDIR = args.outdir.resolve()
    if getattr(args, "allow_non_acceptance", False) and OUTDIR == HERE:
        raise SystemExit("--allow-non-acceptance needs --outdir outside docs/paper/figures/: a "
                         "dry run must never replace the paper's figure")


def load_acceptance(args) -> dict | None:
    """The acceptance report, or None when it is not given or does not exist (the caller then
    writes the placeholder). A report that fails the header checks is refused unless
    ``--allow-non-acceptance`` (dry runs)."""
    path = getattr(args, "acc", None)
    if path is None or not Path(path).exists():
        return None
    rep = json.loads(Path(path).read_text())
    probs = acceptance_problems(rep)
    if probs and not args.allow_non_acceptance:
        raise SystemExit(f"{path} is not the registered acceptance report: " + "; ".join(probs))
    if probs:
        print(f"warning: dry run on a non-acceptance report ({'; '.join(probs)})")
    return rep


PER_ORIGIN_CSV = DATA / "cr_acceptance" / "cr_acceptance_per_origin.csv"  # make_data_cr_per_origin.py


def load_per_origin(args):
    """The per-origin export (make_data_cr_per_origin.py) as a DataFrame, or None when it does not
    exist. Its side file must name the acceptance config, unless --allow-non-acceptance."""
    import pandas as pd

    path = Path(args.csv)
    if not path.exists():
        return None
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if (meta.get("config") != "acceptance" or meta.get("lpd_method") != "integrated") \
            and not args.allow_non_acceptance:
        raise SystemExit(f"{path} is not an export of the acceptance run (meta: {meta})")
    df = pd.read_csv(path, float_precision="round_trip")  # the export writes %.17g: exact
    df["period"] = pd.PeriodIndex(df["origin"], freq="Q")
    df.attrs["lpd_method"] = meta.get("lpd_method", "unknown")
    return df


def add_per_origin_args(ap) -> None:
    ap.add_argument("--csv", type=Path, default=PER_ORIGIN_CSV)
    ap.add_argument("--allow-non-acceptance", action="store_true")
    ap.add_argument("--outdir", type=Path, default=None)


def quarter_axis(ax, periods) -> None:
    """Year ticks every 5 years on an axis indexed by quarterly periods (as floats)."""
    years = sorted({p.year for p in periods if p.year % 5 == 0 and p.quarter == 1})
    ax.set_xticks([p_to_x(f"{y}Q1") for y in years])
    ax.set_xticklabels([str(y) for y in years])


def p_to_x(p) -> float:
    import pandas as pd

    q = pd.Period(p, freq="Q")
    return q.year + (q.quarter - 1) / 4
