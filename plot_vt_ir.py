#!/usr/bin/env python3
"""
plot_vt_ir.py -- Variable-temperature IR (VT-IR) spectrum plotter.

Reads a folder of VT-IR spectra produced by the ACH-VT-IR-Wizard and draws a
publication-style figure. It understands the wizard's file-naming convention,
reads Thermo OMNIC ``.SPA`` files natively (no CSV export needed), and groups
spectra by scan direction (heating "up" vs. cooling "down").

Three display modes (``--mode``):

    overlay   all spectra on a shared baseline (no vertical offset)
    stack     a fixed vertical offset between spectra (waterfall)
    updown    two stacked panels -- up-scan (heating) and down-scan (cooling)

Input formats (auto-detected per file by extension):

    .csv            semicolon-delimited, European-decimal (wizard CSV export)
    .spa            Thermo OMNIC binary -- read natively
    .jdx / .dx      JCAMP-DX

Unit handling (Absorbance vs. Transmittance) is resolved hierarchically, which
matters because a bare CSV does not record which one it is:

    1. ``--input-units``    manual override always wins (one value for all files,
                            or a comma/space-separated list, one token per file).
    2. ``.SPA``             the OMNIC header stores the y data-type natively
                            (17 = absorbance, 16 = %transmittance, ...).
    3. ``.jdx``             uses the JCAMP ``YUNITS`` field if present.
    4. background files     single-beam by definition (naming convention).
    5. ``.csv`` samples     scale-free heuristic on the value distribution:
                            baseline at the bottom (peaks up) -> Absorbance;
                            baseline at the top (dips down)   -> Transmittance.

The desired *display* unit is set with ``--unit {A,T}``; absorbance and
transmittance are interconverted as needed.

Usage examples:

    python plot_vt_ir.py                       # plot ./ as a waterfall
    python plot_vt_ir.py data/ --mode updown   # split heating / cooling
    python plot_vt_ir.py --mode overlay --unit T
    python plot_vt_ir.py --list                # just report what was found
    python plot_vt_ir.py --input-units A       # force every file to absorbance
    python plot_vt_ir.py --silent --ext png    # render to <folder>.png, no window

Refactored from the author's original ``plot_IR.py`` template.
"""
from __future__ import annotations

import argparse
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib import colormaps


# --------------------------------------------------------------------------- #
# File-naming convention (emitted by ACH-VT-IR-Wizard)
#
#   [NN_]BG_<sample>_<T>C.<ext>                    background (single beam)
#   [NN_]<sample>_<T>C[_up|_down|_return].<ext>    sample spectrum
#
# The leading ``NN_`` is an optional chronological index.
# --------------------------------------------------------------------------- #
NAME_RE = re.compile(
    r"^(?:(?P<idx>\d+)_)?"
    r"(?P<bg>BG_)?"
    r"(?P<sample>.+?)_"
    r"(?P<temp>-?\d+(?:[.,]\d+)?)C"
    r"(?:_(?P<direction>up|down|return))?$",
    re.IGNORECASE,
)

SUPPORTED_EXTS = (".csv", ".spa", ".jdx", ".dx", ".jcm", ".txt")

# OMNIC SPA y data-type code -> internal unit token (per spectrochempy).
OMNIC_YCODE: Dict[int, str] = {
    17: "A",     # absorbance
    16: "T",     # %transmittance
    11: "R%",    # reflectance (percent)
    12: "logR",  # log(1/R)
    15: "SB",    # single beam
    20: "KM",    # Kubelka-Munk
    21: "R%",    # reflectance
    22: "IFG",   # detector signal / interferogram (V)
    26: "PA",    # photoacoustic
    31: "Raman",
}

UNIT_LABEL: Dict[str, str] = {
    "A": r"$\mathrm{Absorbance}\ /\ \mathrm{arb.\ units}$",
    "T": r"$\mathrm{Transmittance}\ /\ \%$",
    "SB": r"$\mathrm{Single\text{-}beam\ intensity}$",
    "R%": r"$\mathrm{Reflectance}\ /\ \%$",
    "logR": r"$\log(1/R)$",
    "KM": r"$\mathrm{Kubelka\text{-}Munk}$",
    "IFG": r"$\mathrm{Interferogram}\ /\ \mathrm{V}$",
    "PA": r"$\mathrm{Photoacoustic}$",
    "Raman": r"$\mathrm{Raman\ intensity}$",
    "INT": r"$\mathrm{Intensity}$",
}

# How the up / down / return scan directions are split into the two panels.
DIRECTION_LABEL = {"up": "heating (up)", "down": "cooling (down)", "return": "return"}


# --------------------------------------------------------------------------- #
# Spectrum container
# --------------------------------------------------------------------------- #
@dataclass
class Spectrum:
    path: Path
    x: np.ndarray            # wavenumber, ascending
    y: np.ndarray
    sample: str
    temperature: float
    direction: Optional[str]  # 'up' | 'down' | 'return' | None (background)
    kind: str                 # 'sample' | 'bg'
    index: Optional[int]      # chronological NN_ prefix, if present
    unit: str = "A"           # internal unit token; see UNIT_LABEL
    unit_source: str = ""     # how the unit was decided (for the report)

    @property
    def order_key(self) -> Tuple[int, float]:
        # Order by acquisition (NN_ index) when available, else by temperature.
        idx = self.index if self.index is not None else 10 ** 9
        return (idx, self.temperature)


# --------------------------------------------------------------------------- #
# Readers -- each returns (x, y, native_unit_or_None)
# --------------------------------------------------------------------------- #
def read_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """Two-column (wavenumber, intensity) text. Auto-detects the ``;`` / ``,`` /
    whitespace delimiter and European (comma) decimals, and skips any
    non-numeric header lines. A bare CSV carries no unit information."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty file")

    # Pick a delimiter. ';' and tab are unambiguous; a lone ',' is only a
    # delimiter when the decimals are dots (otherwise it is the decimal mark).
    # Sniff from the first line that actually holds numbers, so a header row
    # doesn't throw off the guess.
    delim: Optional[str] = None
    probe = next((ln for ln in lines if any(c.isdigit() for c in ln)), lines[0])
    if ";" in probe:
        delim = ";"
    elif "\t" in probe:
        delim = "\t"
    elif probe.count(",") == 1 and "." in probe:
        delim = ","

    rows: List[Tuple[str, str]] = []
    for ln in lines:
        parts = ln.split(delim) if delim else ln.split()
        if len(parts) < 2:
            continue
        a, b = parts[0].strip(), parts[1].strip()
        try:
            float(a.replace(",", "."))
        except ValueError:
            continue  # header / comment line
        rows.append((a, b))
    if not rows:
        raise ValueError("no numeric rows found")

    arr = np.array([[c.replace(",", ".") for c in r] for r in rows], dtype=float)
    return arr[:, 0], arr[:, 1], None


def read_spa(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """Thermo OMNIC ``.SPA`` binary. Walks the section table to find the
    spectral-header block (key 2: point count + first/last wavenumber + y
    data-type code) and the intensity block (key 3: float32 values).

    Tuned for the iS5 output of ACH-VT-IR-Wizard; matches spectrochempy's
    layout (header key read as uint8 at +8 for x-units, +12 for y-units)."""
    raw = Path(path).read_bytes()
    n = len(raw)
    nx = first_x = last_x = None
    ycode: Optional[int] = None
    data_off = data_size = None

    pos = 304  # section table start
    for _ in range(64):
        if pos + 10 > n:
            break
        key = struct.unpack_from("<H", raw, pos)[0]
        off = struct.unpack_from("<I", raw, pos + 2)[0]
        size = struct.unpack_from("<I", raw, pos + 6)[0]
        if key == 0 and off == 0:
            break
        if key == 2 and 0 < off < n:           # spectral header
            nx = struct.unpack_from("<I", raw, off + 4)[0]
            ycode = raw[off + 12]               # y data-type code (uint8)
            first_x = struct.unpack_from("<f", raw, off + 16)[0]
            last_x = struct.unpack_from("<f", raw, off + 20)[0]
        elif key == 3 and 0 < off < n:         # intensities
            data_off, data_size = off, size
        pos += 16

    if data_off is None or first_x is None:
        raise ValueError("not a recognizable OMNIC SPA file (no key 2/3 block)")

    y = np.frombuffer(raw, dtype="<f4", count=data_size // 4, offset=data_off).astype(float)
    x = np.linspace(first_x, last_x, len(y))
    unit = OMNIC_YCODE.get(ycode) if ycode is not None else None
    return x, y, unit


def read_jcampdx(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """JCAMP-DX reader (handles the common ``(X++(Y..Y))`` tabular form).
    Reads the explicit x grid from FIRST/LAST/DELTAX/NPOINTS and the y values
    from the data table, applying the X/Y scaling factors."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    fields = [f for f in text.split("##") if f.strip()]

    meta: Dict[str, object] = {}
    for field in fields:
        key, _, val = field.strip().partition("=")
        val = val.strip().replace("\n", " ")
        key = key.strip().upper()
        try:
            meta[key] = float(val)
        except ValueError:
            meta[key] = val

    npoints = int(meta.get("NPOINTS", 0))
    first_x = float(meta.get("FIRSTX", 0.0))
    last_x = float(meta.get("LASTX", 0.0))
    xfactor = float(meta.get("XFACTOR", 1.0))
    yfactor = float(meta.get("YFACTOR", 1.0))
    x = np.linspace(first_x, last_x, npoints) * xfactor

    m = re.search(r"##XYDATA=\s*\(X\+\+\(Y\.\.Y\)\)", text)
    if not m:
        raise ValueError("unsupported JCAMP-DX variant (no (X++(Y..Y)) table)")
    body = text[m.end():].split("##", 1)[0].strip()
    y_vals: List[float] = []
    for line in body.splitlines():
        toks = line.split()[1:]  # first token is the line's x anchor
        y_vals.extend(float(t) for t in toks)
    y = np.array(y_vals, dtype=float) * yfactor
    if len(y) != len(x):  # tolerate small grid mismatches
        x = np.linspace(first_x, last_x, len(y)) * xfactor

    yunits = str(meta.get("YUNITS", "")).strip().lower()
    unit = None
    if "abs" in yunits:
        unit = "A"
    elif "trans" in yunits:
        unit = "T"
    return x, y, unit


def read_xy(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    ext = path.suffix.lower()
    if ext == ".spa":
        return read_spa(path)
    if ext in (".jdx", ".dx", ".jcm"):
        return read_jcampdx(path)
    return read_csv(path)


# --------------------------------------------------------------------------- #
# Classification + loading
# --------------------------------------------------------------------------- #
def classify(path: Path) -> Optional[dict]:
    """Parse the wizard naming convention from a filename stem."""
    m = NAME_RE.match(path.stem)
    if not m:
        return None
    direction = m["direction"].lower() if m["direction"] else None
    kind = "bg" if m["bg"] else "sample"
    if kind == "sample" and direction is None:
        direction = "up"  # up-only runs omit the suffix
    return {
        "index": int(m["idx"]) if m["idx"] else None,
        "temperature": float(m["temp"].replace(",", ".")),
        "direction": direction,
        "kind": kind,
        "sample": m["sample"],
    }


def heuristic_unit(y: np.ndarray) -> str:
    """Scale-free Absorbance/Transmittance guess for unit-less CSVs.

    Absorbance sits on a low baseline with peaks pointing up; transmittance sits
    on a high baseline (~1 or ~100 %) with dips pointing down. We locate the
    baseline within the robust 1-99 percentile range: bottom -> A, top -> T."""
    y = y[np.isfinite(y)]
    if y.size == 0:
        return "A"
    lo, med, hi = np.percentile(y, [1, 50, 99])
    rng = hi - lo
    if rng <= 0:
        return "A"
    baseline_pos = (med - lo) / rng  # ~0 baseline low (A); ~1 baseline high (T)
    return "T" if baseline_pos > 0.55 else "A"


def resolve_unit(meta: dict, native_unit: Optional[str], y: np.ndarray,
                 override: Optional[str]) -> Tuple[str, str]:
    """Apply the unit-resolution hierarchy. Returns (unit_token, source)."""
    if override:
        return override, "override"
    if native_unit:
        return native_unit, "native"
    if meta["kind"] == "bg":
        return "SB", "background"
    return heuristic_unit(y), "heuristic"


def load_directory(folder: Path, select: str,
                   overrides: Optional[List[str]]) -> List[Spectrum]:
    """Read and classify every supported, recognizable file in ``folder``."""
    chosen = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        and classify(p) and (select == "all" or classify(p)["kind"] == select)
    )
    if overrides is not None and len(overrides) not in (1, len(chosen)):
        listing = "\n  ".join(p.name for p in chosen)
        raise SystemExit(
            f"--input-units got {len(overrides)} value(s) but there are "
            f"{len(chosen)} selected file(s). Pass one value (applied to all) or "
            f"one per file, in this order:\n  {listing}"
        )

    out: List[Spectrum] = []
    for i, p in enumerate(chosen):
        meta = classify(p)
        try:
            x, y, native_unit = read_xy(p)
        except Exception as e:  # noqa: BLE001 -- one bad file shouldn't abort
            print(f"[warn] skipping {p.name}: {e}", file=sys.stderr)
            continue
        if x.size and x[0] > x[-1]:           # normalize to ascending wavenumber
            x, y = x[::-1], y[::-1]
        override = None
        if overrides is not None:
            override = (overrides[0] if len(overrides) == 1 else overrides[i]).upper()
        unit, source = resolve_unit(meta, native_unit, y, override)
        out.append(Spectrum(path=p, x=x, y=y, unit=unit, unit_source=source, **meta))
    return out


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def convert_unit(y: np.ndarray, src: str, dst: str) -> Tuple[np.ndarray, str]:
    """Convert between absorbance and %transmittance. Other unit pairs are left
    untouched (returned as-is)."""
    if src == dst:
        return y, dst
    if src == "A" and dst == "T":
        return 100.0 * 10.0 ** (-y), "T"
    if src == "T" and dst == "A":
        frac = y / 100.0 if np.nanmax(y) > 1.5 else y  # %T vs fractional T
        return -np.log10(np.clip(frac, 1e-6, None)), "A"
    return y, src


def normalize(spectra: Sequence[Spectrum], mode: str) -> None:
    if mode == "none":
        return
    if mode == "global":
        lo = min(float(s.y.min()) for s in spectra)
        hi = max(float(s.y.max()) for s in spectra)
        rng = (hi - lo) or 1.0
        for s in spectra:
            s.y = (s.y - lo) / rng
    elif mode == "individual":
        for s in spectra:
            lo, hi = float(s.y.min()), float(s.y.max())
            s.y = (s.y - lo) / ((hi - lo) or 1.0)


def auto_offset(spectra: Sequence[Spectrum]) -> float:
    """A fixed offset just under one typical spectrum's amplitude, so adjacent
    traces separate with a little overlap."""
    amps = [float(np.percentile(s.y, 99) - np.percentile(s.y, 1)) for s in spectra]
    return 0.9 * (float(np.median(amps)) if amps else 1.0)


# --------------------------------------------------------------------------- #
# Styling + drawing
# --------------------------------------------------------------------------- #
def make_cmap_norm(temps: Sequence[float], cmap_name: str,
                   truncate: Tuple[float, float] = (0.1, 0.85)):
    base = colormaps[cmap_name]
    cmap = LinearSegmentedColormap.from_list(
        f"{cmap_name}_t", base(np.linspace(truncate[0], truncate[1], 256))
    )
    tmin, tmax = (min(temps), max(temps)) if temps else (0.0, 1.0)
    if tmin == tmax:
        tmax = tmin + 1.0
    return cmap, Normalize(tmin, tmax)


def style_axis(ax, unit: str, xlim: Tuple[float, float], hide_yticks: bool,
               tick_step: Optional[float]) -> None:
    hi, lo = max(xlim), min(xlim)
    ax.set_xlim(hi, lo)  # IR convention: high wavenumber on the left
    ax.set_xlabel(r"$\tilde\nu\ /\ \mathrm{cm^{-1}}$", size=11)
    ax.set_ylabel(UNIT_LABEL.get(unit, UNIT_LABEL["INT"]), size=11, labelpad=7)
    if tick_step:
        ax.xaxis.set_major_locator(MultipleLocator(tick_step))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis="both", which="both", direction="in", labelsize=8)
    axt = ax.twiny()  # mirrored ticks along the top
    axt.set_xlim(ax.get_xlim())
    axt.xaxis.set_minor_locator(AutoMinorLocator())
    axt.tick_params(axis="x", which="both", direction="in", labeltop=False)
    if hide_yticks:
        ax.set_yticks(())


def add_mol_image(ax, smiles: str, loc=(0.04, 0.96), zoom: float = 0.32) -> None:
    """Optionally draw a 2-D structure (from a SMILES string) on the axes.
    Requires RDKit; raises a clear error if it is not installed."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError as e:
        raise SystemExit("--smiles needs RDKit:  pip install rdkit") from e
    from io import BytesIO
    from PIL import Image
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise SystemExit(f"could not parse SMILES: {smiles!r}")
    Chem.rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(420, 420)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 4
    opts.clearBackground = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img = np.asarray(Image.open(BytesIO(drawer.GetDrawingText())).convert("RGBA"))
    box = AnnotationBbox(OffsetImage(img, zoom=zoom), loc, xycoords="axes fraction",
                         box_alignment=(0, 1), frameon=False)
    ax.add_artist(box)


def draw_panel(ax, spectra: List[Spectrum], cmap, norm, offset: float,
               label_curves: bool, unit: str, xlim: Tuple[float, float],
               tick_step: Optional[float], hide_yticks: bool) -> None:
    spectra = sorted(spectra, key=lambda s: s.order_key)
    hi = max(xlim)
    for i, s in enumerate(spectra):
        off = i * offset
        color = cmap(norm(s.temperature))
        ax.plot(s.x, s.y + off, color=color, lw=0.8)
        if label_curves and offset:
            tail = s.x >= hi - 0.06 * (max(xlim) - min(xlim))  # flat high-nu end
            base = float(np.mean(s.y[tail])) if tail.any() else float(s.y[-1])
            ax.text(hi, off + base, f" {s.temperature:g} °C", color=color,
                    fontsize=7, ha="left", va="bottom")
    style_axis(ax, unit, xlim, hide_yticks=hide_yticks, tick_step=tick_step)
    ax.margins(y=0.04)


def build_figure(spectra: List[Spectrum], args) -> "plt.Figure":
    unit = args.unit
    for s in spectra:  # convert to the requested display unit where possible
        s.y, s.unit = convert_unit(s.y, s.unit, unit)
    normalize(spectra, args.norm)

    xlim = tuple(args.xlim) if args.xlim else (
        max(float(s.x.max()) for s in spectra),
        min(float(s.x.min()) for s in spectra),
    )
    cmap, norm = make_cmap_norm([s.temperature for s in spectra], args.cmap)
    hide_y = args.norm != "none" or args.mode != "overlay"

    if args.mode == "updown":
        groups = {
            "up": [s for s in spectra if s.direction == "up"],
            "down": [s for s in spectra if s.direction in ("down", "return")],
        }
        groups = {k: v for k, v in groups.items() if v}
        fig, axes = plt.subplots(len(groups), 1, figsize=args.figsize,
                                 sharex=True, squeeze=False, layout="constrained")
        axes = axes[:, 0]
        offset = args.offset if args.offset is not None else auto_offset(spectra)
        for ax, (key, grp) in zip(axes, groups.items()):
            draw_panel(ax, grp, cmap, norm, offset, label_curves=True, unit=unit,
                       xlim=xlim, tick_step=args.tick_step, hide_yticks=True)
            ax.set_title(DIRECTION_LABEL.get(key, key), fontsize=10, loc="left")
    else:
        sel = spectra
        if args.direction != "both":
            sel = [s for s in spectra if (s.direction == args.direction
                   or (args.direction == "down" and s.direction == "return"))]
        fig, ax = plt.subplots(figsize=args.figsize, layout="constrained")
        axes = [ax]
        if args.mode == "overlay":
            offset = 0.0
        else:  # stack
            offset = args.offset if args.offset is not None else auto_offset(sel)
        draw_panel(ax, sel, cmap, norm, offset, label_curves=(args.mode == "stack"),
                   unit=unit, xlim=xlim, tick_step=args.tick_step, hide_yticks=hide_y)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=list(axes), pad=0.015, fraction=0.045)
    cbar.set_label(r"Temperature  /  $^\circ$C", size=10)

    if args.smiles:
        add_mol_image(axes[0], args.smiles)
    if args.title:
        fig.suptitle(args.title, fontweight="bold")
    return fig


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(spectra: List[Spectrum]) -> None:
    print(f"{'file':40s} {'kind':7s} {'dir':7s} {'T/C':>7s}  unit (source)")
    print("-" * 78)
    for s in sorted(spectra, key=lambda s: s.order_key):
        print(f"{s.path.name:40s} {s.kind:7s} {str(s.direction or '-'):7s} "
              f"{s.temperature:7g}  {s.unit} ({s.unit_source})")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                     argparse.RawDescriptionHelpFormatter):
    """Show argument defaults *and* keep the epilog's hand-formatting."""


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot a folder of variable-temperature IR spectra.",
        formatter_class=_HelpFormatter,
        epilog=(
            "examples:\n"
            "  plot_vt_ir.py data/                  waterfall of ./data\n"
            "  plot_vt_ir.py data/ --mode updown    heating vs cooling panels\n"
            "  plot_vt_ir.py data/ --mode overlay --unit T\n"
            "  plot_vt_ir.py data/ --input-units A  force every input to absorbance\n"
            "  plot_vt_ir.py data/ --list           report classification + units\n"
        ),
    )
    p.add_argument("directory", nargs="?", default=".",
                   help="Folder of VT-IR spectra (.csv / .spa / .jdx).")
    p.add_argument("--mode", choices=("overlay", "stack", "updown"), default="stack",
                   help="overlay (no offset), stack (fixed offset), or "
                        "updown (heating/cooling panels).")
    p.add_argument("--unit", choices=("A", "T"), default="A",
                   help="Display unit: A=absorbance, T=transmittance.")
    p.add_argument("--input-units", default=None,
                   help="Override detected input unit(s): a single token "
                        "(A/T/SB) applied to all files, or a comma/space list "
                        "with one token per file (in --list order).")
    p.add_argument("--select", choices=("sample", "bg", "all"), default="sample",
                   help="Which spectra to plot.")
    p.add_argument("--direction", choices=("both", "up", "down"), default="both",
                   help="Scan direction filter (overlay/stack modes only).")
    p.add_argument("--offset", type=float, default=None,
                   help="Fixed vertical offset for stack/updown (default: auto).")
    p.add_argument("--norm", choices=("none", "individual", "global"), default="none",
                   help="Normalize intensities before plotting.")
    p.add_argument("--cmap", default="gnuplot2", help="Matplotlib colormap (by T).")
    p.add_argument("--xlim", type=float, nargs=2, default=None,
                   metavar=("HIGH", "LOW"), help="Wavenumber limits, e.g. 4000 400.")
    p.add_argument("--tick-step", type=float, default=500.0,
                   help="Major x-tick spacing in cm^-1 (0 = auto).")
    p.add_argument("--smiles", default=None, help="Draw this structure (needs RDKit).")
    p.add_argument("--title", default=None, help="Figure title.")
    p.add_argument("--figsize", type=float, nargs=2, default=None,
                   metavar=("W", "H"), help="Figure size in inches.")
    p.add_argument("--save", default=None, help="Output file path.")
    p.add_argument("--ext", default="svg", help="Extension when --save is omitted.")
    p.add_argument("--dpi", type=int, default=300, help="Raster DPI.")
    p.add_argument("--silent", action="store_true",
                   help="Save without opening an interactive window.")
    p.add_argument("--transparent", action="store_true",
                   help="Transparent figure background.")
    p.add_argument("--list", action="store_true",
                   help="Print the classification/unit report and exit.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    folder = Path(args.directory)
    if not folder.is_dir():
        raise SystemExit(f"not a directory: {folder}")

    overrides = None
    if args.input_units:
        overrides = [t for t in re.split(r"[,\s]+", args.input_units.strip()) if t]

    spectra = load_directory(folder, args.select, overrides)
    if not spectra:
        raise SystemExit(f"no recognizable VT-IR files in {folder}")

    if args.list:
        print_report(spectra)
        return 0
    print_report(spectra)

    if args.tick_step == 0:
        args.tick_step = None
    if args.figsize is None:
        args.figsize = (10, 6) if args.mode == "updown" else (10, 5)
    if args.silent:
        plt.switch_backend("Agg")

    fig = build_figure(spectra, args)

    if args.save or args.silent:
        out = args.save or f"{folder.resolve().name}.{args.ext}"
        fig.savefig(out, dpi=args.dpi, transparent=args.transparent)
        print(f"saved {out}")
    if not args.silent:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
