"""
Render manuscript-style figure grids using Pandoc + LaTeX.

Primary use case:
- Make a quick "how will this look in a manuscript figure" preview that respects
  a fixed grid layout and approximate panel sizing.

This module is intentionally lightweight and configuration-driven:
- It discovers subpanels from an output directory (e.g. results/NewFigure1)
- It places them left-to-right, top-to-bottom in an R x C grid
- Missing panels render as labeled placeholders (e.g. 1c, 1d, ...)
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


_PANEL_RE = re.compile(r"(?P<fig>\d+)(?P<panel>[a-z])", re.IGNORECASE)


@dataclass(frozen=True)
class GridSpec:
    rows: int
    cols: int
    figure_number: int = 1

    def panel_labels(self) -> list[str]:
        # 1a, 1b, 1c, ...
        start = ord("a")
        total = self.rows * self.cols
        return [f"{self.figure_number}{chr(start + i)}" for i in range(total)]


def _find_panels(output_dir: Path, filename_prefix: str) -> dict[str, Path]:
    """
    Discover panel image files under output_dir.

    Returns
    -------
    dict mapping panel label (e.g. "1a") -> Path
    """
    panels: dict[str, Path] = {}
    if not output_dir.exists():
        return panels

    # Typical names in this repo: manuscript_1a.png, manuscript_1b.png, ...
    candidates = sorted(output_dir.glob(f"{filename_prefix}*.*"))
    for p in candidates:
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".pdf"}:
            continue
        m = _PANEL_RE.search(p.stem)
        if not m:
            continue
        label = f"{int(m.group('fig'))}{m.group('panel').lower()}"
        panels[label] = p
    return panels


def _latex_escape_text(s: str) -> str:
    # minimal (we only use short labels like "1a")
    return s.replace("_", r"\_")


def _build_pandoc_markdown(
    *,
    grid: GridSpec,
    panels: dict[str, Path],
    title: str,
    cell_width_fraction: float,
    gap_fraction: float,
    image_offset_cells: int,
) -> str:
    """
    Create a Pandoc markdown document containing raw LaTeX for a fixed grid.
    """
    if grid.rows < 1 or grid.cols < 1:
        raise ValueError("grid.rows and grid.cols must be >= 1")

    labels = grid.panel_labels()
    cell_w = max(0.01, min(1.0, cell_width_fraction))
    gap = max(0.0, min(0.25, gap_fraction))

    # IMPORTANT:
    # - Only put package-level LaTeX in header-includes.
    # - Put *all* layout / environments (\begin{center}...) in the document body.
    #
    # Putting layout LaTeX into header-includes can produce broken TeX like:
    #   ! LaTeX Error: Missing \begin{document}.
    header_includes = [
        r"\usepackage{graphicx}",
        r"\usepackage[margin=0.6in]{geometry}",
        r"\usepackage{xcolor}",
        # Use Helvetica (Arial-like) under pdflatex.
        r"\usepackage{helvet}",
        # Fixed panel height to keep alignment consistent across a row.
        r"\newlength{\panelh}",
        # Slot height (set to 0.75x the previous 0.19\textwidth)
        r"\setlength{\panelh}{0.1425\textwidth}",
        # Reserve a small top strip for the panel letter so letters align
        # consistently regardless of the image aspect ratio.
        r"\newlength{\labelh}",
        r"\setlength{\labelh}{0.12\panelh}",
        r"\setlength{\parindent}{0pt}",
    ]

    body_lines: list[str] = []
    body_lines.append(r"\begin{center}")
    body_lines.append(r"\setlength{\tabcolsep}{0pt}")
    body_lines.append(r"\renewcommand{\arraystretch}{1}")

    idx = 0
    for _r in range(grid.rows):
        row_cells: list[str] = []
        for _c in range(grid.cols):
            cell_label = labels[idx]
            idx += 1
            panel_letter = cell_label[-1]

            # Shift *images* across the grid while keeping the cell letter
            # labels fixed to the cell position.
            image_idx = idx - image_offset_cells
            image_label = labels[image_idx] if image_idx >= 0 else None
            panel_path = panels.get(image_label) if image_label is not None else None

            if panel_path is None:
                cell = (
                    r"\fcolorbox{black}{white}{"
                    rf"\begin{{minipage}}[t][\panelh][t]{{{cell_w:.4f}\textwidth}}"
                    r"\raggedright"
                    rf"{{\sffamily\fontfamily{{phv}}\selectfont\bfseries {_latex_escape_text(panel_letter)}}}\par"
                    r"\vfill"
                    r"\centering{\small (missing)}"
                    r"\vfill"
                    r"\end{minipage}"
                    r"}"
                )
            else:
                # Use forward slashes for LaTeX portability on Windows.
                path_for_latex = panel_path.as_posix()
                # Targeted placement tweak: shift only the plot for image "1c"
                # left by 5% of the subpanel width. Labels remain unchanged
                # because they're emitted before this point.
                #
                # Using \makebox[\linewidth][l]{...} keeps the vertical layout stable
                # and avoids interactions with centering in the parent minipage.
                if image_label == f"{grid.figure_number}c":
                    img_block = (
                        # Use a full-width makebox so centering in the parent
                        # minipage doesn't interfere with horizontal shifting.
                        r"\makebox[\linewidth][l]{"
                        r"\hspace{-0.05\linewidth}"
                        r"\scalebox{1.15}{"
                        rf"\includegraphics[height=\dimexpr\panelh-\labelh\relax,keepaspectratio]{{{path_for_latex}}}"
                        r"}"
                        r"}"
                    )
                else:
                    img_block = (
                        r"\scalebox{1.15}{"
                        rf"\includegraphics[height=\dimexpr\panelh-\labelh\relax,keepaspectratio]{{{path_for_latex}}}"
                        r"}"
                    )
                cell = (
                    rf"\begin{{minipage}}[t][\panelh][t]{{{cell_w:.4f}\textwidth}}"
                    r"\centering"
                    # Panel letter anchored to the *cell* (not the image), so it
                    # stays at a consistent height across the row.
                    r"\raggedright"
                    rf"{{\sffamily\fontfamily{{phv}}\selectfont\bfseries {_latex_escape_text(panel_letter)}}}\par"
                    r"\vspace{0pt}\vfill"
                    + img_block
                    + r"\vfill"
                    + r"\end{minipage}"
                )

            row_cells.append(cell)
        body_lines.append("".join(row_cells))
        if gap > 0:
            body_lines.append(rf"\vspace{{{gap:.4f}\textwidth}}")
        body_lines.append(r"\par")
    body_lines.append(r"\end{center}")

    md = []
    md.append("---")
    md.append(f"title: {title}")
    md.append("header-includes:")
    for line in header_includes:
        md.append(f"  - '{line}'")
    md.append("---")
    md.append("")
    md.append(r"```{=latex}")
    md.extend(body_lines)
    md.append(r"```")
    md.append("")

    return "\n".join(md)


def _run_pandoc_to_pdf(md_path: Path, pdf_path: Path, pdf_engine: str) -> None:
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        raise RuntimeError(
            "pandoc was not found on PATH. Install Pandoc or add it to PATH."
        )

    # Pandoc accepts either an executable name (resolved via PATH) or an explicit path.
    resolved_engine = None
    engine_path = Path(pdf_engine)
    if engine_path.exists():
        resolved_engine = str(engine_path)
    else:
        resolved_engine = shutil.which(pdf_engine)

    print(f"Pandoc: {pandoc}")
    if resolved_engine is None:
        print(
            f'PDF engine: {pdf_engine} (not found on PATH in this process; letting Pandoc try)'
        )
    else:
        print(f"PDF engine: {pdf_engine} ({resolved_engine})")
    print(f"Input MD: {md_path}")
    print(f"Output PDF: {pdf_path}")

    cmd = [
        pandoc,
        str(md_path),
        "-o",
        str(pdf_path),
        "--pdf-engine",
        pdf_engine,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Pandoc PDF render failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )


def _try_pdf_to_png(pdf_path: Path, png_path: Path, dpi: int) -> bool:
    # Prefer poppler's pdftocairo if present; otherwise just skip.
    pdftocairo = shutil.which("pdftocairo")
    if pdftocairo is None:
        return False

    cmd = [
        pdftocairo,
        "-png",
        "-singlefile",
        "-r",
        str(int(dpi)),
        str(pdf_path),
        str(png_path.with_suffix("")),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0 and png_path.exists()


def render_manuscript_grid(
    *,
    output_dir: Path,
    filename_prefix: str,
    grid: GridSpec,
    out_basename: str,
    title: str,
    pdf_engine: str,
    dpi: int,
    panel_image_offset_cells: int = 0,
) -> dict[str, Path]:
    """
    Render a manuscript-style grid PDF (and optionally PNG) into output_dir.

    Returns
    -------
    dict with keys: "md", "pdf", optionally "png"
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    panels = _find_panels(output_dir, filename_prefix=filename_prefix)

    # For 5 columns, a good default is ~0.19 textwidth per cell.
    md_text = _build_pandoc_markdown(
        grid=grid,
        panels=panels,
        title=title,
        cell_width_fraction=0.19,
        gap_fraction=0.01,
        image_offset_cells=panel_image_offset_cells,
    )

    md_path = output_dir / f"{out_basename}.md"
    pdf_path = output_dir / f"{out_basename}.pdf"
    png_path = output_dir / f"{out_basename}.png"

    md_path.write_text(md_text, encoding="utf-8")
    _run_pandoc_to_pdf(md_path, pdf_path, pdf_engine=pdf_engine)

    results: dict[str, Path] = {"md": md_path, "pdf": pdf_path}
    if _try_pdf_to_png(pdf_path, png_path, dpi=dpi):
        results["png"] = png_path
    return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a manuscript-style grid preview using Pandoc + LaTeX."
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help='Directory containing manuscript_* panels (default: figure_1_output_dir from src.figure_1_config).',
    )
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--figure-number", type=int, default=1)
    p.add_argument("--filename-prefix", type=str, default="manuscript_")
    p.add_argument("--out-basename", type=str, default="manuscript_grid_preview")
    p.add_argument("--title", type=str, default="Figure layout preview")
    # Prefer tectonic if installed (single binary, auto-fetches LaTeX packages).
    default_engine = "tectonic" if shutil.which("tectonic") else "pdflatex"
    p.add_argument("--pdf-engine", type=str, default=default_engine)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument(
        "--panel-image-offset-cells",
        type=int,
        default=0,
        help="Shift discovered panel images right by N grid slots while keeping cell letter labels fixed.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output_dir is None:
        from src import figure_1_config as fig_cfg

        output_dir = Path(fig_cfg.figure_1_output_dir)
    else:
        output_dir = Path(args.output_dir)

    results = render_manuscript_grid(
        output_dir=output_dir,
        filename_prefix=args.filename_prefix,
        grid=GridSpec(rows=args.rows, cols=args.cols, figure_number=args.figure_number),
        out_basename=args.out_basename,
        title=args.title,
        pdf_engine=args.pdf_engine,
        dpi=args.dpi,
        panel_image_offset_cells=args.panel_image_offset_cells,
    )

    print(f"Wrote: {results['md']}")
    print(f"Wrote: {results['pdf']}")
    if "png" in results:
        print(f"Wrote: {results['png']}")
    else:
        print("PNG not generated (pdftocairo not found on PATH). PDF is available.")


if __name__ == "__main__":
    main()

