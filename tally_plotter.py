from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Literal

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import openmc

Agg = Literal["mean", "sum", "max", "std"]
LineStyleMode = Literal["lines", "points", "linespoints"]


# -------------------------
# Mesh definitions
# -------------------------

@dataclass
class AxisDef:
    name: str
    edges: np.ndarray
    centers: np.ndarray
    size: int


@dataclass
class MeshDef:
    mesh: openmc.MeshBase
    axes: List[AxisDef]
    order: Literal["F", "C"] = "F"


def _centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def _filters_debug(t: openmc.Tally) -> str:
    parts = []
    for f in t.filters:
        nm = type(f).__name__
        if isinstance(f, openmc.MeshFilter):
            parts.append(f"{nm}(mesh={type(f.mesh).__name__})")
        else:
            parts.append(nm)
    return ", ".join(parts) if parts else "(no filters)"


def _meshdef_from_mesh(mesh: openmc.MeshBase) -> MeshDef:
    if isinstance(mesh, openmc.RegularMesh):
        nx, ny, nz = mesh.dimension
        ll = mesh.lower_left
        ur = mesh.upper_right
        x_edges = np.linspace(ll[0], ur[0], int(nx) + 1)
        y_edges = np.linspace(ll[1], ur[1], int(ny) + 1)
        z_edges = np.linspace(ll[2], ur[2], int(nz) + 1)
        axes = [
            AxisDef("x", x_edges, _centers(x_edges), int(nx)),
            AxisDef("y", y_edges, _centers(y_edges), int(ny)),
            AxisDef("z", z_edges, _centers(z_edges), int(nz)),
        ]
        return MeshDef(mesh=mesh, axes=axes, order="F")

    if isinstance(mesh, openmc.RectilinearMesh):
        x_edges = np.asarray(mesh.x_grid, dtype=float)
        y_edges = np.asarray(mesh.y_grid, dtype=float)
        z_edges = np.asarray(mesh.z_grid, dtype=float)
        nx, ny, nz = mesh.dimension
        axes = [
            AxisDef("x", x_edges, _centers(x_edges), int(nx)),
            AxisDef("y", y_edges, _centers(y_edges), int(ny)),
            AxisDef("z", z_edges, _centers(z_edges), int(nz)),
        ]
        return MeshDef(mesh=mesh, axes=axes, order="F")

    if hasattr(openmc, "CylindricalMesh") and isinstance(mesh, openmc.CylindricalMesh):
        r_edges = np.asarray(mesh.r_grid, dtype=float)
        phi_edges = np.asarray(mesh.phi_grid, dtype=float)
        z_edges = np.asarray(mesh.z_grid, dtype=float)
        nr, nphi, nz = mesh.dimension
        axes = [
            AxisDef("r", r_edges, _centers(r_edges), int(nr)),
            AxisDef("phi", phi_edges, _centers(phi_edges), int(nphi)),
            AxisDef("z", z_edges, _centers(z_edges), int(nz)),
        ]
        return MeshDef(mesh=mesh, axes=axes, order="F")

    if hasattr(openmc, "SphericalMesh") and isinstance(mesh, openmc.SphericalMesh):
        r_edges = np.asarray(mesh.r_grid, dtype=float)
        theta_edges = np.asarray(mesh.theta_grid, dtype=float)
        phi_edges = np.asarray(mesh.phi_grid, dtype=float)
        nr, ntheta, nphi = mesh.dimension
        axes = [
            AxisDef("r", r_edges, _centers(r_edges), int(nr)),
            AxisDef("theta", theta_edges, _centers(theta_edges), int(ntheta)),
            AxisDef("phi", phi_edges, _centers(phi_edges), int(nphi)),
        ]
        return MeshDef(mesh=mesh, axes=axes, order="F")

    raise ValueError(
        f"Unsupported mesh type: {type(mesh).__name__}. "
        "Supported: RegularMesh, RectilinearMesh, CylindricalMesh, SphericalMesh."
    )


def _energy_bin_centers_from_filter(filt: openmc.EnergyFilter) -> np.ndarray:
    bins = np.asarray(filt.bins, dtype=float)

    if bins.ndim == 2 and bins.shape[1] == 2:
        e_lo = bins[:, 0]
        e_hi = bins[:, 1]
        return np.sqrt(e_lo * e_hi)

    bins = np.ravel(bins)
    if bins.size < 2:
        raise ValueError("EnergyFilter does not contain enough bin edges.")
    return np.sqrt(bins[:-1] * bins[1:])


def _energy_bin_edges_from_filter(filt: openmc.EnergyFilter) -> np.ndarray:
    bins = np.asarray(filt.bins, dtype=float)

    if bins.ndim == 2 and bins.shape[1] == 2:
        edges = np.concatenate([[bins[0, 0]], bins[:, 1]])
        return edges

    bins = np.ravel(bins)
    if bins.size < 2:
        raise ValueError("EnergyFilter does not contain enough bin edges.")
    return bins


def _plot_fmt(style: LineStyleMode) -> str:
    if style == "lines":
        return "-"
    if style == "points":
        return "o"
    if style == "linespoints":
        return "-o"
    raise ValueError(style)


# -------------------------
# Reduction helpers
# -------------------------

def _reduce(arr: np.ndarray, axes: Tuple[int, ...], agg: Literal["mean", "sum", "max"]) -> np.ndarray:
    if agg == "mean":
        return np.mean(arr, axis=axes)
    if agg == "sum":
        return np.sum(arr, axis=axes)
    if agg == "max":
        return np.max(arr, axis=axes)
    raise ValueError(agg)


def _reduce_unc(sd: np.ndarray, mean: np.ndarray, axes: Tuple[int, ...], agg: Literal["mean", "sum", "max"]) -> np.ndarray:
    if agg == "sum":
        return np.sqrt(np.sum(sd**2, axis=axes))
    if agg == "mean":
        N = 1
        for ax in axes:
            N *= sd.shape[ax]
        return np.sqrt(np.sum(sd**2, axis=axes)) / max(N, 1)
    if agg == "max":
        keep_axes = tuple(i for i in range(mean.ndim) if i not in axes)
        perm = keep_axes + axes
        mean_p = np.transpose(mean, perm)
        sd_p = np.transpose(sd, perm)

        keep_shape = mean_p.shape[: len(keep_axes)]
        red_shape = mean_p.shape[len(keep_axes):]
        red_n = int(np.prod(red_shape)) if red_shape else 1

        mean_flat = mean_p.reshape((*keep_shape, red_n))
        sd_flat = sd_p.reshape((*keep_shape, red_n))

        idx = np.argmax(mean_flat, axis=-1)
        out = np.take_along_axis(sd_flat, idx[..., None], axis=-1)[..., 0]
        return out
    raise ValueError(agg)


def _reduce_sd(sd: np.ndarray, axes: Tuple[int, ...], mode: Literal["mean", "sum", "max"] = "mean") -> np.ndarray:
    if mode == "sum":
        return np.sqrt(np.sum(sd**2, axis=axes))
    if mode == "mean":
        N = 1
        for ax in axes:
            N *= sd.shape[ax]
        return np.sqrt(np.sum(sd**2, axis=axes)) / max(N, 1)
    if mode == "max":
        keep_axes = tuple(i for i in range(sd.ndim) if i not in axes)
        perm = keep_axes + axes
        sd_p = np.transpose(sd, perm)

        keep_shape = sd_p.shape[: len(keep_axes)]
        red_shape = sd_p.shape[len(keep_axes):]
        red_n = int(np.prod(red_shape)) if red_shape else 1

        sd_flat = sd_p.reshape((*keep_shape, red_n))
        idx = np.argmax(sd_flat, axis=-1)
        out = np.take_along_axis(sd_flat, idx[..., None], axis=-1)[..., 0]
        return out
    raise ValueError(mode)


def _mask_for_lognorm(arr: np.ndarray, eps: float = 0.0) -> np.ndarray:
    out = np.array(arr, copy=True)
    out[out <= eps] = np.nan
    return out


def _resample_2d_nearest(data2d: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx <= 0 or dy <= 0:
        raise ValueError("Dx and Dy must be positive integers.")
    nx, ny = data2d.shape
    if (nx == dx) and (ny == dy):
        return data2d
    xi = np.clip((np.linspace(0, nx - 1, dx)).round().astype(int), 0, nx - 1)
    yi = np.clip((np.linspace(0, ny - 1, dy)).round().astype(int), 0, ny - 1)
    return data2d[np.ix_(xi, yi)]


def _figsize_for_pixels(dx: int, dy: int, dpi: int, min_in: float = 3.0) -> Tuple[float, float]:
    w = max(dx / max(dpi, 1), min_in)
    h = max(dy / max(dpi, 1), min_in)
    return (w, h)


def _sanitize_filename(s: str) -> str:
    if not s:
        return "tally"
    out = []
    for ch in s.strip():
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        elif ch.isspace():
            out.append("_")
        else:
            out.append("_")
    return "".join(out).strip("_") or "tally"


def _default_output_stem(tally: openmc.Tally, ident: Union[int, str]) -> str:
    name = (tally.name or "").strip()
    if name:
        return _sanitize_filename(name)
    return f"tally{int(tally.id) if hasattr(tally, 'id') else _sanitize_filename(str(ident))}"


def _default_png_name(stem: str, section: str) -> str:
    sec = _sanitize_filename(section.lower())
    return f"{stem}_{sec}.png"


def _default_csv_name(stem: str, section: str) -> str:
    sec = _sanitize_filename(section.lower())
    return f"{stem}_{sec}.csv"


def _save_csv_1d(path: str, x: np.ndarray, y: np.ndarray, yerr: Optional[np.ndarray], x_label: str, y_label: str) -> None:
    cols = [x, y]
    header = [x_label, y_label]
    if yerr is not None:
        cols.append(yerr)
        header.append("yerr")
    data = np.column_stack(cols)
    np.savetxt(path, data, delimiter=",", header=",".join(header), comments="")


def _save_csv_2d(path: str, x_centers: np.ndarray, y_centers: np.ndarray, z: np.ndarray,
                 x_label: str, y_label: str, z_label: str) -> None:
    X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")
    out = np.column_stack([X.ravel(order="C"), Y.ravel(order="C"), z.ravel(order="C")])
    np.savetxt(path, out, delimiter=",", header=f"{x_label},{y_label},{z_label}", comments="")


# -------------------------
# Main Plotter
# -------------------------

class OpenMCTallyPlotter:
    def __init__(self, statepoint_path: str):
        self.sp = openmc.StatePoint(statepoint_path)

    def list_tallies(self) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for t in self.sp.tallies.values():
            out.append(
                dict(
                    id=int(t.id),
                    name=t.name,
                    scores=list(t.scores),
                    nuclides=list(t.nuclides),
                    filters=[type(f).__name__ for f in t.filters],
                )
            )
        return out

    def get_tally(self, ident: Union[int, str]) -> openmc.Tally:
        if isinstance(ident, int):
            return self.sp.get_tally(id=ident)
        try:
            return self.sp.get_tally(name=ident)
        except Exception:
            return self.sp.get_tally(id=int(ident))

    def _has_mesh_filter(self, tally: openmc.Tally) -> bool:
        return any(isinstance(f, openmc.MeshFilter) for f in tally.filters)

    def _has_energy_filter(self, tally: openmc.Tally) -> bool:
        return any(isinstance(f, openmc.EnergyFilter) for f in tally.filters)

    def _get_mesh_filter(self, tally: openmc.Tally) -> openmc.MeshFilter:
        for f in tally.filters:
            if isinstance(f, openmc.MeshFilter):
                return f
        raise ValueError(
            "This tally has no MeshFilter — cannot make spatial profiles/heatmaps.\n"
            f"Filters: {_filters_debug(tally)}"
        )

    def _get_energy_filter(self, tally: openmc.Tally) -> openmc.EnergyFilter:
        for f in tally.filters:
            if isinstance(f, openmc.EnergyFilter):
                return f
        raise ValueError(
            "This tally has no EnergyFilter — cannot make an energy spectrum.\n"
            f"Filters: {_filters_debug(tally)}"
        )

    def _meshdef(self, tally: openmc.Tally) -> MeshDef:
        mf = self._get_mesh_filter(tally)
        return _meshdef_from_mesh(mf.mesh)

    def _default_score_nuclide(
        self, tally: openmc.Tally, score: Optional[str], nuclide: Optional[str]
    ) -> Tuple[str, str]:
        if score is None:
            if len(tally.scores) != 1:
                raise ValueError(f"Tally has multiple scores: {tally.scores}. Provide --score ...")
            score = tally.scores[0]

        if nuclide is None:
            if len(tally.nuclides) == 0:
                nuclide = "total"
            elif "total" in tally.nuclides:
                nuclide = "total"
            elif len(tally.nuclides) == 1:
                nuclide = tally.nuclides[0]
            else:
                nuclide = tally.nuclides[0]
        return str(score), str(nuclide)

    def _reshaped_compatible(
        self,
        tally: openmc.Tally,
        value: Literal["mean", "std_dev"],
        score: str,
        nuclide: str,
    ) -> np.ndarray:
        if hasattr(tally, "get_slice"):
            try:
                ts = tally.get_slice(scores=[score], nuclides=[nuclide])
                arr = getattr(ts, value)
                return np.squeeze(arr)
            except Exception:
                pass

        try:
            arr = tally.get_reshaped_data(value, [score], [nuclide])
            return np.squeeze(arr)
        except Exception:
            pass

        try:
            arr_all = tally.get_reshaped_data(value=value)
        except TypeError:
            arr_all = tally.get_reshaped_data(value)

        arr_all = np.squeeze(arr_all)
        n_scores = len(tally.scores)
        n_nucs = len(tally.nuclides) if len(tally.nuclides) > 0 else 1

        try:
            i_score = list(tally.scores).index(score)
        except Exception as e:
            raise ValueError(f"Score '{score}' does not exist. Available: {tally.scores}") from e

        if len(tally.nuclides) == 0:
            i_nuc = 0
        else:
            try:
                i_nuc = list(tally.nuclides).index(nuclide)
            except Exception as e:
                raise ValueError(f"Nuclide '{nuclide}' does not exist. Available: {tally.nuclides}") from e

        if arr_all.ndim == 0:
            return arr_all

        shape = arr_all.shape

        if len(shape) >= 2 and shape[-2] == n_nucs and shape[-1] == n_scores:
            return np.squeeze(arr_all[..., i_nuc, i_score])

        if len(shape) >= 2 and shape[-2] == n_scores and shape[-1] == n_nucs:
            return np.squeeze(arr_all[..., i_score, i_nuc])

        if len(shape) >= 1 and shape[-1] == n_scores:
            return np.squeeze(arr_all[..., i_score])

        if len(shape) >= 1 and shape[-1] == n_nucs:
            return np.squeeze(arr_all[..., i_nuc])

        raise ValueError(
            "Cannot map get_reshaped_data() array dimensions to scores/nuclides "
            f"(shape={shape}, n_scores={n_scores}, n_nuclides={n_nucs})."
        )

    def _select_other_filters(
        self,
        arr: np.ndarray,
        tally: openmc.Tally,
        select: Optional[Dict[str, int]],
    ) -> np.ndarray:
        if arr.ndim == 0:
            return arr
        if arr.ndim == 1:
            return arr

        has_mesh = any(isinstance(f, openmc.MeshFilter) for f in tally.filters)
        if not has_mesh:
            return arr

        other_filters = [type(f).__name__ for f in tally.filters if not isinstance(f, openmc.MeshFilter)]
        if not other_filters:
            return np.ravel(arr)

        if select is None:
            raise ValueError(
                "This tally has additional filters besides MeshFilter, so you must select a specific bin.\n"
                f"Other filters: {other_filters}\n"
                "Use e.g. select={'EnergyFilter': 0} (or the appropriate index)."
            )

        idx = []
        for f in tally.filters:
            if isinstance(f, openmc.MeshFilter):
                idx.append(slice(None))
            else:
                key = type(f).__name__
                if key not in select:
                    raise ValueError(f"No selection provided for filter {key}. Provide select={{'{key}': i}}")
                idx.append(int(select[key]))

        try:
            out = arr[tuple(idx)]
            out = np.squeeze(out)
            return out
        except Exception:
            raise ValueError(
                "Failed to apply select to the additional filters (dimension ordering differs in this OpenMC version).\n"
                "Quick fix: use tally.get_slice(...) manually and pass the already-sliced tally into the plotter."
            )

    def _mesh_vector_3d(
        self,
        tally: openmc.Tally,
        value: Literal["mean", "std_dev"],
        score: Optional[str],
        nuclide: Optional[str],
        select: Optional[Dict[str, int]],
    ) -> Tuple[np.ndarray, MeshDef, str, str]:
        md = self._meshdef(tally)
        score_s, nuclide_s = self._default_score_nuclide(tally, score, nuclide)

        arr = self._reshaped_compatible(tally, value=value, score=score_s, nuclide=nuclide_s)
        arr = self._select_other_filters(arr, tally, select)

        if np.ndim(arr) == 0:
            return np.asarray(arr), md, score_s, nuclide_s

        if arr.ndim != 1:
            arr = np.ravel(arr)

        n_expected = md.axes[0].size * md.axes[1].size * md.axes[2].size
        if arr.size != n_expected:
            raise ValueError(
                f"Data size ({arr.size}) does not match mesh size ({n_expected}).\n"
                f"Tally: id={int(tally.id)}, name='{tally.name}', filters={_filters_debug(tally)}\n"
                "If you have additional filters, make sure select chooses a single bin."
            )

        vol = arr.reshape((md.axes[0].size, md.axes[1].size, md.axes[2].size), order=md.order)
        return vol, md, score_s, nuclide_s

    def _spectrum_vector_1d(
        self,
        tally: openmc.Tally,
        value: Literal["mean", "std_dev"],
        score: Optional[str],
        nuclide: Optional[str],
        select: Optional[Dict[str, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
        ef = self._get_energy_filter(tally)
        score_s, nuclide_s = self._default_score_nuclide(tally, score, nuclide)

        arr = self._reshaped_compatible(tally, value=value, score=score_s, nuclide=nuclide_s)
        arr = np.asarray(arr)

        if arr.ndim == 0:
            raise ValueError("Energy spectrum tally is scalar, expected EnergyFilter bins.")

        idx = []
        used_energy = False

        for f in tally.filters:
            if isinstance(f, openmc.EnergyFilter):
                idx.append(slice(None))
                used_energy = True
            else:
                key = type(f).__name__
                if arr.ndim == 1:
                    continue
                if select is None:
                    idx.append(0)
                else:
                    idx.append(int(select.get(key, 0)))

        if used_energy and arr.ndim > 1:
            try:
                arr = arr[tuple(idx)]
            except Exception as e:
                raise ValueError(
                    "Failed to slice tally with EnergyFilter. "
                    "Provide select={'MaterialFilter': i, ...} if needed."
                ) from e

        arr = np.squeeze(arr)

        if arr.ndim != 1:
            arr = np.ravel(arr)

        e_centers = _energy_bin_centers_from_filter(ef)
        e_edges = _energy_bin_edges_from_filter(ef)

        if arr.size != e_centers.size:
            raise ValueError(
                f"Spectrum size ({arr.size}) does not match number of energy bins ({e_centers.size}).\n"
                f"Filters: {_filters_debug(tally)}"
            )

        return arr, e_centers, e_edges, score_s, nuclide_s

    def _overlay_geometry_contours(
        self,
        ax: plt.Axes,
        geometry_path: str,
        plane: str,
        extent: List[float],
        slice_axis: Optional[str],
        slice_coord: Optional[float],
    ) -> None:
        plane = plane.lower()
        if plane not in ("xy", "xz", "yz"):
            raise ValueError(
                f"Geometry overlay is supported only for Cartesian planes xy/xz/yz, got '{plane}'."
            )

        if plane == "xy":
            z0 = 0.0 if slice_axis is None else float(slice_coord or 0.0)
            origin = (0.0, 0.0, z0)
        elif plane == "xz":
            y0 = 0.0 if slice_axis is None else float(slice_coord or 0.0)
            origin = (0.0, y0, 0.0)
        else:
            x0 = 0.0 if slice_axis is None else float(slice_coord or 0.0)
            origin = (x0, 0.0, 0.0)

        width = (extent[1] - extent[0], extent[3] - extent[2])

        geom = openmc.Geometry.from_xml(geometry_path)

        geom.plot(
            basis=plane,
            origin=origin,
            width=width,
            pixels=(1600, 1600),
            axes=ax,
            outline="only",
        )

    def plot_1d(
        self,
        tally_id_or_name: Union[int, str],
        axis: str,
        score: Optional[str] = None,
        nuclide: Optional[str] = None,
        agg: Agg = "mean",
        select: Optional[Dict[str, int]] = None,
        fname: Optional[str] = None,
        title: Optional[str] = None,
        show: bool = True,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xscale: Optional[str] = None,
        yscale: Optional[str] = None,
        grid: bool = True,
        capsize: float = 3.0,
        dpi: int = 200,
        figsize: Optional[Tuple[float, float]] = None,
        csv_path: Optional[str] = None,
        plot_style: LineStyleMode = "lines",
        multiplyx: float = 1.0,
        multiplyz: float = 1.0,
    ) -> plt.Figure:
        t = self.get_tally(tally_id_or_name)

        mean3d, md, score_s, _ = self._mesh_vector_3d(t, "mean", score, nuclide, select)
        sd3d, _, _, _ = self._mesh_vector_3d(t, "std_dev", score, nuclide, select)

        axis_names = [a.name for a in md.axes]
        if axis not in axis_names:
            raise ValueError(
                f"Mesh of type {type(md.mesh).__name__} does not have axis '{axis}'. "
                f"Available axes: {axis_names}"
            )

        ax_i = axis_names.index(axis)
        x = md.axes[ax_i].centers * multiplyx
        reduce_axes = tuple(i for i in range(3) if i != ax_i)

        fig, axp = plt.subplots(figsize=figsize)
        fmt = _plot_fmt(plot_style)

        if agg == "std":
            y = _reduce_sd(sd3d, reduce_axes, mode="mean")
            y = y * multiplyz
            yerr = None
            axp.plot(x, y, fmt)
            y_label = ylabel or f"{score_s} (std_dev)"
            axp.set_ylabel(y_label)
        else:
            y = _reduce(mean3d, reduce_axes, agg)
            yerr = _reduce_unc(sd3d, mean3d, reduce_axes, agg)
            y = y * multiplyz
            yerr = yerr * multiplyz
            axp.errorbar(x, y, yerr=yerr, fmt=fmt, capsize=capsize)
            y_label = ylabel or f"{score_s} ({agg})"
            axp.set_ylabel(y_label)

        x_label = xlabel or axis
        axp.set_xlabel(x_label)

        if xscale:
            axp.set_xscale(xscale)
        if yscale:
            axp.set_yscale(yscale)

        if xlim:
            axp.set_xlim(*xlim)
        if ylim:
            axp.set_ylim(*ylim)

        if grid:
            axp.grid(True, alpha=0.3)

        axp.set_title(title or f"Tally {int(t.id)} '{t.name}' | {axis} | aggregation={agg}")

        fig.tight_layout()

        if csv_path:
            _save_csv_1d(csv_path, x=x, y=y, yerr=yerr, x_label=x_label, y_label=y_label)

        if fname:
            fig.savefig(fname, dpi=dpi)
        if show:
            plt.show()
        return fig

    def plot_spectrum(
        self,
        tally_id_or_name: Union[int, str],
        score: Optional[str] = None,
        nuclide: Optional[str] = None,
        select: Optional[Dict[str, int]] = None,
        fname: Optional[str] = None,
        title: Optional[str] = None,
        show: bool = True,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xscale: Optional[str] = "log",
        yscale: Optional[str] = "log",
        grid: bool = True,
        capsize: float = 3.0,
        dpi: int = 200,
        figsize: Optional[Tuple[float, float]] = None,
        csv_path: Optional[str] = None,
        plot_style: LineStyleMode = "lines",
        multiplyx: float = 1.0,
        multiplyz: float = 1.0,
    ) -> plt.Figure:
        t = self.get_tally(tally_id_or_name)

        y, e_centers, _e_edges, score_s, _ = self._spectrum_vector_1d(
            t, "mean", score, nuclide, select
        )
        yerr, _, _, _, _ = self._spectrum_vector_1d(
            t, "std_dev", score, nuclide, select
        )

        e_centers = e_centers * multiplyx
        y = y * multiplyz
        yerr = yerr * multiplyz

        fig, ax = plt.subplots(figsize=figsize)
        fmt = _plot_fmt(plot_style)
        ax.errorbar(e_centers, y, yerr=yerr, fmt=fmt, capsize=capsize)

        x_label = xlabel or "Energy [eV]"
        y_label = ylabel or score_s
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if xscale:
            ax.set_xscale(xscale)
        if yscale:
            ax.set_yscale(yscale)

        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)

        if grid:
            ax.grid(True, alpha=0.3)

        ax.set_title(title or f"Tally {int(t.id)} '{t.name}' | energy spectrum")
        fig.tight_layout()

        if csv_path:
            _save_csv_1d(csv_path, x=e_centers, y=y, yerr=yerr, x_label=x_label, y_label=y_label)

        if fname:
            fig.savefig(fname, dpi=dpi)
        if show:
            plt.show()

        return fig

    def plot_heatmap(
        self,
        tally_id_or_name: Union[int, str],
        plane: str,
        field: Agg,
        score: Optional[str] = None,
        nuclide: Optional[str] = None,
        select: Optional[Dict[str, int]] = None,
        slice_axis: Optional[str] = None,
        slice_index: Optional[int] = None,
        slice_coord: Optional[float] = None,
        slice_range: Optional[Tuple[float, float]] = None,
        reduce_other: Literal["mean", "sum", "max"] = "mean",
        fname: Optional[str] = None,
        title: Optional[str] = None,
        show: bool = True,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        norm: Optional[str] = None,
        cmap: Optional[str] = None,
        colorbar_label: Optional[str] = None,
        log_eps: float = 0.0,
        dpi: int = 200,
        figsize: Optional[Tuple[float, float]] = None,
        Dx: Optional[int] = None,
        Dy: Optional[int] = None,
        csv_path: Optional[str] = None,
        geometry_path: Optional[str] = None,
        multiplyx: float = 1.0,
        multiplyy: float = 1.0,
        multiplyz: float = 1.0,
    ) -> plt.Figure:
        t = self.get_tally(tally_id_or_name)
        mean3d, md, score_s, _ = self._mesh_vector_3d(t, "mean", score, nuclide, select)
        sd3d, _, _, _ = self._mesh_vector_3d(t, "std_dev", score, nuclide, select)

        axis_names = [a.name for a in md.axes]

        def parse_plane(p: str) -> Tuple[str, str]:
            p = p.replace(" ", "").lower()
            for a in axis_names:
                for b in axis_names:
                    if a != b and (a + b) == p:
                        return a, b
            raise ValueError(
                f"Cannot interpret plane='{plane}'. "
                f"Available axes: {axis_names}. Examples: 'xy', 'rz', 'rphi'."
            )

        a_name, b_name = parse_plane(plane)
        a_i = axis_names.index(a_name)
        b_i = axis_names.index(b_name)
        other_i = [i for i in range(3) if i not in (a_i, b_i)][0]

        centers_other = md.axes[other_i].centers

        if slice_axis is not None:
            if slice_axis not in axis_names:
                raise ValueError(f"slice_axis='{slice_axis}' does not exist. Available: {axis_names}")
            if axis_names.index(slice_axis) != other_i:
                raise ValueError(
                    f"slice_axis='{slice_axis}' must be the third axis (not in plane={a_name+b_name}). "
                    f"The third axis is '{axis_names[other_i]}'."
                )

            if slice_coord is not None:
                slice_index = int(np.argmin(np.abs(centers_other - slice_coord)))
            if slice_index is None:
                raise ValueError("Provide slice_index or slice_coord when using slice_axis.")
            if not (0 <= slice_index < md.axes[other_i].size):
                raise ValueError(f"slice_index={slice_index} out of range 0..{md.axes[other_i].size-1}")

            mean2d = np.take(mean3d, slice_index, axis=other_i)
            sd2d = np.take(sd3d, slice_index, axis=other_i)
            slice_label = f"slice {slice_axis} idx={slice_index} (coord~{centers_other[slice_index]:g})"

        elif slice_range is not None:
            lo, hi = float(slice_range[0]), float(slice_range[1])
            if lo > hi:
                lo, hi = hi, lo
            mask = (centers_other >= lo) & (centers_other <= hi)
            idxs = np.where(mask)[0]
            if idxs.size == 0:
                raise ValueError(
                    f"slice_range={slice_range} does not hit any bin of the third axis '{axis_names[other_i]}'. "
                    f"Available range is ~[{centers_other[0]:g}, {centers_other[-1]:g}]"
                )
            mean_cut = np.take(mean3d, idxs, axis=other_i)
            sd_cut = np.take(sd3d, idxs, axis=other_i)

            mean2d = _reduce(mean_cut, (other_i,), reduce_other)
            sd2d = _reduce_unc(sd_cut, mean_cut, (other_i,), reduce_other)
            slice_label = f"reduced over {axis_names[other_i]} in [{lo:g},{hi:g}] ({reduce_other})"
        else:
            mean2d = _reduce(mean3d, (other_i,), reduce_other)
            sd2d = _reduce_unc(sd3d, mean3d, (other_i,), reduce_other)
            slice_label = f"reduced over '{axis_names[other_i]}' ({reduce_other})"

        remaining = [i for i in range(3) if i != other_i]
        a_pos = remaining.index(a_i)
        b_pos = remaining.index(b_i)
        if (a_pos, b_pos) == (0, 1):
            m_plot = mean2d
            s_plot = sd2d
        else:
            m_plot = np.swapaxes(mean2d, 0, 1)
            s_plot = np.swapaxes(sd2d, 0, 1)

        data = s_plot if field == "std" else m_plot
        default_label = f"{score_s} std_dev" if field == "std" else f"{score_s} {field}"
        data = np.asarray(data) * multiplyz

        a_cent = md.axes[a_i].centers * multiplyx
        b_cent = md.axes[b_i].centers * multiplyy

        if Dx is not None and Dy is not None:
            data = _resample_2d_nearest(data, int(Dx), int(Dy))
            a_cent_csv = np.linspace(a_cent[0], a_cent[-1], int(Dx))
            b_cent_csv = np.linspace(b_cent[0], b_cent[-1], int(Dy))
            if figsize is None:
                figsize = _figsize_for_pixels(int(Dx), int(Dy), dpi=dpi)
        else:
            a_cent_csv = a_cent
            b_cent_csv = b_cent

        extent = [a_cent[0], a_cent[-1], b_cent[0], b_cent[-1]]

        norm_obj = None
        if norm is not None:
            n = norm.strip().lower()
            if n == "log":
                d_for = _mask_for_lognorm(data, eps=log_eps)
                dmin = np.nanmin(d_for) if np.isfinite(np.nanmin(d_for)) else None
                use_vmin = vmin if vmin is not None else dmin
                if use_vmin is None or not np.isfinite(use_vmin) or use_vmin <= 0:
                    raise ValueError("logscale requires positive values (use --log-eps or vmin>0).")
                norm_obj = LogNorm(vmin=use_vmin, vmax=vmax)
            else:
                raise ValueError("norm currently supports: None or 'log'")

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(
            data.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            interpolation="nearest",
            vmin=vmin if norm_obj is None else None,
            vmax=vmax if norm_obj is None else None,
            norm=norm_obj,
            cmap=cmap,
        )

        if geometry_path is not None:
            self._overlay_geometry_contours(
                ax=ax,
                geometry_path=geometry_path,
                plane=(a_name + b_name),
                extent=extent,
                slice_axis=axis_names[other_i],
                slice_coord=(0.0 if slice_coord is None else slice_coord),
            )

        x_label = xlabel or a_name
        y_label = ylabel or b_name
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)

        ax.set_title(title or f"Tally {int(t.id)} '{t.name}' | {a_name+b_name} | {slice_label}")
        plt.colorbar(im, ax=ax, label=(colorbar_label or default_label))
        fig.tight_layout()

        if csv_path:
            _save_csv_2d(
                csv_path,
                x_centers=a_cent_csv,
                y_centers=b_cent_csv,
                z=data,
                x_label=x_label,
                y_label=y_label,
                z_label=(colorbar_label or default_label),
            )

        if fname:
            fig.savefig(fname, dpi=dpi)
        if show:
            plt.show()

        return fig


# -------------------------
# CLI helpers
# -------------------------

def _normalize_section(section: str) -> str:
    return section.replace(" ", "").lower()


def _infer_plot_mode(section: str) -> Literal["1d", "2d", "spectrum"]:
    s = _normalize_section(section)
    if s in ("energy", "e", "spectrum"):
        return "spectrum"
    if s in ("phi", "theta"):
        return "1d"
    if len(s) == 1:
        return "1d"
    return "2d"


def _build_slice_from_at(
    md: MeshDef,
    section: str,
    at_values: Dict[str, Optional[float]],
) -> Tuple[Optional[str], Optional[float], str]:
    axes = [a.name for a in md.axes]
    sec = _normalize_section(section)

    if _infer_plot_mode(sec) != "2d":
        return None, None, sec

    def parse_plane(p: str) -> Tuple[str, str]:
        p = p.replace(" ", "").lower()
        for a in axes:
            for b in axes:
                if a != b and (a + b) == p:
                    return a, b
        raise ValueError(f"Cannot interpret section/plane='{section}'. Available axes: {axes}")

    a, b = parse_plane(sec)
    plane_set = {a, b}
    third = [x for x in axes if x not in plane_set]
    if len(third) != 1:
        raise ValueError("Mesh is expected to have exactly 3 axes.")
    third = third[0]

    provided = {k: v for k, v in at_values.items() if v is not None}

    if not provided:
        return None, None, sec

    if third in provided:
        return third, float(provided[third]), sec

    for ax in (a, b):
        if ax in provided:
            slice_axis = ax
            slice_coord = float(provided[ax])
            other_two = [x for x in axes if x != slice_axis]
            effective_plane = other_two[0] + other_two[1]
            return slice_axis, slice_coord, effective_plane

    return None, None, sec


def _print_tallies(tp: OpenMCTallyPlotter) -> None:
    rows = tp.list_tallies()
    if not rows:
        print("No tallies found in the statepoint.")
        return
    for t in rows:
        print(
            f"id={t['id']}  name='{t['name']}'  "
            f"scores={t['scores']}  nuclides={t['nuclides']}  filters={t['filters']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tally_plotter.py",
        description="OpenMC tally plotter (scalar, mesh, and energy spectrum tallies).",
    )

    parser.add_argument("--input", "-i", dest="input_path", help="Path to statepoint .h5 file")
    parser.add_argument("--list", action="store_true", help="List all tallies in the input statepoint and exit")

    parser.add_argument("--tally", "-t", required=False, help="Tally id or name (e.g. 1 or 'flux')")
    parser.add_argument(
        "--geometry",
        default=None,
        help="Optional path to geometry.xml. For 2D Cartesian sections (xy/xz/yz), geometry contours will be overlaid on the tally plot.",
    )

    parser.add_argument(
        "--section",
        "-s",
        required=False,
        default=None,
        help="Plot section: "
             "1D mesh axis (x/y/z/r/phi/theta), "
             "2D mesh plane (xy/xz/yz/rz/rphi/thetaphi/etc.), "
             "or energy spectrum ('energy', 'E', 'spectrum'). "
             "Default: xy for mesh tallies, energy for spectrum tallies.",
    )

    parser.add_argument(
        "--aggregation",
        "-a",
        default="mean",
        choices=["mean", "sum", "max", "std"],
        help="For 1D mesh: reduce mean with mean/sum/max, or plot std. "
             "For 2D mesh: choose one heatmap field. "
             "Ignored for spectrum tallies.",
    )

    parser.add_argument("--score", default=None, help="Score name (optional if tally has exactly one score)")
    parser.add_argument("--nuclide", default=None, help="Nuclide name (default: total/first available)")

    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file path. If not provided, defaults to <tally_name_or_id>_<section>.png.",
    )

    parser.add_argument(
        "--csv",
        action="store_true",
        help="Write a CSV with exactly what is plotted (and only that). The filename defaults to <tally_name_or_id>_<section>.csv "
             "unless --output is provided (then it uses the same stem with .csv).",
    )

    parser.add_argument("--atx", type=float, default=None, help="Slice at x=<value>")
    parser.add_argument("--aty", type=float, default=None, help="Slice at y=<value>")
    parser.add_argument("--atz", type=float, default=None, help="Slice at z=<value> (default for 2D: 0 when section is not provided)")
    parser.add_argument("--atr", type=float, default=None, help="Slice at r=<value>")
    parser.add_argument("--atphi", type=float, default=None, help="Slice at phi=<value>")
    parser.add_argument("--attheta", type=float, default=None, help="Slice at theta=<value>")

    parser.add_argument("--dpi", type=int, default=300, help="Output DPI (default: 300)")
    parser.add_argument("--no-show", action="store_true", help="Do not display figures (useful in batch runs)")

    parser.add_argument("--Dx", type=int, default=None, help="Heatmap output resolution in X pixels (e.g. 800)")
    parser.add_argument("--Dy", type=int, default=None, help="Heatmap output resolution in Y pixels (e.g. 600)")

    parser.add_argument(
        "--logscale",
        action="store_true",
        help="Use log scale (1D/spectrum: y-axis log, 2D: LogNorm).",
    )
    parser.add_argument(
        "--log-eps",
        type=float,
        default=0.0,
        help="Mask values <= log-eps when using logscale.",
    )

    parser.add_argument(
        "--points",
        action="store_true",
        help="Plot 1D/spectrum as points only.",
    )
    parser.add_argument(
        "--linespoints",
        action="store_true",
        help="Plot 1D/spectrum as lines with points.",
    )

    parser.add_argument(
        "--multiplyx",
        type=float,
        default=1.0,
        help="Multiply X-axis values by this factor.",
    )

    parser.add_argument(
        "--multiplyy",
        type=float,
        default=1.0,
        help="Multiply Y-axis values by this factor (used as the second spatial axis scale in 2D plots).",
    )

    parser.add_argument(
        "--multiplyz",
        type=float,
        default=1.0,
        help="Multiply plotted values by this factor. For scalar tallies it multiplies mean and std_dev.",
    )

    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        metavar=("XMIN", "XMAX"),
        default=None,
        help="Set x-axis limits.",
    )

    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        metavar=("YMIN", "YMAX"),
        default=None,
        help="Set y-axis limits.",
    )

    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom plot title.",
    )

    parser.add_argument(
        "--xtitle",
        type=str,
        default=None,
        help="Custom X-axis title.",
    )

    parser.add_argument(
        "--ytitle",
        type=str,
        default=None,
        help="Custom Y-axis title.",
    )

    args = parser.parse_args()

    if args.points and args.linespoints:
        parser.error("Use at most one of --points or --linespoints")

    if args.points:
        plot_style: LineStyleMode = "points"
    elif args.linespoints:
        plot_style = "linespoints"
    else:
        plot_style = "lines"

    if not args.input_path:
        parser.error("You must provide --input statepoint.h5")

    tp = OpenMCTallyPlotter(args.input_path)

    if args.list:
        _print_tallies(tp)
        return

    if args.tally is None:
        parser.error("For plotting, provide --tally ... (or use --list)")

    tally_ident: Union[int, str]
    try:
        tally_ident = int(args.tally)
    except Exception:
        tally_ident = str(args.tally)

    t = tp.get_tally(tally_ident)

    is_mesh = tp._has_mesh_filter(t)
    is_spectrum = tp._has_energy_filter(t) and not is_mesh

    if args.section is None:
        section = "energy" if is_spectrum else "xy"
    else:
        section = _normalize_section(args.section)

    if is_mesh and args.section is None and args.atz is None:
        args.atz = 0.0

    show = not args.no_show
    yscale = "log" if args.logscale else None
    norm = "log" if args.logscale else None

    stem = _default_output_stem(t, tally_ident)

    if args.output is None:
        out_png = _default_png_name(stem, section)
    else:
        out_png = args.output

    if args.csv:
        if args.output is not None:
            base, _ext = os.path.splitext(args.output)
            out_csv = base + ".csv"
        else:
            out_csv = _default_csv_name(stem, section)
    else:
        out_csv = None

    try:
        score_s, nuc_s = tp._default_score_nuclide(t, args.score, args.nuclide)
        scalar_mean = tp._reshaped_compatible(t, value="mean", score=score_s, nuclide=nuc_s)
        scalar_mean = np.asarray(scalar_mean)

        if scalar_mean.ndim == 0 and (not is_mesh) and (not tp._has_energy_filter(t)):
            scalar_sd = tp._reshaped_compatible(t, value="std_dev", score=score_s, nuclide=nuc_s)
            scalar_sd = np.asarray(scalar_sd)

            mean_val = float(scalar_mean) * args.multiplyz
            sd_val = (float(scalar_sd) if scalar_sd.ndim == 0 else float(scalar_sd.ravel()[0])) * args.multiplyz
            print(f"{mean_val}  (std_dev={sd_val})")

            if args.csv and out_csv is not None:
                np.savetxt(out_csv, np.array([[mean_val, sd_val]]), delimiter=",", header="mean,std_dev", comments="")
            return
    except Exception:
        pass

    if is_spectrum or _infer_plot_mode(section) == "spectrum":
        tp.plot_spectrum(
            tally_ident,
            score=args.score,
            nuclide=args.nuclide,
            fname=out_png,
            title=args.title,
            show=show,
            xlim=tuple(args.xlim) if args.xlim is not None else None,
            ylim=tuple(args.ylim) if args.ylim is not None else None,
            xlabel=args.xtitle,
            ylabel=args.ytitle,
            dpi=args.dpi,
            xscale="log",
            yscale=yscale,
            csv_path=out_csv,
            plot_style=plot_style,
            multiplyx=args.multiplyx,
            multiplyz=args.multiplyz,
        )
        return

    if not is_mesh:
        raise SystemExit(
            "This tally is neither a mesh tally nor a pure energy spectrum tally.\n"
            f"Filters: {_filters_debug(t)}"
        )

    md = tp._meshdef(t)
    axes = [a.name for a in md.axes]

    at_values = {
        "x": args.atx,
        "y": args.aty,
        "z": args.atz,
        "r": args.atr,
        "phi": args.atphi,
        "theta": args.attheta,
    }
    at_values = {k: v for k, v in at_values.items() if k in axes}

    mode = _infer_plot_mode(section)

    if mode == "1d":
        axis = section
        if axis not in axes:
            raise SystemExit(f"Requested 1D axis '{axis}' not available. Available axes: {axes}")

        tp.plot_1d(
            tally_ident,
            axis=axis,
            score=args.score,
            nuclide=args.nuclide,
            agg=args.aggregation,  # type: ignore[arg-type]
            fname=out_png,
            title=args.title,
            show=show,
            xlim=tuple(args.xlim) if args.xlim is not None else None,
            ylim=tuple(args.ylim) if args.ylim is not None else None,
            xlabel=args.xtitle,
            ylabel=args.ytitle,
            dpi=args.dpi,
            yscale=yscale,
            csv_path=out_csv,
            plot_style=plot_style,
            multiplyx=args.multiplyx,
            multiplyz=args.multiplyz,
        )
        return

    slice_axis, slice_coord, effective_plane = _build_slice_from_at(md, section, at_values)

    tp.plot_heatmap(
        tally_ident,
        plane=effective_plane,
        field=args.aggregation,  # type: ignore[arg-type]
        score=args.score,
        nuclide=args.nuclide,
        slice_axis=slice_axis,
        slice_coord=slice_coord,
        reduce_other=("mean" if args.aggregation == "std" else args.aggregation),  # type: ignore[arg-type]
        fname=out_png,
        title=args.title,
        show=show,
        xlim=tuple(args.xlim) if args.xlim is not None else None,
        ylim=tuple(args.ylim) if args.ylim is not None else None,
        xlabel=args.xtitle,
        ylabel=args.ytitle,
        dpi=args.dpi,
        norm=norm,
        log_eps=args.log_eps,
        Dx=args.Dx,
        Dy=args.Dy,
        csv_path=out_csv,
        geometry_path=args.geometry,
        multiplyx=args.multiplyx,
        multiplyy=args.multiplyy,
        multiplyz=args.multiplyz,
    )


if __name__ == "__main__":
    main()
