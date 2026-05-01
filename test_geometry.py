#!/usr/bin/env python3
import argparse
import colorsys
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import openmc


def make_colors(objects):
    colors = {}
    n = max(len(objects), 1)

    for i, obj in enumerate(objects):
        h = i / n
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
        colors[obj] = (int(255 * r), int(255 * g), int(255 * b))

    return colors


def load_tracks(filename):
    """Load particle tracks from tracks.h5."""
    return openmc.Tracks(filename)


def build_dummy_materials_from_geometry(geometry_xml):
    """Create placeholder materials for all material IDs referenced in geometry.xml."""
    tree = ET.parse(geometry_xml)
    root = tree.getroot()

    material_ids = set()

    for cell in root.iter("cell"):
        material = cell.get("material")
        if material is None:
            continue

        material = material.strip()
        if material in ("", "void"):
            continue

        if material.isdigit():
            material_ids.add(int(material))
            continue

        for token in material.split():
            if token.isdigit():
                material_ids.add(int(token))

    dummy_materials = openmc.Materials()

    for mat_id in sorted(material_ids):
        mat = openmc.Material(material_id=mat_id, name=f"dummy_{mat_id}")
        dummy_materials.append(mat)

    return dummy_materials


def get_track_points(particle_track, basis, interactions=None):
    """Return projected coordinates for one particle track."""
    states = particle_track.states
    r = states["r"]

    n_points = len(r)
    if n_points == 0:
        return None, None

    if interactions is not None:
        n_keep = min(interactions + 1, n_points)
        if n_keep < 1:
            n_keep = 1
        r = r[:n_keep]

    if basis == "xy":
        x = r["x"]
        y = r["y"]
    elif basis == "xz":
        x = r["x"]
        y = r["z"]
    elif basis == "yz":
        x = r["y"]
        y = r["z"]
    else:
        return None, None

    return x, y


def draw_tracks(
    ax,
    tracks,
    basis,
    max_particles=None,
    interactions=None,
    primary_only=False,
):
    """Overlay tracks on the current plot."""
    n_drawn = 0

    for track in tracks:
        if primary_only:
            particle_tracks = track.particle_tracks[:1]
        else:
            particle_tracks = track.particle_tracks

        for particle_track in particle_tracks:
            if max_particles is not None and n_drawn >= max_particles:
                return

            x, y = get_track_points(
                particle_track,
                basis=basis,
                interactions=interactions
            )

            if x is None or len(x) == 0:
                continue

            if len(x) == 1:
                ax.plot(x, y, "o", markersize=4, color="black")
            else:
                ax.plot(x, y, "-", linewidth=1.0, alpha=0.8, color="black")
                ax.plot(x[0], y[0], "o", markersize=3, color="black")

            n_drawn += 1


def point_from_basis(x, y, basis, origin):
    """Convert 2D plot coordinates to 3D geometry point."""
    if basis == "xy":
        return (x, y, origin[2])
    elif basis == "xz":
        return (x, origin[1], y)
    elif basis == "yz":
        return (origin[0], x, y)
    else:
        raise ValueError(f"Unsupported basis: {basis}")


def label_cells(ax, geom, basis, origin, xlim=None, ylim=None, grid_n=100):
    """Draw cell IDs using representative points found with geom.find()."""
    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()

    xs = np.linspace(xlim[0], xlim[1], grid_n)
    ys = np.linspace(ylim[0], ylim[1], grid_n)

    label_positions = {}

    for y in ys:
        for x in xs:
            p = point_from_basis(x, y, basis, origin)

            try:
                path = geom.find(p)
            except Exception:
                continue

            if not path:
                continue

            cell = None
            for obj in reversed(path):
                if isinstance(obj, openmc.Cell):
                    cell = obj
                    break

            if cell is None:
                continue

            if cell.id not in label_positions:
                label_positions[cell.id] = (x, y)

    for cell_id, (x, y) in label_positions.items():
        ax.text(
            x,
            y,
            str(cell_id),
            ha="center",
            va="center",
            fontsize=7,
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.15",
                fc="white",
                alpha=0.7,
                ec="none"
            ),
            zorder=10,
        )


def save_plot(
    geom,
    filename,
    basis,
    origin,
    plot_kwargs,
    xlim,
    ylim,
    tracks=None,
    max_particles=None,
    interactions=None,
    primary_only=False,
    label_cells_flag=False,
):
    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)

    geom.plot(
        basis=basis,
        origin=origin,
        axes=ax,
        **plot_kwargs
    )

    if tracks is not None:
        draw_tracks(
            ax,
            tracks,
            basis,
            max_particles=max_particles,
            interactions=interactions,
            primary_only=primary_only,
        )

    if xlim:
        ax.set_xlim(xlim)

    if ylim:
        ax.set_ylim(ylim)

    if label_cells_flag and plot_kwargs.get("color_by") == "cell":
        label_cells(ax, geom, basis, origin, xlim=xlim, ylim=ylim)

    ax.set_xlabel("cm")
    ax.set_ylabel("cm")

    fig.savefig(filename, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot OpenMC geometry and optionally overlay particle tracks "
            "from tracks.h5."
        )
    )

    parser.add_argument(
        "geometry",
        help="Path to geometry XML file"
    )
    parser.add_argument(
        "--materials",
        help="Path to materials XML file; if given, plot is colored by material"
    )
    parser.add_argument(
        "--tracks",
        help="Path to tracks.h5 file to overlay particle tracks"
    )

    parser.add_argument(
        "--particles",
        type=int,
        help="Show only first N particle tracks"
    )
    parser.add_argument(
        "--interactions",
        type=int,
        help="Cut each track after I interactions; I=0 shows only the source point"
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Show only primary particles, skip secondaries"
    )

    parser.add_argument(
        "--atx",
        type=float,
        default=0.0,
        help="YZ section at x = atx"
    )
    parser.add_argument(
        "--aty",
        type=float,
        default=0.0,
        help="XZ section at y = aty"
    )
    parser.add_argument(
        "--atz",
        type=float,
        default=0.0,
        help="XY section at z = atz"
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=2000,
        help="Plot resolution in pixels per axis, e.g. 2000 means 2000x2000"
    )

    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        metavar=("XMIN", "XMAX"),
        help="X axis limits"
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        metavar=("YMIN", "YMAX"),
        help="Y axis limits"
    )

    parser.add_argument(
        "--section",
        choices=["xy", "xz", "yz"],
        help="Plot only selected section"
    )

    parser.add_argument(
        "--contours-only",
        action="store_true",
        help="Draw only geometry outlines and save as *_outline.png"
    )

    parser.add_argument(
        "--label-cells",
        action="store_true",
        help="Write cell IDs on the plot when coloring by cells"
    )

    args = parser.parse_args()

    if args.materials:
        materials = openmc.Materials.from_xml(args.materials)
        geom = openmc.Geometry.from_xml(args.geometry, materials=materials)
        mats = list(geom.get_all_materials().values())

        plot_kwargs = {
            "color_by": "material",
            "colors": make_colors(mats),
            "legend": not args.contours_only,
            "pixels": (args.resolution, args.resolution),
        }
    else:
        dummy_materials = build_dummy_materials_from_geometry(args.geometry)
        geom = openmc.Geometry.from_xml(args.geometry, materials=dummy_materials)
        cells = list(geom.get_all_cells().values())

        plot_kwargs = {
            "color_by": "cell",
            "colors": make_colors(cells),
            "legend": False,
            "pixels": (args.resolution, args.resolution),
        }

    if args.contours_only:
        plot_kwargs["outline"] = "only"

    tracks = load_tracks(args.tracks) if args.tracks else None

    if args.contours_only:
        sections = {
            "xy": ("xy_outline.png", "xy", (0.0, 0.0, args.atz)),
            "xz": ("xz_outline.png", "xz", (0.0, args.aty, 0.0)),
            "yz": ("yz_outline.png", "yz", (args.atx, 0.0, 0.0)),
        }
    else:
        sections = {
            "xy": ("xy.png", "xy", (0.0, 0.0, args.atz)),
            "xz": ("xz.png", "xz", (0.0, args.aty, 0.0)),
            "yz": ("yz.png", "yz", (args.atx, 0.0, 0.0)),
        }

    if args.section:
        filename, basis, origin = sections[args.section]
        save_plot(
            geom,
            filename,
            basis,
            origin,
            plot_kwargs,
            args.xlim,
            args.ylim,
            tracks=tracks,
            max_particles=args.particles,
            interactions=args.interactions,
            primary_only=args.primary_only,
            label_cells_flag=args.label_cells,
        )
        print(f"Saved: {filename}")
    else:
        saved = []
        for filename, basis, origin in sections.values():
            save_plot(
                geom,
                filename,
                basis,
                origin,
                plot_kwargs,
                args.xlim,
                args.ylim,
                tracks=tracks,
                max_particles=args.particles,
                interactions=args.interactions,
                primary_only=args.primary_only,
                label_cells_flag=args.label_cells,
            )
            saved.append(filename)

        print("Saved: " + ", ".join(saved))


if __name__ == "__main__":
    main()
