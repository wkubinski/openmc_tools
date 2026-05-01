#!/usr/bin/env python3
import openmc
import csv
import argparse
import glob

# =========================
# 1. Arguments
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--materials", default="materials.xml")
parser.add_argument("--volume", default=None, help="volume*.h5 file")
parser.add_argument("--output", default="materials_summary.csv")

args = parser.parse_args()

# =========================
# 2. Find volume file
# =========================
if args.volume:
    vol_file = args.volume
else:
    files = glob.glob("volume*.h5") + glob.glob("volumes*.h5")
    if not files:
        raise RuntimeError("No volume file found (volume*.h5)")
    vol_file = sorted(files)[-1]

print("Using volume file:", vol_file)

# =========================
# 3. Load materials
# =========================
materials = openmc.Materials.from_xml(args.materials)
mat_dict = {m.id: m for m in materials}

# =========================
# 4. Load volume results
# =========================
vol_calc = openmc.VolumeCalculation.from_hdf5(vol_file)
volumes = vol_calc.volumes

# =========================
# 5. Compute masses
# =========================
results = []
total_mass = 0.0

for mat_id, v in volumes.items():
    vol = v.n      # nominal volume
    err = v.s      # standard deviation

    mat = mat_dict.get(mat_id)

    if mat is None:
        print(f"Warning: material {mat_id} not found")
        continue

    rho = mat.get_mass_density()
    mass = rho * vol

    total_mass += mass

    results.append([mat_id, mat.name, vol, err, rho, mass])

# =========================
# 6. Save CSV
# =========================
with open(args.output, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "id",
        "name",
        "volume_cm3",
        "volume_std_dev",
        "density_g_cm3",
        "mass_g"
    ])

    for r in results:
        writer.writerow(r)

    writer.writerow([])
    writer.writerow(["TOTAL", "", "", "", "", total_mass])

print("Saved:", args.output)

# =========================
# 7. Print summary
# =========================
print("\nSummary:\n")

for r in results:
    print(f"{r[0]:<5} {r[1]:<20} V={r[2]:.3e} ±{r[3]:.2e}  rho={r[4]:.3f}  m={r[5]:.3e}")

print("\nTotal mass:", f"{total_mass:.3e}", "g")
