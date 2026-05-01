#!/usr/bin/env python3
import openmc
import csv
import argparse
import subprocess
import glob

# =========================
# 1. Arguments
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--geometry", default="geometry.xml")
parser.add_argument("--materials", default="materials.xml")
parser.add_argument("--output", default="materials_summary.csv")
parser.add_argument("--samples", type=int, default=1_000_000)

args = parser.parse_args()

# =========================
# 2. Load geometry/materials
# =========================
geometry = openmc.Geometry.from_xml(args.geometry)
materials = openmc.Materials.from_xml(args.materials)

mat_dict = {m.id: m for m in materials}
all_materials = list(geometry.get_all_materials().values())

# =========================
# 3. Volume calculation setup
# =========================
lower_left, upper_right = geometry.bounding_box

vol_calc = openmc.VolumeCalculation(
    domains=all_materials,
    samples=args.samples,
    lower_left=lower_left,
    upper_right=upper_right
)

settings = openmc.Settings()
settings.run_mode = 'volume'
settings.batches = 1
settings.volume_calculations = [vol_calc]

geometry.export_to_xml()
materials.export_to_xml()
settings.export_to_xml()

# =========================
# 4. Run OpenMC
# =========================
print("Running OpenMC...")
subprocess.run(["openmc"], check=True)

# =========================
# 5. Get statepoint
# =========================
sp_file = sorted(glob.glob("statepoint.*.h5"))[-1]
print("Using:", sp_file)

sp = openmc.StatePoint(sp_file)
volumes = sp.volume_calculations[0].volumes

# =========================
# 6. Compute masses
# =========================
results = []
total_mass = 0.0

for mat_id, (vol, err) in volumes.items():
    mat = mat_dict[mat_id]
    density = mat.get_mass_density()
    mass = density * vol

    total_mass += mass

    results.append([mat_id, mat.name, vol, err, density, mass])

# =========================
# 7. Save CSV
# =========================
with open(args.output, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "name", "volume_cm3", "std_dev", "density_g_cm3", "mass_g"])

    for row in results:
        writer.writerow(row)

    writer.writerow([])
    writer.writerow(["TOTAL", "", "", "", "", total_mass])

print("Saved:", args.output)

# =========================
# 8. Print summary
# =========================
print("\nSummary:\n")
for r in results:
    print(f"{r[0]:<5} {r[1]:<20} V={r[2]:.3e}  rho={r[4]:.3f}  m={r[5]:.3e}")

print("\nTotal mass:", f"{total_mass:.3e}", "g")
