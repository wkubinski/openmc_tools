import numpy as np
import openmc
from pathlib import Path


def count_neutrons_ever_in_materials(material_ids, tracks_path="tracks.h5"):
    # Convert all material IDs to integers
    target_ids = {int(mid) for mid in material_ids}

    if not Path(tracks_path).exists():
        raise FileNotFoundError(f"File not found: {tracks_path}")

    tracks = openmc.Tracks(tracks_path)

    total_source_histories = 0
    matched_source_histories = 0

    total_neutron_tracks = 0
    matched_neutron_tracks = 0

    per_material_neutron_tracks = {mid: 0 for mid in target_ids}
    per_material_source_histories = {mid: 0 for mid in target_ids}

    for track in tracks:
        total_source_histories += 1

        history_hit_any = False
        history_hit_materials = set()

        for ptrack in track.particle_tracks:
            # Count only neutron tracks
            if ptrack.particle != "neutron":
                continue

            total_neutron_tracks += 1

            states = ptrack.states
            if len(states) == 0:
                continue

            # Get material IDs visited by this neutron track
            visited_materials = set(np.unique(states["material_id"])) & target_ids

            if visited_materials:
                matched_neutron_tracks += 1
                history_hit_any = True
                history_hit_materials.update(visited_materials)

                for mid in visited_materials:
                    per_material_neutron_tracks[mid] += 1

        if history_hit_any:
            matched_source_histories += 1

        for mid in history_hit_materials:
            per_material_source_histories[mid] += 1

    return {
        "matched_neutron_tracks": matched_neutron_tracks,
        "total_neutron_tracks": total_neutron_tracks,
        "matched_source_histories": matched_source_histories,
        "total_source_histories": total_source_histories,
        "fraction_neutron_tracks": (
            matched_neutron_tracks / total_neutron_tracks
            if total_neutron_tracks > 0 else 0.0
        ),
        "fraction_source_histories": (
            matched_source_histories / total_source_histories
            if total_source_histories > 0 else 0.0
        ),
        "per_material_neutron_tracks": per_material_neutron_tracks,
        "per_material_source_histories": per_material_source_histories,
    }


def print_report(result):
    print("=== TRACKS.H5 ANALYSIS REPORT ===")
    print(
        f"Neutron tracks that entered selected materials: "
        f"{result['matched_neutron_tracks']} / {result['total_neutron_tracks']} "
        f"({100.0 * result['fraction_neutron_tracks']:.4f}%)"
    )
    print(
        f"Source histories containing at least one such neutron: "
        f"{result['matched_source_histories']} / {result['total_source_histories']} "
        f"({100.0 * result['fraction_source_histories']:.4f}%)"
    )

    print("\n--- Per material: neutron tracks ---")
    for mid, count in sorted(result["per_material_neutron_tracks"].items()):
        print(f"material_id={mid}: {count}")

    print("\n--- Per material: source histories ---")
    for mid, count in sorted(result["per_material_source_histories"].items()):
        print(f"material_id={mid}: {count}")


if __name__ == "__main__":
    material_ids = [4,5,6,8]  # Replace with your material IDs

    result = count_neutrons_ever_in_materials(
        material_ids=material_ids,
        tracks_path="tracks.h5"
    )

    print_report(result)



    tracks = openmc.Tracks("tracks.h5")

    all_materials = set()

    materials = [8,9,11]

    no_hisotory_counter = 0
    tracks_counter = 0
    event_counter = 0
    for track in tracks:
        event = False
        # print(len(track))
        tracks_counter += 1
        history = []
        for t in track:
            for tt in t[1]: 
                material = int(tt[7])
                if material in materials: event = True
                if material !=-1: 
                    history.append(material)
        if len(history)==0: no_hisotory_counter += 1
        if event: event_counter +=1
    print("events", event_counter)
    print("number of zero histories", no_hisotory_counter,"out of", tracks_counter)
