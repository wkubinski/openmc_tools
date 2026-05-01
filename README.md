# openmc_tools

This project provides a set of scripts for post-processing and analyzing OpenMC simulations, as well as basic geometry handling and visualization tools.

## Features

- Stochastic estimation of material masses and volumes  
- Particle track analysis and reaction inspection  
- Quick tally plotting with geometry overlays  
- Geometry generation from CAD (STEP) files  
- Fast visualization of geometry slices and particle tracks  

## Requirements

- Python 3.x  
- OpenMC  
- FreeCAD (required for CAD to MC conversion)

## Scripts

### `analyse_masses.py`
Stochastically calculates material masses.

### `analyse_tracks.py`
Used to inspect particle histories and determine in which materials reactions occurred.

### `analyse_volumes.py`
Stochastically calculates volumes.

### `geouned_cad_to_mc.py`
Generates OpenMC/MCNP geometry from STEP files.  
Requires FreeCAD to be installed and the path to `FreeCAD/lib` to be specified.

### `tally_plotter.py`
Quick plotting tool for OpenMC tallies.  
Supports overlaying geometry boundaries on the results.

### `test_geometry.py`
Allows quick generation of geometry slices in the **xy**, **xz**, and **yz** planes.  
Also supports overlaying particle tracks on the geometry.
