# CCPi-Diamond-Laminography
This repository contains code for processing and understanding the laminography data from the DIAD beamline at Diamond Light Source

## Repository structure
The code is arranged in the following folders
#### Alignment
  Notebooks demonstrating the automatic alignment tool designed to align the tilt and centre of rotation offset in laminography data
  - `laminography_automatic_alignment.ipynb` can be used with DIAD laminography data, the user needs to input the filename, an initial guess of tilt and COR, search ranges and tolerance
    ```
    file_path = '/mnt/share/ALC_laminography/folder/k11-67208_binned.nxs'
    cor = 0 # pix
    tilt = 35 # deg

    optimise_geometry = True
    cor_bounds = (-20, 20) # pixels
    tilt_bounds = (30, 40) # deg
    tilt_tol = 0.01 # deg
    cor_tol = 0.01 # pixels
    ```
  - `example_alignment_TEM_grid.ipynb` and `example_alignment_spheres.ipynb` show examples of the scripts use on real datasets
#### Artefacts
Code demonstrating the kind of artefacts caused by laminography and some methods to combat them
- `example_artefacts_grid.ipynb` can be run without needing any data to be pre-created
- `example_artefacts_cylinder.ipynb` requires simulated data generated in the simulations folder

#### Simulations
Code to create simulated laminography data used in testing the alignment tools and understanding artefacts. Note: code in this folder requires a pre-release version of the gvxr package which is not currently included in the Docker container.

## Running the code 
To run the code in a Jupyter notebook enabled Docker container on a system with access to a GPU, run the following command:
```
docker run --gpus all -p 8888:8888 ghcr.io/tomographicimaging/ccpi-diamond-laminography:latest
```
