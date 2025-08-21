# %%
import os
import sys
import numpy as np
import scipy
import json
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from gvxrPython3 import gvxr
from gvxrPython3.JSON2gVXRDataReader import *


from cil.processors import TransmissionAbsorptionConverter
from cil.utilities.display import show_geometry, show2D
from cil.utilities.jupyter import islicer

from cil.processors import FluxNormaliser, Normaliser, CentreOfRotationCorrector
from cil.framework import AcquisitionData, AcquisitionGeometry
from cil.plugins.astra.processors import FBP
from cil.plugins.astra.operators import ProjectionOperator
from cil.io import NEXUSDataWriter

# %%
def create_cylinder_with_spheres(simulation_name='cylinder', cylinder_radius = 100, plane = 'xy', tilt=0, tilt_axis=np.array([0, 1, 0])):
    
    gvxr.removePolygonMeshesFromSceneGraph()
    sphere_radius =  cylinder_radius/10
    gvxr.makeCylinder(simulation_name, 100, sphere_radius*2, cylinder_radius, "um")

    sphere_spacing = 2 * sphere_radius + (sphere_radius/2)

    n_steps = int((cylinder_radius - sphere_radius) // sphere_spacing)


    i_values = np.arange(-n_steps, n_steps + 1) * sphere_spacing
    j_values = np.arange(-n_steps, n_steps + 1) * sphere_spacing

    
    positions = [
        (i, j)
        for i in i_values
        for j in j_values
        if np.sqrt(i**2 + j**2) <= cylinder_radius - sphere_radius
    ]

    if plane == 'xy': # I am using CIL definitions
        translations = [(i, j, 0) for i, j in positions]
        gvxr.rotateNode(simulation_name, 90, 1, 0, 0)
        gvxr.applyCurrentLocalTransformation(simulation_name)
    elif plane == 'xz': # I am using CIL definitions
        translations = [(0, i, j) for i, j in positions]
        gvxr.rotateNode(simulation_name, 90, 1, 0, 0)
        gvxr.rotateNode(simulation_name, 90, 0, 0, 1)
        gvxr.applyCurrentLocalTransformation(simulation_name)
    elif plane == 'yz': # I am using CIL definitions
        translations = [(i, 0, j) for i, j in positions]
    else:
        raise ValueError(f"Unsupported plane: {plane}")

                
    for N, (x, y, z) in enumerate(translations):
        sphere_name = f"sphere_{N}"
        # print(sphere_name)
        gvxr.makeSphere(sphere_name, 50, 50, sphere_radius, "um")
        gvxr.translateNode(sphere_name, x, y, z, "um")
        gvxr.applyCurrentLocalTransformation(sphere_name)
        gvxr.addMesh(simulation_name, sphere_name)

    gvxr.addPolygonMeshAsInnerSurface(simulation_name)
    gvxr.setCompound(simulation_name, "SiO2")
    gvxr.setDensity(simulation_name, 2.2,"g.cm-3")

    gvxr.rotateNode(simulation_name, tilt, *tilt_axis)

# point to the digital twin code
sys.path.append(os.path.abspath('../../DIAD2gVXR/code'))
from DiadModel import DiadModel

# create a digital twin simulation and initialise with some experimental parameters
diad_model = DiadModel()
pixels_x = 500
pixels_y = 500
diad_model.detector_cols = pixels_x
diad_model.detector_rows = pixels_y
diad_model.initSimulationEnegine()
energy_in_keV = 25
exposure_in_sec = 3
diad_model.initExperimentalParameters(1, "m", energy_in_keV, exposure_in_sec)
# %%
simulation_name = "cylinder"
tilt = 30 # degrees
tilt_axis = np.array([0, 1, 0]) # around the detector x direction
create_cylinder_with_spheres(simulation_name=simulation_name, cylinder_radius=100, plane='xy', tilt=tilt, tilt_axis=tilt_axis)
gvxr.displayScene()
xray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
show2D([xray_image], 'Tilted sample')
# %%
# specify number of projections
start = 0
stop = 360
step = 1
angle_set = np.arange(start, stop, step)
xray_image_set = np.zeros((stop, pixels_x, pixels_y), dtype=np.float32)

# specify the rotation axis, around z
rotation_axis = np.array([0, 0, 1])
pixel_cor_offset = 5

for N in angle_set:
    # Rotate
    gvxr.translateNode(simulation_name, 0, pixel_cor_offset*0.5, 0, "um") # pixels are 0.5um
    gvxr.rotateNode(simulation_name, N, *rotation_axis)
    gvxr.translateNode(simulation_name, 0, -pixel_cor_offset*0.5, 0, "um") # pixels are 0.5um
    # Compute xray image
    xray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
    xray_image_set[N] = xray_image
    # Rotate back to origin
    gvxr.translateNode(simulation_name, 0, pixel_cor_offset*0.5, 0, "um") # pixels are 0.5um

    gvxr.rotateNode(simulation_name, -N, *rotation_axis)
    gvxr.translateNode(simulation_name, 0, -pixel_cor_offset*0.5, 0, "um") # pixels are 0.5um


# use the islicer tool to scroll through the projections
islicer(xray_image_set)
# %%
beam_direction = np.array([0, 1, 0])
detector_x_direction = np.array([1, 0, 0])
detector_y_direction = np.array([0, 0, -1])
rotation_axis = np.array([0, 0, 1]) # the untilted rotation axis

# create the tilted rotation axis
tilt_rad = np.deg2rad(tilt)
rotation_matrix = R.from_rotvec(tilt_rad * detector_x_direction)
tilted_rotation_axis = rotation_matrix.apply(rotation_axis)

ag = AcquisitionGeometry.create_Parallel3D(ray_direction = beam_direction,
                                      detector_direction_x = np.array([1, 0, 0]),
                                      detector_direction_y = np.array([0, 0, -1]),
                                      rotation_axis_direction = list(rotation_axis))                   
ag.set_angles(angle_set)
ag.set_panel((pixels_x, pixels_y),
             list([diad_model.effective_pixel_spacing_in_um[0]/1000, diad_model.effective_pixel_spacing_in_um[0]/1000]))

show_geometry(ag)

data = AcquisitionData(xray_image_set, geometry=ag)
data.reorder('astra')
# %%
# apply Beer-Lambert law
data = TransmissionAbsorptionConverter(white_level=1.0)(data)
show2D(data, slice_list=[('angle', 0), ('angle', 45), ('angle',90), ('angle', 135), ('angle',180)], num_cols=5)
# %%
# Reconstruct using FBP
recon = FBP(image_geometry=None, acquisition_geometry=ag)(data)

# Plot the results
slice_list = [('vertical','centre'), ('horizontal_y',int(recon.shape[2]/2)), ('horizontal_x',int(recon.shape[1]/2))]
show2D(recon,
       slice_list=slice_list,
       num_cols=3)
# %%
data.geometry.set_centre_of_rotation(offset=5, distance_units='pixels')
data.geometry.config.system.rotation_axis.direction = tilted_rotation_axis
print(data.geometry.config.system.rotation_axis.position)
print(data.geometry.config.system.rotation_axis.direction)
# %%
ag = data.geometry
ig = ag.get_ImageGeometry()
ig.voxel_num_z = 1
recon = FBP(image_geometry=ig, acquisition_geometry=ag)(data)
show2D(recon,
       slice_list=slice_list,
       num_cols=3)
# %%
output_path = '../output_data'
file_name = os.path.join(output_path, 'cylinder_tilt_'+ str(tilt) + '_cor_offset_' + str(pixel_cor_offset)) #### update the filename here ####
NEXUSDataWriter(data=data, file_name=file_name).write()
# %%
gvxr.destroy()