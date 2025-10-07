# %%
import os
import sys
import numpy as np

import scipy
import json

import matplotlib.pyplot as plt
from gvxrPython3 import gvxr
from gvxrPython3 import gvxr2json
from gvxrPython3.JSON2gVXRDataReader import *

sys.path.append(os.path.abspath('../../DIAD2gVXR/code'))
from DiadModel import DiadModel

from cil.processors import TransmissionAbsorptionConverter
from cil.utilities.display import show_geometry, show2D
from cil.utilities.jupyter import islicer
from scipy.spatial.transform import Rotation as R
from cil.processors import FluxNormaliser, Normaliser
from cil.framework import AcquisitionData, AcquisitionGeometry
from cil.plugins.astra.processors import FBP


# %%
tilt = 30
output_path = "output_data"
simulation_name = "cylinder_exp"
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
# %% Use DIAD model
diad_model = DiadModel()
# bin to make the array smaller
diad_model.detector_cols = diad_model.detector_cols/4
diad_model.detector_rows = diad_model.detector_rows/4
diad_model.effective_pixel_spacing_in_um = [diad_model.effective_pixel_spacing_in_um[0]*4, diad_model.effective_pixel_spacing_in_um[1]*4]
diad_model.initSimulationEnegine()

energy_in_keV = 25
exposure_in_sec = 20
gain = diad_model.initExperimentalParameters(energy_in_keV, exposure_in_sec)

# %%
gvxr.removePolygonMeshesFromSceneGraph()
# Locate the sample STL file
fname =  "spheres_2.stl" # file from Tristan

gvxr.loadMeshFile(simulation_name, fname, "mm")

gvxr.moveToCentre(simulation_name)
gvxr.applyCurrentLocalTransformation(simulation_name)

gvxr.addPolygonMeshAsInnerSurface(simulation_name)
gvxr.setCompound(simulation_name, "SiO2")
gvxr.setDensity(simulation_name, 2.2,"g.cm-3")

# Compute an X-ray image
print("Compute an X-ray image")
gvxr.displayScene()
x_ray_image = (gain * np.array(gvxr.computeXRayImage()).astype(np.single)).astype(np.uint16)
show2D(x_ray_image)

# %%
# Tilt the sample and compute an x-ray image
tilt_axis = np.array([0, 1, 0])
gvxr.rotateNode(simulation_name, tilt, *tilt_axis)
print("Compute an X-ray image")
x_ray_image = (gain * np.array(gvxr.computeXRayImage()).astype(np.single)).astype(np.uint16)
show2D(x_ray_image)

# %% Check CT rotation axis
gvxr.rotateNode(simulation_name, -20, 0, 0, 1)

print("Compute an X-ray image")
x_ray_image = (gain * np.array(gvxr.computeXRayImage()).astype(np.single)).astype(np.uint16)
show2D(x_ray_image)
# %% Simulate a CT scan
start = 0
stop = 360
step = 0.5
angle_set = np.arange(start, stop, step)
xray_image_set = np.zeros((len(angle_set), gvxr.getDetectorNumberOfPixels()[1], gvxr.getDetectorNumberOfPixels()[0]))

for i in np.arange(len(angle_set)):
    # Rotate
    
    gvxr.rotateNode(simulation_name, step, 0, 0, 1)

    # Compute xray image
    xray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
    xray_image_set[i] = xray_image

islicer(xray_image_set)

# %%
beam_direction = np.array(gvxr.getDetectorPosition("mm"))/np.linalg.norm(np.array(gvxr.getDetectorPosition("mm")))
axis_to_apply_tilt = np.array([0, -1, 0])
orthogonal_axis = np.array(gvxr.getDetectorUpVector())

print("Apply tilt: ", tilt, " degrees, along direction: ", axis_to_apply_tilt)
rotation_matrix = R.from_rotvec(np.radians(tilt) * axis_to_apply_tilt)
print("Orthogonal axis: ", orthogonal_axis)
tilted_axis = rotation_matrix.apply(orthogonal_axis)
print("Tilted rotation axis: ", tilted_axis)

# Also apply a little error in other axes
error = -10
axis_error = np.array([1, 0, 0])
print("Apply error: ", error, " degrees, along direction: ", axis_error)
rotation_matrix = R.from_rotvec(np.radians(error) * axis_error)
rotation_axis = rotation_matrix.apply(tilted_axis)
print("Tilted rotation axis plus error: ", rotation_axis)

error = 10
axis_error = np.array([0, 0, 1])
print("Apply error: ", error, " degrees, along direction: ", axis_error)
rotation_matrix = R.from_rotvec(np.radians(error) * axis_error)
rotation_axis = rotation_matrix.apply(rotation_axis)
print("Tilted rotation axis plus error: ", rotation_axis)

# %%
ag = AcquisitionGeometry.create_Parallel3D(ray_direction = beam_direction,
                                      detector_position = gvxr.getDetectorPosition("mm"),
                                      detector_direction_x = gvxr.getDetectorRightVector(),
                                      detector_direction_y = gvxr.getDetectorUpVector(),
                                      rotation_axis_position = gvxr.getCentreOfRotationPositionCT("mm"),
                                      rotation_axis_direction = rotation_axis)                                 
ag.set_angles(angle_set)
ag.set_panel(list(gvxr.getDetectorNumberOfPixels()),
             list([diad_model.effective_pixel_spacing_in_um[0]/1000, diad_model.effective_pixel_spacing_in_um[0]/1000]))


ag.set_centre_of_rotation(offset=0.0080)
# show_geometry(ag)

data = AcquisitionData(xray_image_set, geometry=ag)
# islicer(data)

# Normaliser
max = data.max()
min = data.min()
data_norm = Normaliser(flat_field=max*np.ones((data.shape[1],data.shape[2])),
                       dark_field=(min-0.1*(max-min))*np.ones((data.shape[1],data.shape[2])))(data)
# print(data_norm.min())
# print(data_norm.max())
# show2D(data_norm)

# Apply Beer-Lambert law
data_norm = TransmissionAbsorptionConverter(white_level=1.01)(data_norm)
# print(data_norm.min())
# print(data_norm.max())
# show2D(data_norm)

# FluxNormaliser
data_norm = FluxNormaliser(flux=data_norm.max(), target=1)(data_norm)
# print(data_norm.min())
# print(data_norm.max())
# show2D(data_norm)

data_norm.reorder('astra')
ag = data_norm.geometry
ig = ag.get_ImageGeometry()
fbp = FBP(ig, ag)
fbp.set_input(data_norm)
recon = fbp.get_output()
recon.apply_circular_mask(0.9)
show2D(recon, slice_list=('horizontal_x', 320), origin='lower-right')
# %%
print(ag)