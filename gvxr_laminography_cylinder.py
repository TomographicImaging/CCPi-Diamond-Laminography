# %%
import os
import sys
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from gvxrPython3 import gvxr
from gvxrPython3.JSON2gVXRDataReader import *

#see https://github.com/TomographicImaging/DIAD2gVXR/tree/main/code
sys.path.append(os.path.abspath('../DIAD2gVXR/code'))
from DiadModel import DiadModel

from cil.processors import TransmissionAbsorptionConverter
from cil.utilities.display import show_geometry, show2D
from cil.utilities.jupyter import islicer

from cil.processors import FluxNormaliser, Normaliser
from cil.framework import AcquisitionData, AcquisitionGeometry
from cil.plugins.astra.processors import FBP
from cil.io import NEXUSDataWriter
# %%
tilt = 30
output_path = "output_data"
simulation_name = "cylinder"
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
# %% Use DIAD model
diad_model = DiadModel()
diad_model.detector_cols = 500
diad_model.detector_rows = 500
diad_model.initSimulationEnegine()

energy_in_keV = 25
exposure_in_sec = 100
gain = diad_model.initExperimentalParameters(energy_in_keV, exposure_in_sec)


# %%
gvxr.removePolygonMeshesFromSceneGraph()
cylinder_radius = 100 # 500
sphere_radius = 10 # 50
gvxr.makeCylinder(simulation_name, 100, sphere_radius*2, cylinder_radius, "um")
gvxr.setNodeTransformationMatrix(simulation_name, [[1, 0, 0, 0], 
                                                   [0, 1, 0, 0], 
                                                   [0, 0, 1, 0], 
                                                   [0, 0, 0, 1]])
gvxr.rotateNode(simulation_name, 90, 1, 0, 0)

sphere_spacing = 2 * sphere_radius + 5

n_steps = int((cylinder_radius - sphere_radius) // sphere_spacing)

x_values = np.arange(-n_steps, n_steps + 1) * sphere_spacing
y_values = np.arange(-n_steps, n_steps + 1) * sphere_spacing

x_positions = []
y_positions = []
for xi in x_values:
    for yi in y_values:
        if np.sqrt(xi**2 + yi**2) <= cylinder_radius - sphere_radius:
            x_positions.append(xi)
            y_positions.append(yi)
            
for i, (xi, yi) in enumerate(zip(x_positions, y_positions)):
    sphere_name = f"sphere_{i}"
    print(sphere_name)

    gvxr.makeSphere(sphere_name, 50, 50, sphere_radius, "um")
    gvxr.translateNode(sphere_name, xi, 0, yi, "um")
    gvxr.applyCurrentLocalTransformation(sphere_name)
    gvxr.addMesh(simulation_name, sphere_name)

gvxr.addPolygonMeshAsInnerSurface(simulation_name)
gvxr.setCompound(simulation_name, "SiO2")
gvxr.setDensity(simulation_name, 2.2,"g.cm-3")

# Compute an X-ray image
print("Compute an X-ray image")
gvxr.displayScene()
x_ray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
show2D(x_ray_image)

# %%
# Tilt the sample and compute an x-ray image
tilt_axis = np.array([0, 0, 1])
gvxr.rotateNode(simulation_name, tilt, *tilt_axis)
print("Compute an X-ray image")
x_ray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
show2D(x_ray_image)

# %% Check CT rotation axis
# gvxr.rotateNode(simulation_name, 20, 0, 1, 0)

# print("Compute an X-ray image")
# x_ray_image = (gain * np.array(gvxr.computeXRayImage()).astype(np.single)).astype(np.uint16)
# show2D(x_ray_image)

# %% Simulate a CT scan
start = 0
stop = 360
step = 1
angle_set = np.arange(start, stop, step)
xray_image_set = np.zeros((stop, gvxr.getDetectorNumberOfPixels()[1], gvxr.getDetectorNumberOfPixels()[0]))

for i in angle_set:
    # Rotate
    gvxr.rotateNode(simulation_name, step, 0, 1, 0)
    # Compute xray image
    xray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
    xray_image_set[i] = xray_image

islicer(xray_image_set)

# %%
beam_direction = np.array(gvxr.getDetectorPosition("mm"))/np.linalg.norm(np.array(gvxr.getDetectorPosition("mm")))
axis_to_apply_tilt = np.array([0, 1, 0]) # I don't understand why this isn't the beam direction

print("Apply tilt: ", tilt, " degrees, along direction: ", axis_to_apply_tilt)
rotation_matrix = R.from_rotvec(np.radians(tilt) * axis_to_apply_tilt)
orthogonal_axis = np.array(gvxr.getDetectorUpVector())
print("Untilted rotation axis: ", orthogonal_axis)
rotation_axis = rotation_matrix.apply(orthogonal_axis)
print("Tilted rotation axis: ", rotation_axis)

# %% 
ag = AcquisitionGeometry.create_Parallel3D(ray_direction = beam_direction,
                                      detector_position = list(gvxr.getDetectorPosition("mm")),
                                      detector_direction_x = list(gvxr.getDetectorRightVector()),
                                      detector_direction_y = list(gvxr.getDetectorUpVector()),
                                      rotation_axis_position = list(gvxr.getCentreOfRotationPositionCT("mm")),
                                      rotation_axis_direction = list(rotation_axis))                                 
ag.set_angles(angle_set)
ag.set_panel(list(gvxr.getDetectorNumberOfPixels()),
             list([diad_model.effective_pixel_spacing_in_um[0]/1000, diad_model.effective_pixel_spacing_in_um[0]/1000]))
show_geometry(ag)
# %%
data = AcquisitionData(xray_image_set, geometry=ag)
islicer(data)
# %%
# Normaliser
max = data.max()
min = data.min()
data_norm = Normaliser(flat_field=max*np.ones((data.shape[1],data.shape[2])),
                       dark_field=(min-0.1*(max-min))*np.ones((data.shape[1],data.shape[2])))(data)
print(data_norm.min())
print(data_norm.max())
show2D(data_norm)

# Apply Beer-Lambert law
data_norm = TransmissionAbsorptionConverter(white_level=1.01)(data_norm)
print(data_norm.min())
print(data_norm.max())
show2D(data_norm)

# FluxNormaliser
data_norm = FluxNormaliser(flux=data_norm.max(), target=1)(data_norm)
print(data_norm.min())
print(data_norm.max())
show2D(data_norm)
# %%

data_norm.reorder('astra')
fbp = FBP(data_norm.geometry.get_ImageGeometry(), data_norm.geometry)
recon = fbp(data_norm)
recon.apply_circular_mask(0.9)
show2D([recon]) 

# %% Save normalised data with geometry
file_name = os.path.join(output_path, simulation_name + str(len(angle_set))) #### update the filename here ####
data_norm.reorder('cil')
NEXUSDataWriter(data=data_norm, file_name=file_name).write()

# %%


# %%
data_filtered = data_norm.copy()
data_filtered.fill(scipy.ndimage.sobel(data_norm.as_array(), axis=0, mode='reflect', cval=0.0))
# %%
data_norm.reorder('astra')
data_filtered.reorder('astra')
ag = data_norm.geometry
ig = ag.get_ImageGeometry()
# %%

offset = 0

recon_filtered_list = []
recon_list = []
tilts = np.arange(28.0, 32.1, 0.5)

evaluation = np.zeros(len(tilts))
evaluation_filtered = np.zeros(len(tilts))
for i, tilt in enumerate(tilts):
    ag_tilt = ag.copy()
    ag_tilt.set_centre_of_rotation(offset=offset)

    rotation_matrix = R.from_rotvec(np.radians(tilt) * axis_to_apply_tilt)
    rotation_axis = np.array(gvxr.getDetectorUpVector())
    rotation_axis = rotation_matrix.apply(rotation_axis)

    ag_tilt.config.system.rotation_axis.direction = rotation_axis
    ig_tilt = ag_tilt.get_ImageGeometry()
    ig_tilt.voxel_num_z = 1

    fbp = FBP(ig_tilt, ag_tilt)
    fbp.set_input(data_filtered)
    recon_filtered = fbp.get_output()
    recon_filtered.apply_circular_mask(0.9)
    recon_filtered_list.append(recon_filtered.array)
    evaluation_filtered[i] = (recon_filtered*recon_filtered).sum()

    fbp.set_input(data_norm)
    recon = fbp.get_output()
    recon.apply_circular_mask(0.9)
    recon_list.append(recon.array)

    evaluation[i] = (recon*recon).sum()
# %%
from cil.framework import DataContainer
DC_filtered = DataContainer(np.stack(recon_filtered_list, axis=0), dimension_labels=('Tilt',) + recon_filtered.geometry.dimension_labels)
DC_recon = DataContainer(np.stack(recon_list, axis=0), dimension_labels=('Tilt',) + recon.geometry.dimension_labels)
# %%
islicer(DC_recon)
# %%
[fig, axs] = plt.subplots(1,2, figsize=(10,3))
ax = axs[0]
ax.plot(tilts, evaluation)
ax.grid()
ax.set_xlabel('Tilt')
ax.set_ylabel('Reconstruction sum of squares')

ax = axs[1]
ax.plot(tilts, evaluation_filtered)
ax.grid()
ax.set_xlabel('Tilt')
ax.set_ylabel('Filtered reconstruction sum of squares')

 # %%
# plt.plot(A)

# plt.plot(B)
A =  (evaluation - evaluation.min()) / (evaluation.max() - evaluation.min())
B =  (evaluation_filtered - evaluation_filtered.min()) / (evaluation_filtered.max() - evaluation_filtered.min())
# plt.plot(tilts, A)
# plt.plot(tilts, B)
plt.plot(tilts, A-B)
# plt.plot(tilts[(A-B).argmax()], (A-B)[(A-B).argmax()], 'rx')
# %%
