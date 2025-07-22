# %%
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from gvxrPython3 import gvxr
from gvxrPython3.JSON2gVXRDataReader import *

from cil.plugins.astra.processors import FBP
from cil.processors import TransmissionAbsorptionConverter
from cil.utilities.display import show_geometry, show2D
from cil.utilities.jupyter import islicer
from cil.processors import FluxNormaliser, Normaliser
from cil.framework import AcquisitionData, AcquisitionGeometry
from cil.io import NEXUSDataWriter
# %%
simulation_name = "test"

tilt = 30 # degrees

tilt_direction = np.array([1, 0, 0])
beam_direction = np.array([0, 1, 0])
untilted_rotation_axis = np.array([0, 0, 1]) # detector up vector

offsets = np.array([0, 0, 0]) # need to try offsets

print("Apply tilt: ", tilt, " degrees, along direction: ", tilt_direction)
rotation_matrix = R.from_rotvec(np.radians(tilt) * tilt_direction)
print("Untilted rotation axis: ", untilted_rotation_axis)
tilted_rotation_axis = rotation_matrix.apply(untilted_rotation_axis)
print("Tilted rotation axis: ", tilted_rotation_axis)

# %% Create the experiment geometry
# Set up the source
size = 500
print("Create an OpenGL context")
gvxr.createOpenGLContext()
print("Set up the beam")
energy = 30
energy_units = "keV"
photons = 10000
gvxr.setSourcePosition(*-0.5*beam_direction, "cm")
gvxr.setMonoChromatic(energy, energy_units, photons)
gvxr.useParallelBeam()

# Set up the detector
print("Set up the detector")
gvxr.setDetectorPosition(*0.5*beam_direction, "cm")
gvxr.setDetectorUpVector(0, 0, -1)
gvxr.setDetectorNumberOfPixels(size, size)
pixel_size_um = 0.5
gvxr.setDetectorPixelSize(pixel_size_um, pixel_size_um, "um")
# %%

gvxr.removePolygonMeshesFromSceneGraph()
fname = "../stl_files/cylinder_with_spheres.stl"
if not os.path.exists(fname):
    raise IOError(fname)
gvxr.loadMeshFile(simulation_name, fname, "mm")

gvxr.rotateNode(simulation_name, 90, 1, 0, 0) # just doing this because the stl file is the wrong way round
gvxr.applyCurrentLocalTransformation(simulation_name) # apply this here to set reference frame to here

gvxr.addPolygonMeshAsInnerSurface(simulation_name)
gvxr.setCompound(simulation_name, "SiO2")
gvxr.setDensity(simulation_name, 2.2,"g.cm-3")

gvxr.getLinearAttenuationCoefficient(simulation_name, energy, energy_units)

# Compute an X-ray image
gvxr.displayScene()
x_ray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
show2D(x_ray_image)

# %%
# Tilt the sample and compute an x-ray image
gvxr.rotateNode(simulation_name, tilt, *tilt_direction)
gvxr.applyCurrentLocalTransformation(simulation_name)
print("Compute an X-ray image")
x_ray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
show2D(x_ray_image)


# %% Check CT rotation axis
# gvxr.rotateNode(simulation_name, 40, 0, 1, 0)

# print("Compute an X-ray image")
# x_ray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
# show2D(x_ray_image)

# %% Simulate a CT scan
start = 0
stop = 360
step = 1
angle_set = np.arange(start, stop, step)
xray_image_set = np.zeros((stop, gvxr.getDetectorNumberOfPixels()[1], gvxr.getDetectorNumberOfPixels()[0]))

for i in angle_set:
    # Rotate
    gvxr.rotateNode(simulation_name, angle_set[i], *tilted_rotation_axis)
    gvxr.applyCurrentLocalTransformation(simulation_name)
    # Compute xray image
    xray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
    xray_image_set[i] = xray_image

    # Reverse rotation
    gvxr.rotateNode(simulation_name, -angle_set[i], *tilted_rotation_axis)
    gvxr.applyCurrentLocalTransformation(simulation_name) # this works ok when applying the rotation_axis but need to think carefully before applying offsets.
islicer(xray_image_set)

show2D(xray_image_set, slice_list=[(0, 0), (0, 45), (0,90), (0, 135), (0,180)], num_cols=5)
# %%


# %%
# untilted detector
ag = AcquisitionGeometry.create_Parallel3D(ray_direction = beam_direction,
                                      detector_position = np.array(gvxr.getDetectorPosition("mm")),
                                      detector_direction_x = np.array(gvxr.getDetectorRightVector()),
                                      detector_direction_y = np.array(gvxr.getDetectorUpVector()),
                                      rotation_axis_position = gvxr.getCentreOfRotationPositionCT("mm"),
                                      rotation_axis_direction = tilted_rotation_axis)                                 
ag.set_angles(angle_set)
ag.set_panel(gvxr.getDetectorNumberOfPixels())#,
            #  [pixel_size_um/1000, pixel_size_um/1000])


# tilted detector
# ag = AcquisitionGeometry.create_Parallel3D(ray_direction = beam_direction,
#                                       detector_position = np.array(gvxr.getDetectorPosition("mm"))/100,
#                                       detector_direction_x = np.array(gvxr.getDetectorRightVector()),
#                                       detector_direction_y = rotation_axis,
#                                       rotation_axis_position = np.array(gvxr.getCentreOfRotationPositionCT("mm")),
#                                       rotation_axis_direction = rotation_axis)                                 
# ag.set_angles(angle_set)
# ag.set_panel(gvxr.getDetectorNumberOfPixels(),
#              [(pixel_size_um/1000), (pixel_size_um/1000)/np.cos(np.deg2rad(tilt))])
show_geometry(ag)#, elevation=0, azimuthal=-90)
# %%


# ig.voxel_size_y = ag.pixel_size_v
data = AcquisitionData(xray_image_set.astype(np.float32), geometry=ag)
# islicer(data)


# # Apply Beer-Lambert law
data_norm = TransmissionAbsorptionConverter(white_level=1.0)(data)
print(data_norm.min())
print(data_norm.max())
show2D(data_norm, slice_list=[('angle', 0), ('angle', 45), ('angle',90), ('angle', 135), ('angle',180)], num_cols=5)


# # FluxNormaliser
# data_norm.reorder('cil')
# data_norm = FluxNormaliser(flux=data_norm.max(), target=1)(data_norm)
# print(data_norm.min())
# print(data_norm.max())
data_norm.reorder('astra')
ag = data_norm.geometry
ig = ag.get_ImageGeometry()
# ig.voxel_num_z = 1

recon_method = 'astra'
if recon_method == 'cil':
    from cil.recon import FBP
    data_norm.reorder('astra')
    fbp = FBP(data_norm, ig, backend='astra')
    recon = fbp.run()
elif recon_method == 'astra':
    from cil.plugins.astra.processors import FBP
    data_norm.reorder('astra')
    fbp = FBP(ig, ag)
    fbp.set_input(data_norm)
    recon = fbp.get_output()
elif recon_method == 'SIRT':
    from cil.optimisation.algorithms import SIRT
    from cil.plugins.astra.operators import ProjectionOperator
    
    A = ProjectionOperator(ig, ag, 'gpu')
    initial = ig.allocate(0)
    sirt = SIRT(initial, A, data_norm)
    sirt.run(10)
    recon = sirt.solution
# recon.apply_circular_mask(0.9)

rotated_centre = rotation_matrix.apply(offsets)
plot_offsets = np.round(size/2 + rotated_centre).astype(np.int32)
show2D(recon)
# %%
show2D(recon, slice_list = [('horizontal_x', plot_offsets[0]), ('vertical', plot_offsets[1]), ('horizontal_y', plot_offsets[2])], num_cols=3)

# %%
# recon5 = recon.copy()

# # %%
# plt.plot(recon50.array[250, 250, :], label=r'50$^\circ$')
# plt.plot(recon45.array[250, 250, :], label=r'45$^\circ$')
# plt.plot(recon40.array[250, 250, :], label=r'40$^\circ$')
# plt.plot(recon35.array[250, 250, :], label=r'35$^\circ$')
# plt.plot(recon30.array[250, 250, :], label=r'30$^\circ$')
# plt.plot(recon25.array[250, 250, :], label=r'25$^\circ$')
# plt.plot(recon20.array[250, 250, :], label=r'20$^\circ$')
# plt.plot(recon15.array[250, 250, :], label=r'15$^\circ$')
# plt.plot(recon10.array[250, 250, :], label=r'10$^\circ$')
# # plt.plot(recon5.array[250, 250, :], label='5 degrees')
# # plt.plot(recon0.array[250, 250, :], label='0 degrees')
# plt.xlabel('horizontal_x')
# plt.ylabel('Intensity')

# plt.legend(loc='upper left')
# plt.grid()
# # %% Save normalised data with geometry
# file_name = os.path.join(output_path, simulation_name + str(len(angle_set))) #### update the filename here ####
# data_norm.reorder('cil')
# NEXUSDataWriter(data=data_norm, file_name=file_name).write()