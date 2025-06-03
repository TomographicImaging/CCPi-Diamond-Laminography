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
tilt = 30
simulation_name = "test"

# %% Create the experiment geometry
# Set up the source
print("Create an OpenGL context")
gvxr.createOpenGLContext()
print("Set up the beam")
energy = 30
energy_units = "keV"
photons = 10000
gvxr.setSourcePosition(-0.40, 0.0, 0.0, "cm")
gvxr.setMonoChromatic(energy, energy_units, photons)
gvxr.useParallelBeam()

# Set up the detector
print("Set up the detector")
gvxr.setDetectorPosition(0.10, 0.0, 0.0, "cm")
gvxr.setDetectorUpVector(0, 0, -1)
gvxr.setDetectorNumberOfPixels(500, 500)
pixel_size_um = 0.5
gvxr.setDetectorPixelSize(pixel_size_um, pixel_size_um, "um")

# %%
sphere_radius = 40
cube_edge = 60
gvxr.removePolygonMeshesFromSceneGraph()
gvxr.makeSphere(simulation_name, 50, 50, sphere_radius, "um")
gvxr.translateNode(simulation_name, 0, -50, -50, "um")
gvxr.applyCurrentLocalTransformation(simulation_name)

gvxr.makeCube("cube", cube_edge, "um")
gvxr.translateNode("cube", 0, 50, 50, "um")
gvxr.applyCurrentLocalTransformation("cube")

gvxr.addMesh(simulation_name, "cube")

gvxr.addPolygonMeshAsInnerSurface(simulation_name)
gvxr.setCompound(simulation_name, "SiO2")
gvxr.setDensity(simulation_name, 2.2,"g.cm-3")

gvxr.displayScene()
x_ray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
show2D(x_ray_image)

# %%
# Tilt the sample and compute an x-ray image
tilt_axis = np.array([0, 1, 0])
gvxr.rotateNode(simulation_name, tilt, *tilt_axis)
print("Compute an X-ray image")
x_ray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
show2D(x_ray_image)

# %% Check CT rotation axis
rotate_angle = 90
gvxr.rotateNode(simulation_name, rotate_angle, 0, 0, 1)

print("Compute an X-ray image")
x_ray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
show2D(x_ray_image)
gvxr.rotateNode(simulation_name, -rotate_angle, 0, 0, 1)

# %% Simulate a CT scan
start = 0
stop = 360
step = 1
angle_set = np.arange(start, stop, step)
xray_image_set = np.zeros((stop, gvxr.getDetectorNumberOfPixels()[1], gvxr.getDetectorNumberOfPixels()[0]))

for i in angle_set:
    # Rotate
    gvxr.rotateNode(simulation_name, angle_set[i], 0, 0, 1)

    # Compute xray image
    xray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
    xray_image_set[i] = xray_image

    # Reverse rotation
    gvxr.rotateNode(simulation_name, -angle_set[i], 0, 0, 1)

islicer(xray_image_set)

# %%
print("Apply tilt: ", tilt, " degrees, along direction: ", tilt_axis)
rotation_matrix = R.from_rotvec(np.radians(tilt) * tilt_axis)
orthogonal_axis = -np.array(gvxr.getDetectorUpVector())
print("Untilted rotation axis: ", orthogonal_axis)
rotation_axis = rotation_matrix.apply(orthogonal_axis)
print("Tilted rotation axis: ", rotation_axis)

# # %% Compare with computeCTAcquisition function
gvxr.computeCTAcquisition('',
                          '',
                          360, # The total number of projections to simulate.
                          0, # The rotation angle corresponding to the first projection.
                          False, # A boolean flag to include or exclude the last angle. It is used to calculate the angular step between successive projections.
                          360, # The rotation angle corresponding to the last projection. Note that depending on the value of anIncludeLastAngleFlag, this angle may be included or excluded
                          1, # The number of white images used to perform the flat-field correction. If zero, then no correction will be performed.
                          *gvxr.getCentreOfRotationPositionCT("mm"), # The location of the rotation centre.
                          "mm", # The corresponding unit of length.
                          *rotation_axis, # The rotation axis
                          True # If true the energy fluence is returned, otherwise the number of photons is returned
                               # (default value: true)
)

xray_image_set2 = np.array(gvxr.getLastProjectionSet())

# %%
beam_direction = np.array(gvxr.getDetectorPosition("cm"))/np.linalg.norm(np.array(gvxr.getDetectorPosition("cm")))
ag = AcquisitionGeometry.create_Parallel3D(ray_direction = beam_direction,
                                      detector_position = np.array(gvxr.getDetectorPosition("cm")),
                                      detector_direction_x = np.array(gvxr.getDetectorRightVector()),
                                      detector_direction_y = -np.array(gvxr.getDetectorUpVector()),
                                      rotation_axis_position = gvxr.getCentreOfRotationPositionCT("cm"),
                                      rotation_axis_direction = np.array(gvxr.getDetectorUpVector()))                                 
ag.set_angles(angle_set)
ag.set_panel(gvxr.getDetectorNumberOfPixels(),
             [pixel_size_um/10000,  pixel_size_um/10000])
show_geometry(ag, elevation=0, azimuthal=-90)


data = AcquisitionData(xray_image_set2, geometry=ag)
# %%
show2D(data, slice_list=[('angle', 0), ('angle', 45), ('angle',90), ('angle', 135), ('angle',180)], num_cols=5)
# %%
# islicer(data)


# Apply Beer-Lambert law
data_norm = TransmissionAbsorptionConverter(white_level=1.)(data)
# print(data_norm.min())
# print(data_norm.max())


data_norm.reorder('astra')
ig = data_norm.geometry.get_ImageGeometry()
fbp = FBP(ig, data_norm.geometry)
recon = fbp(data_norm)
recon.apply_circular_mask(0.9)

show2D(recon, slice_list=[('vertical', 200), ('horizontal_x', 200), ('horizontal_y', 250) ], num_cols=3)

# %%