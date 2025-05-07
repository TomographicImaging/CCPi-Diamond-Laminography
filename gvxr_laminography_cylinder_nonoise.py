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
output_path = "output_data"
simulation_name = "cylinder_nonoise"
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
# %% Create the experiment geometry
# Set up the source
print("Create an OpenGL context")
gvxr.createOpenGLContext()
print("Set up the beam")
energy = 30
energy_units = "keV"
photons = 10000
gvxr.setSourcePosition(-40.0, 0.0, 0.0, "cm")
gvxr.setMonoChromatic(energy, energy_units, photons)
gvxr.useParallelBeam()

# Set up the detector
print("Set up the detector")
gvxr.setDetectorPosition(10.0, 0.0, 0.0, "cm")
gvxr.setDetectorUpVector(0, 0, -1)
gvxr.setDetectorNumberOfPixels(500, 500)
pixel_size_um = 0.5
gvxr.setDetectorPixelSize(pixel_size_um, pixel_size_um, "um")

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
# gvxr.rotateNode(simulation_name, -20, 0, 1, 0)

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
                                      detector_position = gvxr.getDetectorPosition("mm"),
                                      detector_direction_x = gvxr.getDetectorRightVector(),
                                      detector_direction_y = gvxr.getDetectorUpVector(),
                                      rotation_axis_position = gvxr.getCentreOfRotationPositionCT("mm"),
                                      rotation_axis_direction = rotation_axis)                                 
ag.set_angles(angle_set)
ag.set_panel(gvxr.getDetectorNumberOfPixels(),
             [pixel_size_um/1000, pixel_size_um/1000])
show_geometry(ag)
# %%
data = AcquisitionData(xray_image_set, geometry=ag)
islicer(data)
# %%

# Apply Beer-Lambert law
data_norm = TransmissionAbsorptionConverter(white_level=1.)(data)
print(data_norm.min())
print(data_norm.max())

# FluxNormaliser
data_norm.reorder('cil')
data_norm = FluxNormaliser(flux=data_norm.max(), target=1)(data_norm)
print(data_norm.min())
print(data_norm.max())

data_norm.reorder('astra')
fbp = FBP(data_norm.geometry.get_ImageGeometry(), data_norm.geometry)
recon = fbp(data_norm)
recon.apply_circular_mask(0.9)

show2D([recon]) 
# %% Save normalised data with geometry
file_name = os.path.join(output_path, simulation_name + str(len(angle_set))) #### update the filename here ####
data_norm.reorder('cil')
NEXUSDataWriter(data=data_norm, file_name=file_name).write()