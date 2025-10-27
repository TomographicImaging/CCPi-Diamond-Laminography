# %%
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

import matplotlib.pyplot as plt
from gvxrPython3 import gvxr
from gvxrPython3.twins.utils import createDigitalTwin
from gvxrPython3.JSON2gVXRDataReader import *

from cil.processors import TransmissionAbsorptionConverter
from cil.utilities.display import show_geometry, show2D
from cil.utilities.jupyter import islicer

from cil.processors import FluxNormaliser, Normaliser
from cil.framework import AcquisitionData, AcquisitionGeometry
from cil.plugins.astra.processors import FBP
from cil.io import NEXUSDataWriter
# %%
tilt = 30
output_path = "../output_data"
simulation_name = "cylinder_fullsize"
if not os.path.exists(output_path):
    os.makedirs(output_path)
# %%
gvxr.createOpenGLContext(0,
                             4, 6,
                             32)  # 0 for mixed-precision (good compromise between speed and accuracy),
                                                    # 16 for half-precision (the fastest but maybe not that accurate),



# %%

diad = createDigitalTwin(name="DIAD")
diad.beam.kev = 25
diad.detector.exposure = 10.0

g_downsampling_factor = 2
if g_downsampling_factor > 1:

    for resolution in diad.specification.detector.resolutions:
        resolution[0] = round(resolution[0] / g_downsampling_factor)
        resolution[1] = round(resolution[1] / g_downsampling_factor)

    diad.specification.detector.pixel_pitch *= g_downsampling_factor

diad.apply()

# %%
gvxr.removePolygonMeshesFromSceneGraph()
gvxr.enablePoissonNoise()
cylinder_radius =  500
sphere_radius =  50
gvxr.makeCylinder(simulation_name, 50, sphere_radius*2, cylinder_radius, "um")
gvxr.setNodeTransformationMatrix(simulation_name, [[1, 0, 0, 0], 
                                                   [0, 1, 0, 0], 
                                                   [0, 0, 1, 0], 
                                                   [0, 0, 0, 1]])
# gvxr.rotateNode(simulation_name, 90, 1, 0, 0)

sphere_spacing = 2 * sphere_radius + (sphere_radius/2)

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
tilt_axis = np.array([1, 0, 0])
gvxr.rotateNode(simulation_name, tilt, *tilt_axis)
print("Compute an X-ray image")
x_ray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
show2D(x_ray_image)

# %% Check CT rotation axis
# gvxr.rotateNode(simulation_name, -40, 0, 1, 0)

# print("Compute an X-ray image")
# x_ray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
# show2D(x_ray_image)

# %% Simulate a CT scan
start = 0
stop = 360
step = 360/int((np.pi/2)*gvxr.getDetectorNumberOfPixels()[0])
angle_set = np.arange(start, stop, step)
data = np.zeros((len(angle_set), gvxr.getDetectorNumberOfPixels()[1], gvxr.getDetectorNumberOfPixels()[0]))
t0 = time.time()
for i, angle in enumerate(angle_set):
    # Rotate
    gvxr.rotateNode(simulation_name, step, 0, 1, 0)
    # Compute xray image
    xray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
    data[i] = xray_image
print((time.time()-t0)/60)
# islicer(data)

# %%

detector_direction_x = np.array([1, 0, 0])
detector_direction_y = np.array([0, 0, -1])
rotation_axis = np.array([0, 0, 1])

tilt_rad = np.deg2rad(tilt)
rotation_matrix = R.from_rotvec(tilt_rad * detector_direction_x)
tilted_rotation_axis = rotation_matrix.apply(rotation_axis)

ag = AcquisitionGeometry.create_Parallel3D(detector_direction_x = detector_direction_x,
                                      detector_direction_y = detector_direction_y,
                                      rotation_axis_direction = tilted_rotation_axis)              
ag.set_angles(angle_set)
ag.set_panel(list(gvxr.getDetectorNumberOfPixels()),
             list([gvxr.getDetectorPixelSpacing("mm")[1], gvxr.getDetectorPixelSpacing("mm")[0]]))
show_geometry(ag)
# %%
gvxr.destroy()
del x_ray_image
del diad
# %%
data = AcquisitionData(data, deep_copy=False, geometry=ag)

# islicer(data)
# %%
# Apply Beer-Lambert law
data = TransmissionAbsorptionConverter(white_level=1.0)(data)

# show2D(data, slice_list=[('angle', 0), ('angle', 45), ('angle',90), ('angle', 135), ('angle',180)], num_cols=5)

# %%
data.reorder('astra')
t0 = time.time()
fbp = FBP(data.geometry.get_ImageGeometry(), data.geometry)
recon = fbp(data)
print(time.time() - t0)
# recon.apply_circular_mask(0.9)
# %%
slice_list = [('vertical','centre'), ('horizontal_y',int(recon.shape[2]/2)), ('horizontal_x',int(recon.shape[1]/2))]
show2D(recon,
       slice_list=slice_list,
       num_cols=3)

# %% Save normalised data with geometry
file_name = os.path.join(output_path, simulation_name + str(len(angle_set))) #### update the filename here ####
print(file_name)
data.reorder('cil')
NEXUSDataWriter(data=data, file_name=file_name).write()
