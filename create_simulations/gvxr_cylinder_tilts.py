# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import sys, os
from gvxrPython3 import gvxr
from gvxrPython3.twins.utils import createDigitalTwin
from gvxrPython3.JSON2gVXRDataReader import *
from cil.framework import ImageData, ImageGeometry, AcquisitionData, AcquisitionGeometry
from cil.utilities.display import show2D, show_geometry
from cil.utilities.jupyter import islicer
from cil.plugins.astra.processors import FBP
from cil.processors import TransmissionAbsorptionConverter
from cil.io import NEXUSDataWriter


# %%

def create_cylinder_with_spheres(simulation_name='cylinder', cylinder_radius = 100, plane = 'xy'):
    
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
# %%
output_path = "../output_data"
# %%
# create a digital twin simulation and initialise with some experimental parameters
gvxr.createOpenGLContext(0,
                         4, 6,
                         32)  # 0 for mixed-precision (good compromise between speed and accuracy),
                              # 16 for half-precision (the fastest but maybe not that accurate),

diad = createDigitalTwin(name="DIAD")
diad.beam.kev = 25
diad.detector.exposure = 10.0

for resolution in diad.specification.detector.resolutions:
    resolution[0] = round(resolution[0] / 8)
    resolution[1] = round(resolution[1] / 8)

diad.specification.detector.pixel_pitch *= 8

diad.apply()
# %%
simulation_name = "cylinder"
create_cylinder_with_spheres(simulation_name=simulation_name, cylinder_radius=500, plane='xy')
# Compute an X-ray image
gvxr.displayScene()
xray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
show2D(xray_image)
# %%

# specify number of projections
start = 0
stop = 360
step = 1
angle_set = np.arange(start, stop, step)
pixels_x = xray_image.shape[0]
pixels_y = xray_image.shape[1]
xray_image_set = np.zeros((stop, pixels_x, pixels_y), dtype=np.float32)

# specify the rotation axis, around y
rotation_axis = np.array([0, 1, 0])
for N in angle_set:
    # Rotate
    gvxr.rotateNode(simulation_name, N, *rotation_axis)
    # Compute xray image
    xray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
    xray_image_set[N] = xray_image
    # Rotate back to origin
    gvxr.rotateNode(simulation_name, -N, *rotation_axis)

# use the islicer tool to scroll through the projections
islicer(xray_image_set)
# %%
beam_direction = np.array([0, 1, 0])
detector_direction_x = np.array([1, 0, 0])
detector_direction_y = np.array([0, 0, 1])
rotation_axis = np.array([0, 0, 1])

ag = AcquisitionGeometry.create_Parallel3D(ray_direction = beam_direction,
                                      detector_direction_x = detector_direction_x,
                                      detector_direction_y = detector_direction_y,
                                      rotation_axis_direction = rotation_axis)              
ag.set_angles(angle_set)
ag.set_panel((pixels_y, pixels_x))

show_geometry(ag)

data = AcquisitionData(xray_image_set, geometry=ag)
data.reorder('astra')
# %%
data = TransmissionAbsorptionConverter(white_level=1.0)(data)

# %%
show2D(data, slice_list=[('angle', 0), ('angle', 45), ('angle',90), ('angle', 135), ('angle',180)], num_cols=5, fix_range=True)
# %%
file_name = os.path.join(output_path, simulation_name + str(len(angle_set))) #### update the filename here ####
print(file_name)
NEXUSDataWriter(data=data, file_name=file_name).write()
# %%
# Create the tilted simulation
simulation_name = "cylinder_tilt_30"
create_cylinder_with_spheres(simulation_name=simulation_name, cylinder_radius=500, plane='yz')
# Compute an X-ray image
gvxr.displayScene()
xray_image1 = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()

# tilt the sample
tilt = 30 # degrees
tilt_axis = np.array([1, 0, 0]) # around the detector x direction

gvxr.rotateNode(simulation_name, tilt, *tilt_axis)
xray_image2 = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
show2D([xray_image1, xray_image2], ['Flat sample', 'Tilted sample'])
# %%
# specify number of projections
start = 0
stop = 360
step = 1
angle_set = np.arange(start, stop, step)
xray_image_set = np.zeros((stop, pixels_x, pixels_y), dtype=np.float32)

# specify the rotation axis, around y
rotation_axis = np.array([0, 1, 0])
for N in angle_set:
    # Rotate
    gvxr.rotateNode(simulation_name, N, *rotation_axis)
    # Compute xray image
    xray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
    xray_image_set[N] = xray_image
    # Rotate back to origin
    gvxr.rotateNode(simulation_name, -N, *rotation_axis)

# use the islicer tool to scroll through the projections
islicer(xray_image_set)
# %%
beam_direction = np.array([0, 1, 0])
detector_x_direction = np.array([1, 0, 0])
detector_y_direction = np.array([0, 0, -1])
rotation_axis = np.array([0, 0, 1]) # the untilted rotation axis

# create the tilted rotation axis
tilt_rad = np.deg2rad(tilt)
rotation_matrix = Rotation.from_rotvec(tilt_rad * detector_x_direction)
tilted_rotation_axis = rotation_matrix.apply(rotation_axis)

ag = AcquisitionGeometry.create_Parallel3D(ray_direction = beam_direction,
                                      detector_direction_x = np.array([1, 0, 0]),
                                      detector_direction_y = np.array([0, 0, -1]),
                                      rotation_axis_direction = list(tilted_rotation_axis))                   
ag.set_angles(angle_set)
ag.set_panel((pixels_y, pixels_x),
             (0.5/1000, 0.5/1000))

show_geometry(ag)

data = AcquisitionData(xray_image_set, geometry=ag)
data.reorder('astra')
# %%
# apply Beer-Lambert law
data = TransmissionAbsorptionConverter(white_level=1.0)(data)
show2D(data, slice_list=[('angle', 0), ('angle', 45), ('angle',90), ('angle', 135), ('angle',180)], num_cols=5)
# %%
file_name = os.path.join(output_path, simulation_name ) #### update the filename here ####
print(file_name)
NEXUSDataWriter(data=data, file_name=file_name).write()
# %%
# Create the simulation
simulation_name = "cylinder_roi_tilt_30"
create_cylinder_with_spheres(simulation_name=simulation_name, cylinder_radius=1000, plane='yz')
# Compute an X-ray image
gvxr.displayScene()
xray_image1 = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()

# tilt the sample
tilt = 30 # degrees
tilt_axis = np.array([1, 0, 0]) # around the detector x direction

gvxr.rotateNode(simulation_name, tilt, *tilt_axis)
xray_image2 = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
show2D([xray_image1, xray_image2], ['Flat sample', 'Tilted sample'])
# %%
# specify number of projections
start = 0
stop = 360
step = 1
angle_set = np.arange(start, stop, step)
xray_image_set = np.zeros((stop, pixels_x, pixels_y), dtype=np.float32)

# specify the rotation axis, around z
rotation_axis = np.array([0, 1, 0])
for N in angle_set:
    # Rotate
    gvxr.rotateNode(simulation_name, N, *rotation_axis)
    # Compute xray image
    xray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
    xray_image_set[N] = xray_image
    # Rotate back to origin
    gvxr.rotateNode(simulation_name, -N, *rotation_axis)

# create the geometry
beam_direction = np.array([0, 1, 0])
detector_x_direction = np.array([1, 0, 0])
detector_y_direction = np.array([0, 0, -1])
rotation_axis = np.array([0, 0, 1]) # the untilted rotation axis

tilt_rad = np.deg2rad(tilt)
rotation_matrix = Rotation.from_rotvec(tilt_rad * detector_x_direction)
tilted_rotation_axis = rotation_matrix.apply(rotation_axis)

ag = AcquisitionGeometry.create_Parallel3D(ray_direction = beam_direction,
                                      detector_direction_x = np.array([1, 0, 0]),
                                      detector_direction_y = np.array([0, 0, -1]),
                                      rotation_axis_direction = list(tilted_rotation_axis))                   
ag.set_angles(angle_set)
ag.set_panel((pixels_y, pixels_x),
             (0.5, 0.5))

data = AcquisitionData(xray_image_set, geometry=ag,)
data.reorder('astra')
data = TransmissionAbsorptionConverter(white_level=1.0)(data)
show2D(data, slice_list=[('angle', 0), ('angle', 45), ('angle',90), ('angle', 135), ('angle',180)], num_cols=5)
# %%
file_name = os.path.join(output_path, simulation_name ) #### update the filename here ####
print(file_name)
NEXUSDataWriter(data=data, file_name=file_name).write()
# %%
