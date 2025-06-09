# %%
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import rotate


import matplotlib.pyplot as plt
from gvxrPython3 import gvxr
from gvxrPython3.JSON2gVXRDataReader import *

from cil.plugins.astra.processors import FBP
from cil.processors import TransmissionAbsorptionConverter
from cil.utilities.display import show_geometry, show2D, show1D
from cil.utilities.jupyter import islicer
from cil.processors import Normaliser
from cil.framework import AcquisitionData, AcquisitionGeometry
from cil.framework.labels import AngleUnit
from cil.io import NEXUSDataWriter

from cil.plugins.tigre import CIL2TIGREGeometry
import tigre

from cil.framework import DataContainer, ImageData, ImageGeometry, AcquisitionGeometry
from cil.plugins.tigre import ProjectionOperator
# %%
lines = np.zeros((6, 16,16), dtype=np.float32)
lines[2:4,2:14,2:6] = 1
lines[2:4,2:14,10:14] = 1
# lines = np.zeros((2, 16,16), dtype=np.float32)
# lines[:,:,4:8] = 1
# lines[:,:,12:16] = 1
# lines = np.expand_dims(lines,2)
# show2D(lines)
# lines3D = np.expand_dims(lines, 0).transpose()


ig = ImageGeometry(lines.shape[1],lines.shape[2],lines.shape[0])
lines = ImageData(lines, geometry=ig)
# lines = DataContainer(lines, dimension_labels=['X', 'Y', 'Z'])
show2D(lines,
    #    ['X', 'Y', 'Z'], 
       slice_list=[(0, int(lines.shape[0]/2)), (1, int(lines.shape[1]/2)), (2, 3)], 
       num_cols=3,
       origin='lower-left')
# %%
tilt = 30 # degrees
tilt_rad = np.deg2rad(tilt)
tilt_direction = np.array([1, 0, 0])
beam_direction = np.array([0, 1, 0])
untilted_rotation_axis = np.array([0, 0, 1])
rotation_matrix = R.from_rotvec(tilt_rad * tilt_direction)
tilted_rotation_axis = rotation_matrix.apply(untilted_rotation_axis)

# ag = AcquisitionGeometry.create_Parallel3D(rotation_axis_direction=untilted_rotation_axis)\
#     .set_angles([0,90])\
#     .set_panel(lines.shape[1:3])

ag = AcquisitionGeometry.create_Parallel3D(rotation_axis_direction=untilted_rotation_axis)\
    .set_angles(np.arange(0,360,1))\
    .set_panel(lines.shape[1:3])

geo, angles = CIL2TIGREGeometry.getTIGREGeometry(ig, ag)


# %%
show_geometry(ag, ig)
# %%
# A = ProjectionOperator(ig, ag)
# proj = A.direct(lines)
# show2D(proj, slice_list=[(0,0),(0,1),(0,2)])
# %%

geo, angles = CIL2TIGREGeometry.getTIGREGeometry(ig, ag)
euler_angles = []
for angle in angles:
    R1 = R.from_euler("z", angle, degrees=False)
    combined = rotation_matrix * R1
    euler = combined.as_euler("ZYZ", degrees=False)
    euler_angles.append(euler)

euler_angles = np.array(euler_angles) 
out = tigre.Ax(lines.array.astype(np.float32), geo, euler_angles, "interpolated")
show2D([out[0], out[90], out[180]], 
       ['0', r'$\pi/2$', r'$\pi$'],
       num_cols=3)
# show2D([out[0], out[1]], 
#        ['0', r'$\pi$'],
#        num_cols=2)
# %%
weights = np.zeros_like(angles)
weighted_projections = np.zeros_like(out)
for i, theta in enumerate(np.deg2rad(ag.angles)):
       # weights[i] = np.sqrt(1 - (np.sin(tilt_rad) * np.cos(theta))**2)
       Rz = R.from_rotvec(theta * untilted_rotation_axis)
       p_theta = Rz.apply(tilt_direction)
       weights[i] = 1/np.sqrt(1 - np.dot(tilted_rotation_axis, p_theta)**2)

       # v = Rz.apply(det_right)
       # weight = np.abs(u[0])
       weighted_projections[i,:,:] = out[i,:,:]*weights[i]
plt.plot(ag.angles,weights)
plt.xlabel('Angle (degrees)')
plt.ylabel('Weight')
plt.grid()
# %%

vol = tigre.Atb(out, geo, euler_angles)
show2D(vol,   
       slice_list=[(0, int(vol.shape[0]/2)), (1, int(vol.shape[1]/2)), (2, 4)], 
       num_cols=3)
# %%
weighted_vol = tigre.Atb(weighted_projections, geo, euler_angles)
show2D(weighted_vol,   
       slice_list=[(0, int(vol.shape[0]/2)), (1, int(vol.shape[1]/2)), (2, 4)], 
       num_cols=3)
show2D(vol-weighted_vol,   
       slice_list=[(0, int(vol.shape[0]/2)), (1, int(vol.shape[1]/2)), (2, int(vol.shape[2]/2))], 
       num_cols=3, cmap='RdBu_r', fix_range=(-150, 150))
# %%
vol = tigre.algorithms.fbp(out, geo, euler_angles)
show2D(vol,
       slice_list=[(0, int(vol.shape[0]/2)), (1, int(vol.shape[1]/2)), (2,4)], 
       num_cols=3)
# %%
weighted_vol = tigre.algorithms.fbp(weighted_projections, geo, euler_angles)
show2D(weighted_vol,   
       slice_list=[(0, int(vol.shape[0]/2)), (1, int(vol.shape[1]/2)), (2, 4)], 
       num_cols=3)
show2D(vol-weighted_vol,   
       slice_list=[(0, int(vol.shape[0]/2)), (1, int(vol.shape[1]/2)), (2, int(vol.shape[2]/2))], 
       num_cols=3, cmap='RdBu_r', fix_range=(-0.06, 0.06))
# %%
vol = tigre.algorithms.sirt(out, geo, euler_angles, 200)
show2D(vol,
       slice_list=[(0, int(vol.shape[0]/2)), (1, int(vol.shape[1]/2)), (2, 4)], 
       num_cols=3)
# %%
# weighted_vol = tigre.algorithms.sirt(weighted_projections, geo, euler_angles, 200)
# show2D(weighted_vol,   
#        slice_list=[(0, int(vol.shape[0]/2)), (1, int(vol.shape[1]/2)), (2, 4)], 
#        num_cols=3)
show2D(vol-weighted_vol,   
       slice_list=[(0, int(vol.shape[0]/2)), (1, int(vol.shape[1]/2)), (2, int(vol.shape[2]/2))], 
       num_cols=3, cmap='RdBu_r', fix_range=(-2, 2))
# %% Not correct for tilted

ad = AcquisitionData(out, geometry=ag)
# ad.reorder('tigre')
# ag = ad.geometry
# ig = ag.get_ImageGeometry()
from cil.plugins.tigre import FBP
vol = FBP(ig, ag)(ad)
show2D(vol,
       slice_list=[(0, int(vol.shape[0]/2)), (1, int(vol.shape[1]/2)), (2, 4)], 
       num_cols=3)

# %%
geo = tigre.geometry()
geo.DSO = 5
geo.DSD = 5
geo.nDetector = np.array([16, 16])
geo.dDetector = np.array([1, 1]) 
geo.sDetector = geo.nDetector * geo.dDetector
geo.nVoxel = np.array(lines.shape)
geo.sVoxel = np.array(lines.shape)
geo.dVoxel = geo.sVoxel/geo.nVoxel
geo.mode = "parallel"
geo.accuracy = 0.5

out = tigre.Ax(lines.array.astype(np.float32), geo, np.array([0, np.pi/2, np.pi]))
show2D(out, ['0', 'pi/2', 'pi'], 
       slice_list = [(0,0),(0,1),(0,2)],
       num_cols=3)


print(geo)







# %%
simulation_name = "test"
tilt = 30 # degrees
tilt_direction = np.array([1, 0, 0])
beam_direction = np.array([0, 1, 0])
untilted_rotation_axis = np.array([0, 0, 1])

print("Apply tilt: ", tilt, " degrees, along direction: ", tilt_direction)
rotation_matrix = R.from_rotvec(np.radians(tilt) * tilt_direction)
print("Untilted rotation axis: ", untilted_rotation_axis)
tilted_rotation_axis = rotation_matrix.apply(untilted_rotation_axis)
print("Tilted rotation axis: ", tilted_rotation_axis)

rotation_matrix = R.from_rotvec(np.radians(tilt) * tilt_direction)

tilt_the_sample = True # physically rotate the sample, different from if we tilt the rotation axis later
cube_half_edge = 60/3
size = 100
cube_offset = np.array([-0, -0, 0])

centre = (size/2)

# %% Create the experiment geometry
# Set up the source
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
gvxr.setDetectorUpVector(*untilted_rotation_axis)
gvxr.setDetectorNumberOfPixels(size, size)
pixel_size_um = 1
gvxr.setDetectorPixelSize(pixel_size_um, pixel_size_um, "um")

# %%
gvxr.removePolygonMeshesFromSceneGraph()


# gvxr.makeCube(simulation_name, cube_half_edge*2, "um")
gvxr.makeCuboid(simulation_name, cube_half_edge*3, cube_half_edge*3, cube_half_edge*3, "um")
x = [-cube_half_edge, cube_half_edge, 0, -cube_half_edge, cube_half_edge]
y = [-cube_half_edge, -cube_half_edge, 0, cube_half_edge, cube_half_edge]
for i in np.arange(len(x)):
    gvxr.makeCuboid("small", cube_half_edge, cube_half_edge, cube_half_edge*3, "um")
    gvxr.translateNode("small", x[i], y[i], 0, "um")
    gvxr.applyCurrentLocalTransformation("small")
    gvxr.addMesh(simulation_name, "small")

# gvxr.applyCurrentLocalTransformation(simulation_name)
gvxr.addPolygonMeshAsInnerSurface(simulation_name)

rotated_cube_center = rotation_matrix.apply(cube_offset)
cube_plot_offsets = np.round(centre + rotated_cube_center).astype(np.int32)
    
gvxr.setCompound(simulation_name, "SiO2")
gvxr.setDensity(simulation_name, 2.2,"g.cm-3")


# gvxr.setCompound("small", "SiO2")
# gvxr.setDensity("small", 1.2,"g.cm-3")

if tilt_the_sample:
    gvxr.rotateNode(simulation_name, tilt, *tilt_direction)
    gvxr.applyCurrentLocalTransformation(simulation_name)
gvxr.displayScene()
x_ray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
show2D(x_ray_image)

# %% Simulate a CT scan
start = 0
stop = 360
step = 1
angle_set = np.arange(start, stop, step)
xray_image_set = np.zeros((stop, gvxr.getDetectorNumberOfPixels()[1], gvxr.getDetectorNumberOfPixels()[0]))

for i in angle_set:
    # Rotate
    
    gvxr.rotateNode(simulation_name, angle_set[i], *tilted_rotation_axis)
    # gvxr.applyCurrentLocalTransformation(simulation_name)

    # Compute xray image
    xray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
    xray_image_set[i] = xray_image

    # Reverse rotation
    gvxr.rotateNode(simulation_name, -angle_set[i], *tilted_rotation_axis)
    # gvxr.applyCurrentLocalTransformation(simulation_name)

islicer(xray_image_set)

show2D(xray_image_set, slice_list=[(0, 0), (0, 45), (0,90), (0, 135), (0,180)], num_cols=5)
# %%
# rotation_axis = untilted_rotation_axis
# %%
ag = AcquisitionGeometry.create_Parallel3D(ray_direction = beam_direction,
                                      detector_position = gvxr.getDetectorPosition("mm"),
                                      detector_direction_x = gvxr.getDetectorRightVector(),
                                      detector_direction_y = gvxr.getDetectorUpVector(),
                                      rotation_axis_position = gvxr.getCentreOfRotationPositionCT("mm"),
                                      rotation_axis_direction = tilted_rotation_axis)                                 
ag.set_angles(angle_set)
ag.set_panel(gvxr.getDetectorNumberOfPixels(),
             [pixel_size_um/1000, pixel_size_um/1000])
show_geometry(ag)
# %%
data = AcquisitionData(xray_image_set.astype(np.float32), geometry=ag)
# islicer(data)
show2D(data, slice_list=[('angle', 0), ('angle', 45), ('angle',90), ('angle', 135), ('angle',180)], num_cols=5)
# %%
# Apply Beer-Lambert law
data_norm = TransmissionAbsorptionConverter(white_level=1.)(data)
# Recon
data_norm.reorder('astra')
ag = data_norm.geometry
ig = ag.get_ImageGeometry()
fbp = FBP(ig, ag)
recon = fbp(data_norm)

show2D(recon, slice_list = [('horizontal_x', cube_plot_offsets[0]), ('vertical', cube_plot_offsets[1]), ('horizontal_y', cube_plot_offsets[2])], num_cols=3)
# %%
