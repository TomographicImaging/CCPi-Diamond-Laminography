# Test creating a single sphere or cube with gvxr
# Look at cross sections for different tilt angles
# Try lots of different reconstruction methods from the literature
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
from cil.framework.labels import AngleUnit
from cil.processors import FluxNormaliser, Normaliser
from cil.framework import AcquisitionData, AcquisitionGeometry
from cil.io import NEXUSDataWriter

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
include_cube = True
include_sphere = False

sphere_radius = 80/5
cube_half_edge = 60/5
size = 100

sphere_offset = np.array([100, 100, 0])
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

if include_sphere:
    gvxr.makeSphere(simulation_name, 50, 50, sphere_radius, "um")
    gvxr.translateNode(simulation_name, *sphere_offset, "um")
    # gvxr.applyCurrentLocalTransformation(simulation_name)
    gvxr.addPolygonMeshAsInnerSurface(simulation_name)

    rotated_sphere_center = rotation_matrix.apply(sphere_offset)
    sphere_plot_offsets = np.round(centre + rotated_sphere_center).astype(np.int32)

if include_cube:
    # gvxr.makeCube(simulation_name, cube_half_edge*2, "um")
    gvxr.makeCuboid(simulation_name, cube_half_edge*5, cube_half_edge*5,cube_half_edge, "um")
    gvxr.translateNode(simulation_name, *cube_offset, "um")
    # gvxr.applyCurrentLocalTransformation(simulation_name)
    gvxr.addPolygonMeshAsInnerSurface(simulation_name)

    rotated_cube_center = rotation_matrix.apply(cube_offset)
    cube_plot_offsets = np.round(centre + rotated_cube_center).astype(np.int32)
    
    # gvxr.addMesh(simulation_name, sphere_name)

gvxr.setCompound(simulation_name, "SiO2")
gvxr.setDensity(simulation_name, 2.2,"g.cm-3")

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

weight_proj = True
# Weighting
if weight_proj:
    data_norm.reorder('tigre')
    angles = np.deg2rad(angle_set)
    tilt_rad = np.deg2rad(tilt)

    Rx = R.from_rotvec(tilt_rad * np.array([1, 0, 0]))
    det_up = Rx.apply(ag.config.system.detector.direction_y)
    det_right = Rx.apply(ag.config.system.detector.direction_x)

    weights = np.zeros_like(angles)
    for i, theta in enumerate(angles):
        weights[i] = 1/np.sqrt(1 - (np.sin(tilt_rad) * np.cos(theta))**2)
        Rz = R.from_rotvec(theta * np.array([0, 0, 1]))
        p_theta = Rz.apply(np.array([1, 0, 0]))

        weights[i] = np.sqrt(1 - np.dot(tilted_rotation_axis, p_theta)**2)

        # v = Rz.apply(det_right)
        # weight = np.abs(u[0])
        data_norm.array[i,:,:] *= weights[i]

# Recon
data_norm.reorder('astra')
ag = data_norm.geometry
ig = ag.get_ImageGeometry()
fbp = FBP(ig, ag)
if weight_proj:
    recon_weighted = fbp(data_norm)
else:
    recon = fbp(data_norm)
# recon.apply_circular_mask(0.9)

if include_cube:
    show2D(recon, slice_list = [('horizontal_x', cube_plot_offsets[0]), ('vertical', cube_plot_offsets[1]), ('horizontal_y', cube_plot_offsets[2])], num_cols=3)
    # show2D(recon_weighted, slice_list = [('horizontal_x', cube_plot_offsets[0]), ('vertical', cube_plot_offsets[1]), ('horizontal_y', cube_plot_offsets[2])], num_cols=3)
    # show2D(recon-recon_weighted, slice_list = [('horizontal_x', cube_plot_offsets[0]), ('vertical', cube_plot_offsets[1]), ('horizontal_y', cube_plot_offsets[2])], num_cols=3)

if include_sphere:
    show2D(recon, slice_list = [('horizontal_x', sphere_plot_offsets[0]), ('vertical', sphere_plot_offsets[1]), ('horizontal_y', sphere_plot_offsets[2])], num_cols=3)
# %%
plt.plot(angle_set, weights)
plt.xlabel('Angle (degrees)')
plt.ylabel(r'Weight $\frac{1}{|a_{\perp}|}$')
plt.grid()
# recon30 = recon.copy()
# %%
# plt.plot(recon.array[int(size/2), int(size/2), :], label='Unweighted')
plt.plot(recon_weighted.array[int(size/2), int(size/2), :], label='Weighted')
plt.xlabel('Horizontal x index')
plt.ylabel('Value')
# plt.plot(recon30.array[int(size/2), int(size/2), :], label='30 degrees')
# plt.plot(recon25.array[int(size/2), int(size/2), :], label='25 degrees')
# plt.plot(recon20.array[int(size/2), int(size/2), :], label='20 degrees')
# plt.plot(recon15.array[int(size/2), int(size/2), :], label='15 degrees')
# plt.plot(recon10.array[int(size/2), int(size/2), :], label='10 degrees')
# plt.plot(recon5.array[int(size/2), int(size/2), :], label='5 degrees')
# plt.plot(recon0.array[int(size/2), int(size/2), :], label='0 degrees')
plt.legend()
plt.grid()
# %%
# import astra
# vol_geom = astra.create_vol_geom([size, size, size])
import tigre
import tigre.algorithms as algs
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import map_coordinates
from cil.plugins.tigre import ProjectionOperator
from cil.framework import ImageGeometry


data_norm.reorder('tigre')
# %% Tigre attempts using tilted rotation axis method from tigre docs
tilt = 30
geo = tigre.geometry()
geo.DSD = 5
geo.DSO = 5
geo.nDetector = np.array([size, size])   # number of pixels                (px)
geo.dDetector = np.array([1, 1])       # size of each pixel              (mm)
# geo.dDetector = np.array([1, 1/np.cos(np.deg2rad(tilt))]) 
geo.sDetector = geo.nDetector*geo.dDetector # total size of the detector (mm)
geo.nVoxel = np.array([size, size, size]) # number of voxels                (vx)
geo.sVoxel = np.array([size, size, size]) # total size of the image         (mm)
# [(pixel_size_um/1000), (pixel_size_um/1000)/np.cos(np.deg2rad(tilt))]
geo.dVoxel = geo.sVoxel / geo.nVoxel   # size of each voxel              (mm)
geo.offOrigin = np.array([0, 0, 0])    # Offset of image from origin     (mm)

geo.offDetector = np.array([0, 0, 0])  # Offset of Detector              (mm)

geo.accuracy = 0.5
geo.mode = "parallel"  # Or 'parallel'. Geometry type.
geo.COR = 0  # y direction displacement for COR, this can also be defined per angle (mm)

# geo.rotDetector = np.array([1, 0, 0])
# geo.rotDetector = np.array([0, np.deg2rad(tilt), 0])

tilt_direction = np.array([1, 0, 0])
untilted_rotation_axis = np.array([0, 0, 1])
rotation_matrix = R.from_rotvec(np.radians(tilt) * tilt_direction)
tilted_rotation_axis = rotation_matrix.apply(untilted_rotation_axis)

euler_angles = []
for angle in angle_set:
    R1 = R.from_euler("z", angle, degrees=True)
    combined = rotation_matrix * R1
    euler = combined.as_euler("ZYZ", degrees=True)
    euler_angles.append(euler)

euler_angles = np.array(euler_angles) 
recon = algs.fbp(data_norm.array, geo, np.deg2rad(euler_angles))
# show2D(recon)
# plt.plot(recon[int(size/2), int(size/2),:])
# plt.plot(np.linspace(0,size*np.cos(np.deg2rad(tilt)),size),y)

show2D([recon[cube_plot_offsets[0],::], recon[:,cube_plot_offsets[1]], recon[:,:,cube_plot_offsets[2]]], num_cols=3)

# %% Using CIL2tigre geometry 
from cil.plugins.tigre import CIL2TIGREGeometry
# tigre_geom, tigre_angles = CIL2TIGREGeometry.getTIGREGeometry(ig,ag)


ag_in = ag.copy()
system = ag_in.config.system
system.align_reference_frame('tigre')

#TIGRE's interpolation fp must have the detector outside the reconstruction volume otherwise the ray is clipped
#https://github.com/CERN/TIGRE/issues/353
lenx = (ig.voxel_num_x * ig.voxel_size_x)
leny = (ig.voxel_num_y * ig.voxel_size_y)
lenz = (ig.voxel_num_z * ig.voxel_size_z)

panel_width = max(ag_in.config.panel.num_pixels * ag_in.config.panel.pixel_size)*0.5
clearance_len =  np.sqrt(lenx**2 + leny**2 + lenz**2)/2 + panel_width

geo2 = tigre.geometry()
geo2.DSO = clearance_len
geo2.DSD = 2*clearance_len
geo2.mode = 'parallel'


geo2.nVoxel = np.array( [ig.voxel_num_z, ig.voxel_num_y, ig.voxel_num_x] )
# size of each voxel (mm)
# geo2.sVoxel = np.array( [ig.voxel_num_z, ig.voxel_num_y, ig.voxel_num_x] )
# geo2.dVoxel = geo2.sVoxel / geo2.nVoxel
geo2.dVoxel = np.array( [ig.voxel_size_z, ig.voxel_size_y, ig.voxel_size_x]  )

# Detector parameters
# (V,U) number of pixels        (px)
geo2.nDetector = np.array(ag_in.config.panel.num_pixels[::-1])
# size of each pixel            (mm)
geo2.dDetector = np.array(ag_in.config.panel.pixel_size[::-1])
geo2.sDetector = geo2.dDetector * geo2.nDetector    # total size of the detector    (mm)

#TIGRE's interpolation fp must have the detector outside the reconstruction volume otherwise the ray is clipped
#https://github.com/CERN/TIGRE/issues/353
lenx = (ig.voxel_num_x * ig.voxel_size_x)
leny = (ig.voxel_num_y * ig.voxel_size_y)
lenz = (ig.voxel_num_z * ig.voxel_size_z)

panel_width = max(ag_in.config.panel.num_pixels * ag_in.config.panel.pixel_size)*0.5
clearance_len =  np.sqrt(lenx**2 + leny**2 + lenz**2)/2 + panel_width

geo2.is2D = False

ind = np.asarray([2, 0, 1])
flip = np.asarray([1, 1, -1])


geo2.offOrigin = np.array( [0,0,0] )
geo2.offDetector = np.array( [system.detector.position[2], system.detector.position[0], 0])

#shift origin z to match image geometry
#this is in CIL reference frames as the TIGRE geometry rotates the reconstruction volume to match our definitions
geo2.offOrigin[0] += ig.center_z


#convert roll, pitch, yaw
U = system.detector.direction_x[ind] * flip
V = system.detector.direction_y[ind] * flip

roll = np.arctan2(-V[1], V[0])
pitch = np.arcsin(V[2])
yaw = np.arctan2(-U[2],U[1])

#shift origin to match image geometry
geo2.offOrigin[1] += ig.center_y
geo2.offOrigin[2] += ig.center_x

theta = yaw
panel_origin = ag_in.config.panel.origin
if 'right' in panel_origin and 'top' in panel_origin:
    roll += np.pi
elif 'right' in panel_origin:
    yaw += np.pi
elif 'top' in panel_origin:
    pitch += np.pi

geo2.rotDetector = np.array((roll, pitch, yaw))

# total size of the image       (mm)
geo2.sVoxel = geo2.nVoxel * geo2.dVoxel

# Auxiliary
geo2.accuracy = 0.5                        # Accuracy of FWD proj          (vx/sample)

angles = ag.config.angles.angle_data + ag.config.angles.initial_angle
if ag.config.angles.angle_unit == AngleUnit.DEGREE:
    angles *= (np.pi/180.)

#convert CIL to TIGRE angles s
angles = -(angles + np.pi/2 + theta )

#angles in range -pi->pi
for i, a in enumerate(angles):
    while a < -np.pi:
        a += 2 * np.pi
    while a >= np.pi:
        a -= 2 * np.pi
    angles[i] = a

euler_angles = []
for angle in angles:
    R1 = R.from_euler("z", angle, degrees=False)
    combined = rotation_matrix * R1
    euler = combined.as_euler("ZYZ", degrees=False)
    euler_angles.append(euler)

euler_angles = np.array(euler_angles) 
recon = algs.fbp(data_norm.array, geo2, euler_angles)

# show2D(recon)
# plt.plot(recon[int(size/2), int(size/2),:])
# plt.plot(np.linspace(0,size*np.cos(np.deg2rad(tilt)),size),y)

show2D([recon[cube_plot_offsets[0],::], recon[:,cube_plot_offsets[1]], recon[:,:,cube_plot_offsets[2]]], num_cols=3)
# %%

# %%
from scipy.fftpack import fft, ifft, fftfreq

def sl_ramp_filter(proj, dx):
    n = proj.shape[-1]
    filt = shepp_logan_filter(n, dx)
    proj_fft = np.fft.rfft(proj, axis=-1)
    filtered = np.fft.irfft(proj_fft * filt[None, :], axis=-1)
    return filtered

def shepp_logan_filter(n, dx):
    freqs = np.fft.rfftfreq(n, dx)
    ramp = np.abs(freqs)
    sl_filter = ramp * np.sinc(freqs / (2 * freqs.max()))
    return sl_filter

def ramp_filter(projection, dx):
    n = projection.shape[1]
    freqs = fftfreq(n, d=dx)
    filter_kernel = np.abs(freqs)
    projection_fft = fft(projection, axis=1)
    projection_filtered = np.real(ifft(projection_fft * filter_kernel[None, :], axis=1))
    return projection_filtered
# %%

# %% standard implementation
data_2D = data_norm.array[:,int(size/2),:]
n_angles, det_y = data_2D.shape
nx, ny = geo.nVoxel[:2]
dx, dy = geo.dVoxel[:2]

x = (np.arange(nx) - nx / 2 + 0.5) * dx
y = (np.arange(ny) - ny / 2 + 0.5) * dy
X, Y = np.meshgrid(x, y, indexing='ij')  # shape (nx, ny)

recon_image = np.zeros((nx, ny), dtype=np.float32)

angles = np.deg2rad(angle_set)

for i, theta in enumerate(angles):
    proj = data_2D[i]  # shape (det_y,)
    proj_filtered = ramp_filter(proj[np.newaxis, :], dx=dy)[0]

    x_rot = X * np.cos(theta) + Y * np.sin(theta)

    iy = x_rot / dy + det_y / 2


    iy_flat = iy.ravel() 
    coords = np.vstack([iy_flat]) 
    proj_interp = map_coordinates(proj_filtered, coords, order=1, mode='nearest')

    recon_image += proj_interp.reshape((nx, ny))
    
show2D(recon_image)
plt.plot(recon_image[int(size/2), int(size/2), :])
# %% standard implementation 3D
data_3D = data_norm.array
n_angles, det_y, det_x = data_3D.shape

nx, ny, nz = geo.nVoxel
dx, dy, dz = geo.dVoxel

x = (np.arange(nx) - nx / 2 + 0.5) * dx
y = (np.arange(ny) - ny / 2 + 0.5) * dy
z = (np.arange(nz) - nz/2 + 0.5) * dz
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # shape (nx, ny)
coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1) 

recon_image = np.zeros((nx, ny, nz), dtype=np.float32)

limited_angles = np.arange(0, 360, 1)
angles = np.deg2rad(limited_angles)

for i, theta in enumerate(angles):
    print(str(i) + ' / ' + str(len(angles)))
    proj = data_3D[i] 
    proj_filtered = ramp_filter(proj[np.newaxis, :], dx=dy)[0]

    Rz = R.from_rotvec(theta * np.array([0, 0, 1]))
    coords_rot = Rz.apply(coords) 

    iy = coords_rot[:, 1] / dy + det_y / 2
    ix = coords_rot[:, 2] / dz + det_x / 2

    coords_proj = np.vstack([iy, ix])  
    proj_interp = map_coordinates(proj_filtered, coords_proj, order=1, mode='nearest')

    recon_image += proj_interp.reshape((nx, ny, nz))
    
show2D([recon_image[int(size/2),:,:],recon_image[:,int(size/2),:],recon_image[:,:,int(size/2)]], num_cols=1)
plt.plot(recon_image[int(size/2), int(size/2), :])

# %% The tilted implementation
# Following the implementation in Yang, Min & Wang, Gao & Liu, Yongzhan. (2010). New reconstruction method for x-ray testing of 
# multilayer printed circuit board. Optical Engineering - OPT ENG. 49. http://dx.doi.org/10.1117/1.3430629
# - but for parallel beam. This expresses how to apply the projection along a tilted rotation axis. The filter is applied in the projection domain 
data_norm.reorder('tigre')

tilt = 30
n_angles, det_y, det_x = data_norm.shape

nx, ny, nz = geo.nVoxel
dx, dy, dz = geo.dVoxel

x = (np.arange(nx) - nx / 2 + 0.5) * dx
y = (np.arange(ny) - ny / 2 + 0.5) * dy
z = (np.arange(nz) - nz/2 + 0.5) * dz
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  
coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1) 

tilt_rad = np.deg2rad(tilt)
Rx = R.from_rotvec(tilt_rad * np.array([1, 0, 0]))
tilted_axis = Rx.apply([0, 0, 1])
det_up = Rx.apply([0, 0, 1])
det_right = Rx.apply([1, 0, 0])

recon_image = np.zeros((nx, ny, nz), dtype=np.float32)

limited_angles = np.arange(0, 360, 1)
angles = np.deg2rad(limited_angles)

for i, theta in enumerate(angles):

    print(str(i) + ' / ' + str(len(angles)))
    proj = data_norm.array[i]
    proj_filtered = sl_ramp_filter(proj, dx=dy)

    # proj_filtered = proj

    # Rz = R.from_rotvec(theta * tilted_axis)
    Rz = R.from_rotvec(theta * np.array([0, 0, 1]))

    u = Rz.apply(det_up)
    v = Rz.apply(det_right)

    du = coords @ u
    dv = coords @ v

    iy = du / dy + det_y/2
    ix = dv / dx + det_x/2

    coords_proj = np.vstack([iy, ix])  
    
    proj_interp = map_coordinates(proj_filtered, coords_proj, order=1, mode='nearest')

    recon_image += proj_interp.reshape((nx, ny, nz))
    
show2D([recon_image[int(size/2),:,:],
        recon_image[:,int(size/2),:],
        recon_image[:,:,int(size/2)]], num_cols=3)
# %%
plt.plot(recon_image[int(size/2), :, int(size/2)], label='x')
plt.plot(recon_image[:, int(size/2), int(size/2)], label='y')
plt.xlabel('Horizontal index')
plt.ylabel('Value')
plt.legend()
plt.grid()
# %% Back projection inverse filtration. Apply the reconstruction without filtering first, then apply filtering to the reconstructed volume
# This is following the implementation in Lauritsch, Guenter and Haerer, Wolfgang H. (1998) Theoretical framework for filtered back projection 
# in tomosynthesis Proc. SPIE 3338, Medical Imaging 1998: Image Processing, (24 June 1998); https://doi.org/10.1117/12.310839
# this paper is about tomosynthesis but they show how to create the filter and apply it after the unfiltered projection


data_norm.reorder('tigre')

tilt = 30
n_angles, det_y, det_x = data_norm.shape

nx, ny, nz = geo.nVoxel
dx, dy, dz = geo.dVoxel

x = (np.arange(nx) - nx / 2 + 0.5) * dx
y = (np.arange(ny) - ny / 2 + 0.5) * dy
z = (np.arange(nz) - nz/2 + 0.5) * dz
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  
coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1) 

tilt_rad = np.deg2rad(tilt)
Rx = R.from_rotvec(tilt_rad * np.array([1, 0, 0]))
tilted_axis = Rx.apply([0, 0, 1])
det_up = Rx.apply([0, 0, 1])
det_right = Rx.apply([1, 0, 0])

recon_image = np.zeros((nx, ny, nz), dtype=np.float32)

limited_angles = np.arange(0, 360, 1)
angles = np.deg2rad(limited_angles)

for i, theta in enumerate(angles):

    print(str(i) + ' / ' + str(len(angles)))
    proj = data_norm.array[i]

    Rz = R.from_rotvec(theta * np.array([0, 0, 1]))
    u = Rz.apply(det_up)
    v = Rz.apply(det_right)
    du = coords @ u
    dv = coords @ v
    iy = du / dy + det_y/2
    ix = dv / dx + det_x/2
    coords_proj = np.vstack([iy, ix])  
    proj_interp = map_coordinates(proj, coords_proj, order=1, mode='nearest')

    recon_image += proj_interp.reshape((nx, ny, nz))

show2D([recon_image[int(size/2),:,:],
        recon_image[:,int(size/2),:],
        recon_image[:,:,int(size/2)]], num_cols=3)
plt.plot(recon_image[int(size/2), int(size/2), :])

kx = np.fft.fftfreq(nx, d=dx)[:,None,None]
ky = np.fft.fftfreq(ny, d=dy)[None,:,None]
kz = np.fft.fftfreq(nz, d=dz)[None,None,:]
freqs = np.sqrt(kx**2 + ky**2 + kz**2)


ksq = np.zeros_like(recon_image)
H = np.zeros_like(recon_image, dtype=np.float32)
for theta in angles:
    u = np.array([ np.cos(theta)*np.cos(tilt_rad),
                   np.sin(theta)*np.cos(tilt_rad),
                   np.sin(tilt_rad) ])
    kdotu = kx * u[0] + ky * u[1] + kz * u[2]
    H += np.abs(kdotu)

H = H / len(angles)

window = np.sinc(freqs / (2 * freqs.max()))
H *= window

alpha = 1
H **= alpha

F = np.fft.fftn(recon_image)
F_filtered = F * H
recon_filt = np.real(np.fft.ifftn(F_filtered))

show2D([recon_filt[int(size/2),:,:],
        recon_filt[:,int(size/2),:],
        recon_filt[:,:,int(size/2)]], num_cols=3)
# %%
# plt.plot(recon_filt[:, int(size/2), int(size/2)])
plt.plot(recon_image[int(size/2), :, int(size/2)], label='x')
plt.plot(recon_image[:, int(size/2), int(size/2)], label='y')
plt.xlabel('Horizontal index')
plt.ylabel('Value')
plt.legend()
plt.grid()
# %%
plt.plot(recon_filt[int(size/2), :, int(size/2)], label='x')
plt.plot(recon_filt[:, int(size/2), int(size/2)], label='y')
plt.xlabel('Horizontal index')
plt.ylabel('Value')
plt.legend()
plt.grid()
# %% Computed laminography reprojection following Liang Sun et al 2021 A reconstruction method for cone-beam computed
# laminography based on projection transformation Meas. Sci. Technol. 32 045403 https://doi.org/10.1088/1361-6501/abc965

# interpolation
def interp_4points(image, x0, y0):
    # equation 5
    
    h, w = image.shape
    x1 = int(np.floor(x0))
    y1 = int(np.floor(y0))
    value = 0.0
    for xi in [x1, x1 + 1]:
        for yi in [y1, y1 + 1]:
            if 0 <= xi < w and 0 <= yi < h:
                weight = abs((x0 - xi - 1) * (y0 - yi - 1))
                value += weight * image[yi, xi]
    return value


data_norm.reorder('tigre')

tilt = 30
n_angles, det_y, det_x = data_norm.shape

nx, ny, nz = geo.nVoxel
dx, dy, dz = geo.dVoxel

x = (np.arange(nx) - nx / 2 + 0.5) * dx
y = (np.arange(ny) - ny / 2 + 0.5) * dy
z = (np.arange(nz) - nz/2 + 0.5) * dz
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  
coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1) 

tilt_rad = np.deg2rad(tilt)
Rx = R.from_rotvec(tilt_rad * np.array([1, 0, 0]))
tilted_axis = Rx.apply([0, 0, 1])
det_up = Rx.apply([0, 0, 1])
det_right = Rx.apply([1, 0, 0])

recon_image = np.zeros((nx, ny, nz), dtype=np.float32)

limited_angles = np.arange(0, 360, 1)
angles = np.deg2rad(limited_angles)

# transformed = np.zeros_like(data_norm.array)
for i, theta in enumerate(angles):
    print(str(i) + ' / ' + str(len(angles)))
    proj = data_norm.array[i]
    h, w = proj.shape

    # Apply interpolation
    transformed_proj = np.zeros_like(proj)
    for j in range(h):
        for k in range(w):
            x_ct = (k - w / 2 + 0.5) * dx
            y_ct = (j - h / 2 + 0.5) * dy

            y_tilt = y_ct * np.cos(tilt_rad)
            z_tilt = y_ct * np.sin(tilt_rad)

            x_proj = x_ct
            y_proj = y_tilt

            x_pix = x_proj / dx + w / 2 - 0.5
            y_pix = y_proj / dy + h / 2 - 0.5
            transformed_proj[j, k] = interp_4points(proj, x_pix, y_pix)


    transformed_proj_filtered = sl_ramp_filter(transformed_proj, dx=dy)


    Rz = R.from_rotvec(theta * np.array([0, 0, 1]))

    u = Rz.apply(det_up)
    v = Rz.apply(det_right)

    du = coords @ u
    dv = coords @ v

    iy = du / dy + det_y/2
    ix = dv / dx + det_x/2

    coords_proj = np.vstack([iy, ix])  

    proj_interp = map_coordinates(transformed_proj_filtered, coords_proj, order=1, mode='nearest')

    recon_image += proj_interp.reshape((nx, ny, nz))

show2D([recon_image[int(size/2),:,:],
        recon_image[:,int(size/2),:],
        recon_image[:,:,int(size/2)]], num_cols=3)
plt.plot(recon_image[:, int(size/2), int(size/2)])
# %%
plt.plot(recon_image[int(size/2), :, int(size/2)], label='x')
plt.plot(recon_image[:, int(size/2), int(size/2)], label='y')
plt.xlabel('Horizontal index')
plt.ylabel('Value')
plt.legend()
plt.grid()
# %% Weight projections

data_norm.reorder('tigre')

tilt = 30
n_angles, det_y, det_x = data_norm.shape

nx, ny, nz = geo.nVoxel
dx, dy, dz = geo.dVoxel

x = (np.arange(nx) - nx / 2 + 0.5) * dx
y = (np.arange(ny) - ny / 2 + 0.5) * dy
z = (np.arange(nz) - nz/2 + 0.5) * dz
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  
coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1) 

tilt_rad = np.deg2rad(tilt)
Rx = R.from_rotvec(tilt_rad * np.array([1, 0, 0]))
tilted_axis = Rx.apply([0, 0, 1])
det_up = Rx.apply([0, 0, 1])
det_right = Rx.apply([1, 0, 0])

recon_image = np.zeros((nx, ny, nz), dtype=np.float32)

limited_angles = np.arange(0, 360, 1)
angles = np.deg2rad(limited_angles)
dus = np.zeros((len(angles), 3))
weights = np.zeros_like(angles)
D = 1
for i, theta in enumerate(angles):

    Rz = R.from_rotvec(theta * np.array([0, 0, 1]))
    p_theta = Rz.apply(np.array([1, 0, 0]))

    weights[i] = np.sqrt(1 - np.dot(tilted_rotation_axis, p_theta)**2)


    print(str(i) + ' / ' + str(len(angles)))
    

    u = Rz.apply(det_up)
    v = Rz.apply(det_right)

    du = coords @ u
    dv = coords @ v

    iy = du / dy + det_y/2
    ix = dv / dx + det_x/2




    coords_proj = np.vstack([iy, ix])  

    proj = 1*data_norm.array[i]
    proj_filtered = sl_ramp_filter(proj, dx=dy)
    
    proj_interp = map_coordinates(proj_filtered, coords_proj, order=1, mode='nearest')

    weights[i] = abs(u[1]) # this one works well

    # weight = np.abs(np.dot(u, [0, 0, 1])) / np.linalg.norm(u)
    recon_image += 1*proj_interp.reshape((nx, ny, nz))


    
show2D([recon_image[int(size/2),:,:],
        recon_image[:,int(size/2),:],
        recon_image[:,:,int(size/2)]], num_cols=3)
# plt.plot(recon_image[int(size/2), int(size/2), :])


# %%

plt.plot(recon_image[int(size/2), :, int(size/2)], label='Unweighted')
plt.plot(5*recon_weighted[int(size/2), :, int(size/2)], label='Weighted')
plt.xlabel('Horizontal x index')
plt.ylabel('Value')
plt.legend()
plt.grid()
# %%
plt.plot(recon_image[:, int(size/2), int(size/2)], label='Unweighted')
plt.plot(5*recon_weighted[:, int(size/2), int(size/2)], label='Weighted')
plt.xlabel('Horizontal y index')
plt.ylabel('Value')
plt.legend()
plt.grid()
# %%
if include_cube:
    show2D(recon_image, slice_list = [('X', cube_plot_offsets[0]), ('Y', cube_plot_offsets[1]), ('Z', cube_plot_offsets[2])], num_cols=3)
    show2D(recon_weighted, slice_list = [('X', cube_plot_offsets[0]), ('Y', cube_plot_offsets[1]), ('Z', cube_plot_offsets[2])], num_cols=3)
    show2D(recon_image-recon_weighted, slice_list = [('X', cube_plot_offsets[0]), ('Y', cube_plot_offsets[1]), ('Z', cube_plot_offsets[2])], num_cols=3)
# %%
plt.plot(angle_set, weights)
plt.xlabel('Angle (degrees)')
plt.ylabel(r'Weight $\frac{1}{|a_{\perp}|}$')
plt.grid()

# %%
plt.plot(recon_image[int(size/2), :, int(size/2)], label='x')
plt.plot(recon_image[:, int(size/2), int(size/2)], label='y')
plt.xlabel('Horizontal index')
plt.ylabel('Value')
plt.legend()
plt.grid()
# %%
weights1 = np.zeros_like(angles)
weights2 = np.zeros_like(angles)

Rx = R.from_rotvec(tilt_rad * np.array([1, 0, 0]))
for i, theta in enumerate(angles):

    weights1[i] = np.cos((tilt_rad)*np.cos(theta))


    # Rz = R.from_rotvec(theta * tilted_axis)
    Rz = R.from_rotvec(theta * np.array([0, 0, 1]))

    Rtotal = Rx*Rz

    u = Rz.apply(det_up)
    v = Rz.apply(det_right)

    du = coords @ u
    dv = coords @ v

    iy = du / dy + det_y/2
    ix = dv / dx + det_x/2

    R_total = Rz*Rx


    dus[i,:] = u 

    weights2[i] = abs(u[1])
    # weight = np.abs(np.dot(u, [0, 0, 1])) / np.linalg.norm(u)
    # recon_image += weights[i]*proj_interp.reshape((nx, ny, nz))

plt.plot(weights1)
plt.plot(weights2)
# %%
weights1 = np.cos((tilt_rad)*np.cos((np.pi/2)-angles))
plt.plot(weights1)
plt.plot(abs(dus[:,1]))
# %%


# %%


