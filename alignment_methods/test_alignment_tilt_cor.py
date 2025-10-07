# %%
import os
import sys
import numpy as np
import scipy
import json
import matplotlib.pyplot as plt

from cil.processors import TransmissionAbsorptionConverter
from cil.utilities.display import show_geometry, show2D, show1D
from cil.utilities.jupyter import islicer
from scipy.spatial.transform import Rotation as R
from cil.processors import FluxNormaliser, Normaliser
from cil.framework import AcquisitionData, AcquisitionGeometry
from cil.plugins.astra.processors import FBP
from cil.plugins.astra.operators import ProjectionOperator
# %%
from cil.io import NEXUSDataReader
data = NEXUSDataReader(file_name='../output_data/cylinder_tilt_30_cor_offset_5.nxs').read()
ag = data.geometry
show_geometry(ag)
islicer(data, direction=1)
# %%

data_filtered = data.copy()
data_filtered.fill(scipy.ndimage.sobel(data.as_array(), axis=0, mode='reflect', cval=0.0))
# %% 
# data.reorder('astra')
ig = ag.get_ImageGeometry()

beam_direction = np.array([0, 1, 0])
detector_x_direction = np.array([1, 0, 0])
detector_y_direction = np.array([0, 0, -1])
rotation_axis = np.array([0, 0, 1]) # the untilted rotation axis

tilt = 0
offset = 0
# create the tilted rotation axis
tilt_rad = np.deg2rad(tilt)
rotation_matrix = R.from_rotvec(tilt_rad * detector_x_direction)
tilted_rotation_axis = rotation_matrix.apply(rotation_axis)

# %% set centre of rotation
ag.set_centre_of_rotation(offset=offset, distance_units='pixels')
ag.config.system.rotation_axis.direction = tilted_rotation_axis
print(ag.config.system.rotation_axis.position)
print(ag.config.system.rotation_axis.direction)
show_geometry(ag)

# %% reconstruct
ig = ag.get_ImageGeometry()
ig.voxel_num_z = 300 
fbp = FBP(ig, ag)
recon = fbp(data)
show2D(recon)
# %%

tilt = 30
offset = 5
# create the tilted rotation axis
tilt_rad = np.deg2rad(tilt)
rotation_matrix = R.from_rotvec(tilt_rad * detector_x_direction)
tilted_rotation_axis = rotation_matrix.apply(rotation_axis)

ag.set_centre_of_rotation(offset=offset, distance_units='pixels')
ag.config.system.rotation_axis.direction = tilted_rotation_axis
print(ag.config.system.rotation_axis.position)
print(ag.config.system.rotation_axis.direction)
show_geometry(ag)
ig = ag.get_ImageGeometry()
ig.voxel_num_z = 300 
fbp = FBP(ig, ag)
recon = fbp(data)
show2D(recon)
# %%


# %%

tilt_rad = np.radians(tilt)

ag2 = AcquisitionGeometry.create_Parallel3D(
       rotation_axis_direction=[0, -np.sin(tilt_rad), np.cos(tilt_rad)],
       units="microns",
)

ag2.set_panel(
       num_pixels=[data.shape[2], data.shape[0]],
       origin='top-left',
       pixel_size=1
)

ag2.set_angles(ag.angles)  
ag2.set_centre_of_rotation(offset, distance_units='pixels')
ag2.dimension_labels = ag.dimension_labels

ig = ag2.get_ImageGeometry()
ig.voxel_num_z = 300 
fbp = FBP(ig, ag2)
data.reorder('astra')
recon = fbp(data)
show2D(recon)      
#  acquisition_data = AcquisitionData(processed_data, geometry=geometry)
# %%
ig.voxel_num_z = 300 
fbp = FBP(ig, ag)
recon305 = fbp(data)


# %%
# recon.apply_circular_mask(0.9)
slice_list = [('vertical','centre'), ('horizontal_y',int(recon301.shape[2]/2)), ('horizontal_x',int(recon301.shape[1]/2))]
show2D(recon305, slice_list=slice_list, num_cols=3)
# %%


# Filtered
fbp = FBP(ig, ag)
fbp.set_input(data_filtered)
recon_filtered305 = fbp.get_output()
recon_filtered305.apply_circular_mask(0.9)

slice_list = [('vertical','centre'), ('horizontal_y',int(recon301.shape[2]/2)), ('horizontal_x',int(recon301.shape[1]/2))]
show2D(recon_filtered305, slice_list=slice_list, num_cols=3)

#%%
A = ProjectionOperator(ig, ag)
reprojection305 = A.direct(recon_filtered305)
residual305 = reprojection305 - data_filtered

slice_list_proj = [('angle', 0), ('angle', 45), ('angle',90), ('angle', 135), ('angle',180)]
show2D(residual305, slice_list=slice_list_proj, num_cols=5)

# %%
# %% View the range

# %%

slice_list1d = [[(0,int(recon299.shape[0]/2)),(1,int(recon299.shape[1]/2))],
                [(0,int(recon299.shape[0]/2)),(2,int(recon299.shape[2]/2))],
                [(1,int(recon299.shape[1]/2)),(2,int(recon299.shape[2]/2))]]
dataset_labels = ["tilt=29.5", "tilt=29.9", "tilt=30", "tilt=30.1", "tilt=30.5"]

show1D([recon295, recon299, recon, recon301, recon305], slice_list=slice_list1d, dataset_labels=dataset_labels)

show1D([recon_filtered295, recon_filtered299, recon_filtered, recon_filtered301, recon_filtered305], slice_list=slice_list1d, dataset_labels=dataset_labels)


# %%
slice_list1d = [[(0,int(residual.shape[0]/2)),(1,int(residual.shape[1]/2))],
                [(0,int(residual.shape[0]/2)),(2,int(residual.shape[2]/2))],
                [(1,int(residual.shape[1]/2)),(2,int(residual.shape[2]/2))]]
show1D([residual295, residual299, residual, residual301, residual305], slice_list=slice_list1d, dataset_labels=dataset_labels)
# %%
plt.plot([29.5, 29.9, 30.0, 30.1, 30.5], [(residual295**2).sum(), (residual299**2).sum(), (residual**2).sum(), (residual301**2).sum(), (residual305**2).sum()])
# %%
tilts = [29.5, 29.9, 30.0, 30.1, 30.5]
losses = np.zeros_like(tilts)
names = ["residual295", "residual299", "residual", "residual301", "residual305"]
for i, name in enumerate(names):
      f = np.load(name + ".npy", allow_pickle=True)
      losses[i] = (f**2).sum()
# %%
plt.plot(tilts, losses)
plt.grid()
plt.xlabel("Tilts")
plt.ylabel("Sum of squares of reprojection residual")
# %% Limit the size of the reconstruction volume
ig.voxel_num_z = 300
fbp = FBP(ig, ag)
recon = fbp(data)
# recon.apply_circular_mask(0.9)
show2D(recon)

# %%# %%

# %%
data.reorder('astra')

data_filtered = data.copy()
data_filtered.fill(scipy.ndimage.sobel(data.as_array(), axis=0, mode='reflect', cval=0.0))

# tilts = np.arange(20, 40.1, 5)
# tilts = np.array([28,29,30,31,32])
# offsets = np.array([2,3,4,5,6])



# %% 3D search


tilts = np.array([29.5, 29.6, 29.7, 29.8, 29.9, 30, 30.1, 30.2, 30.3, 30.4, 30.5])
offsets = np.array([4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5])
# %%
losses = np.empty((len(tilts),len(offsets)))
losses_filtered = np.empty((len(tilts),len(offsets)))
for t, tilt in enumerate(tilts):
    for o, offset in enumerate(offsets):
        ag_tilt = ag.copy()

        tilt_rad = np.deg2rad(tilt)
        rotation_matrix = R.from_rotvec(tilt_rad * detector_x_direction)
        tilted_rotation_axis = rotation_matrix.apply(rotation_axis)

        ag_tilt.set_centre_of_rotation(offset=offset, distance_units='pixels')
        ag_tilt.config.system.rotation_axis.direction = tilted_rotation_axis
        print(ag_tilt.config.system.rotation_axis.direction)
        print(ag_tilt.config.system.rotation_axis.position)

        ig = ag_tilt.get_ImageGeometry()
        ig.voxel_num_z = 300

        # Filtered
        fbp = FBP(ig, ag_tilt)
        fbp.set_input(data_filtered)
        recon_filtered = fbp.get_output()
        recon_filtered.apply_circular_mask(0.9)

        A = ProjectionOperator(ig, ag_tilt)
        reprojection = A.direct(recon_filtered)
        residual = reprojection - data_filtered

        # show2D([data_filtered, reprojection, residual, residual**2], num_cols=4,
        #        title=['Data filtered', 'Reprojection', 'Residual', 'L2 Norm Residual'], 
        #        fix_range=[(-0.3, 1), (-0.3, 1), (-0.5, 0), (0, 0.25)])
        
        # show2D([data_filtered, reprojection, residual_centered, residual_centered**2], num_cols=4,
        #        title=['Data filtered', 'Reprojection', 'Residual', 'L2 Norm Residual'], 
        #        fix_range=[(-0.3, 1), (-0.3, 1), (-0.5, 0), (0, 0.25)])
        
        loss = (residual ** 2).sum()
        losses_filtered[t,o] = loss

# %%
losses_filtered = np.load('losses_filtered.npy')

# %%
# A = np.array(losses_filtered)
# A = (A - A.min()) / (A.max() - A.min())
# B = np.array(losses)
# B = (B - B.min()) / (B.max() - B.min())
# %%
plt.figure(figsize=(6, 4))
plt.imshow( losses_filtered, aspect='equal', origin='lower',
           extent=[offsets[0], offsets[-1], tilts[0], tilts[-1]])
plt.xlabel('Offset (pixels)')
plt.ylabel('Tilt angle (degrees)')
plt.ylabel('Tilt angle (degrees)')

plt.colorbar(label='Filtered projection residual (sum of squares)')

# %%
for t, tilt in enumerate(tilts):
       plt.plot(offsets, losses_filtered[t,:], label="tilt = " + str(tilt))

plt.xlabel("Centre of rotation offset (pixels)")
plt.ylabel("Filtered projection residual (sum of squares)")
plt.grid()
plt.legend()
# %%
for o, offset in enumerate(tilts):
       plt.plot(tilts, losses_filtered[:,o], label="offset = " + str(offset))

plt.xlabel("Tilt angle (degrees)")
plt.ylabel("Filtered projection residual (sum of squares)")
plt.grid()
plt.legend()
# plt.plot(tilts[np.argmin(losses_filtered)], np.min(losses_filtered), 'rx')

# %%
# %%
tilt = 85
ag_tilt = ag.copy()

tilt_rad = np.deg2rad(tilt)
rotation_matrix = R.from_rotvec(tilt_rad * detector_x_direction)
tilted_rotation_axis = rotation_matrix.apply(rotation_axis)

ag_tilt.config.system.rotation_axis.direction = tilted_rotation_axis
print(ag_tilt.config.system.rotation_axis.direction)

ig = ag_tilt.get_ImageGeometry()
ig.voxel_num_z = 10

# Unfiltered
fbp = FBP(ig, ag_tilt)
fbp.set_input(data)
recon = fbp.get_output()
recon.apply_circular_mask(0.9)
slice_list = [('vertical','centre'), ('horizontal_y',int(recon.shape[2]/2)), ('horizontal_x',int(recon.shape[1]/2))]
show2D(recon,
       slice_list=slice_list,
       num_cols=3)
# %%

A = ProjectionOperator(ig, ag_tilt)
reprojection = A.direct(recon)
residual = reprojection - data

show2D([data, reprojection, residual, residual**2], slice_list=(1,250), num_cols=4,
       title=['Data', 'Reprojection', 'Residual', 'L2 Norm Residual'])


print((residual**2).sum())
# loss = (residual ** 2).sum()
# losses.append(loss)
# %%
# Filtered
fbp = FBP(ig, ag_tilt)
fbp.set_input(data_filtered)
recon_filtered = fbp.get_output()
recon_filtered.apply_circular_mask(0.9)
show2D(recon_filtered,
       slice_list=slice_list,
       num_cols=3)
# %%
A = ProjectionOperator(ig, ag_tilt)
reprojection = A.direct(recon_filtered)
residual = reprojection - data_filtered

show2D([data_filtered, reprojection, residual, residual**2], slice_list=(1,250), num_cols=4,
       title=['Data filtered', 'Reprojection', 'Residual', 'L2 Norm Residual'], 
       fix_range=[(-0.3, 1), (-0.3, 1), (-0.5, 0), (0, 0.25)])
print((residual**2).sum())
# show2D([data_filtered, reprojection, residual_centered, residual_centered**2], num_cols=4,
#        title=['Data filtered', 'Reprojection', 'Residual', 'L2 Norm Residual'], 
#        fix_range=[(-0.3, 1), (-0.3, 1), (-0.5, 0), (0, 0.25)])

