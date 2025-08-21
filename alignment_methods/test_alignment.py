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
data = NEXUSDataReader(file_name='../output_data/cylinder_tilt_30.nxs').read()
ag = data.geometry
show_geometry(ag)
islicer(data, direction=1)
# %% 
# data.reorder('astra')
ig = ag.get_ImageGeometry()

tilt = 30

beam_direction = np.array([0, 1, 0])
detector_x_direction = np.array([1, 0, 0])
detector_y_direction = np.array([0, 0, -1])
rotation_axis = np.array([0, 0, 1]) # the untilted rotation axis

# create the tilted rotation axis
tilt_rad = np.deg2rad(tilt)
rotation_matrix = R.from_rotvec(tilt_rad * detector_x_direction)
tilted_rotation_axis = rotation_matrix.apply(rotation_axis)

ag.config.system.rotation_axis.direction = tilted_rotation_axis

# ig.voxel_num_z = 1 
fbp = FBP(ig, ag)
recon = fbp(data)
# recon.apply_circular_mask(0.9)
show2D(recon)

# %% View the range
show1D(recon, slice_list=[(0,int(recon.shape[0]/2)),(1,int(recon.shape[1]/2))])
show1D(recon, slice_list=[(0,int(recon.shape[0]/2)),(2,int(recon.shape[2]/2))])
show1D(recon, slice_list=[(1,int(recon.shape[1]/2)),(2,int(recon.shape[2]/2))])
# %% Limit the size of the reconstruction volume
ig.voxel_num_z = 300
fbp = FBP(ig, ag)
recon = fbp(data)
# recon.apply_circular_mask(0.9)
show2D(recon)
# %%# %%

losses = []
losses_filtered = []
data.reorder('astra')

data_filtered = data.copy()
data_filtered.fill(scipy.ndimage.sobel(data.as_array(), axis=0, mode='reflect', cval=0.0))

tilts = np.array([20.0, 22, 24, 26, 28, 28.5, 29, 29.2, 29.4, 29.6, 29.8, 30.0, 30.2, 30.4, 30.6, 30.8, 31., 31.5, 32, 34, 36., 38, 40.0 ])
# tilts = np.arange(20, 40.1, 5)
for tilt in tilts:
    ag_tilt = ag.copy()

    tilt_rad = np.deg2rad(tilt)
    rotation_matrix = R.from_rotvec(tilt_rad * detector_x_direction)
    tilted_rotation_axis = rotation_matrix.apply(rotation_axis)

    ag_tilt.config.system.rotation_axis.direction = tilted_rotation_axis
    print(ag_tilt.config.system.rotation_axis.direction)

    ig = ag_tilt.get_ImageGeometry()
    ig.voxel_num_z = 1

    # Unfiltered
    fbp = FBP(ig, ag_tilt)
    fbp.set_input(data)
    recon = fbp.get_output()
    recon.apply_circular_mask(0.9)

    A = ProjectionOperator(ig, ag_tilt)
    reprojection = A.direct(recon)
    residual = reprojection - data

    # show2D([data, reprojection, residual, residual**2], num_cols=4,
    #        title=['Data', 'Reprojection', 'Residual', 'L2 Norm Residual'], 
    #        fix_range=[(-0.3, 1), (-0.3, 1), (-0.5, 0), (0, 0.25)])
    
    loss = (residual ** 2).sum()
    losses.append(loss)

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
    losses_filtered.append(loss)


# %%
A = np.array(losses_filtered)
A = (A - A.min()) / (A.max() - A.min())
B = np.array(losses)
B = (B - B.min()) / (B.max() - B.min())

plt.figure(figsize=(6, 4))
plt.plot(tilts, A/B)
plt.xlabel("Tilt angle (degrees)")
plt.ylabel("Filtered projection residual (sum of squares)")
plt.grid()
# plt.plot(tilts[np.argmin(losses_filtered)], np.min(losses_filtered), 'rx')


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


