# %%
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

from cil.plugins.astra.processors import FBP
from cil.processors import TransmissionAbsorptionConverter
from cil.utilities.display import show_geometry, show2D
from cil.utilities.jupyter import islicer
from cil.processors import FluxNormaliser, Normaliser
from cil.framework import AcquisitionData, AcquisitionGeometry, DataContainer
from cil.io import NEXUSDataWriter
from cil.plugins.astra import ProjectionOperator

import numpy as np
from cil.framework import ImageGeometry, ImageData
from cil.utilities.display import show2D

import numpy as np
from cil.framework import ImageGeometry, ImageData
from cil.utilities.display import show2D
# %%


# Volume shape (Z, Y, X)
shape = (64, 256, 256)
volume = np.zeros(shape, dtype=np.float32)

margin = 16
center_x, center_y = shape[2] // 2, shape[1] // 2
radius = min(center_x, center_y) - margin
edge_thickness = 1
yy, xx = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
r2 = (xx - center_x)**2 + (yy - center_y)**2
cylinder_mask = r2 <= radius**2
cylinder_edge_mask = (r2 >= (radius - edge_thickness)**2) & (r2 <= (radius + edge_thickness)**2)

grid_spacing = 16

x_range = range(margin, shape[2] - margin)
y_range = range(margin, shape[1] - margin)

for y in y_range:
    for x in x_range:
        if not cylinder_mask[y, x]:
            continue
        if (x % grid_spacing == 0) or (y % grid_spacing == 0):
            volume[margin:shape[0]-margin, y, x] = 1.0 

for z in range(margin, shape[0] - margin):
    volume[z][cylinder_edge_mask] = 1  

ig = ImageGeometry(voxel_num_x=shape[2], voxel_num_y=shape[1], voxel_num_z=shape[0])
lines = ImageData(volume, geometry=ig)

show2D(lines,
       slice_list=[
           (0, shape[0]//2),  
           (1, 100),  
           (2, 100),  # YZ
       ],
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
#     .set_angles([0])\
#     .set_panel(lines.shape[1:3])

ag = AcquisitionGeometry.create_Parallel3D(rotation_axis_direction=tilted_rotation_axis)\
    .set_angles(np.arange(0,360))\
    .set_panel(lines.shape[1:3])
ag.dimension_labels = ('vertical', 'angle','horizontal')
# %%
A = ProjectionOperator(ig, ag)
proj = A.direct(lines)
ag_slice_list = [('angle', 0),('angle',45), ('angle',90), ('angle',135), ('angle',180)]
show2D(proj,
       slice_list = ag_slice_list, 
       num_cols=5,
       fix_range=(0,25))
# %%
FBP_recon = FBP(ig, ag)(proj)
# %%
slice_list = [('vertical','centre'), ('horizontal_x',int(FBP_recon.shape[1]/2)), ('horizontal_y',int(FBP_recon.shape[2]/2))]
show2D(FBP_recon,
       slice_list=slice_list,
       num_cols=3)
# %%
from cil.optimisation.functions import LeastSquares, ZeroFunction
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from cil.optimisation.algorithms import FISTA
Projector = ProjectionOperator(ig, ag)
LS = LeastSquares(A=Projector, b=proj)
alpha = 0.05
TV = FGP_TV(alpha=alpha, nonnegativity=True, device='gpu')
fista_TV = FISTA(initial=FBP_recon, f=LS, g=TV, update_objective_interval=10)
# %%
fista_TV.run(70)
LS_reco = fista_TV.solution
show2D(LS_reco,slice_list=slice_list, num_cols=3)
# %%
show2D(LS_reco-FBP_recon,slice_list=slice_list, num_cols=3, cmap='RdBu_r', fix_range=(-0.2, 0.2))
# %%
import scipy
losses = []
losses_filtered = []

data_filtered = proj.copy()
data_filtered.fill(scipy.ndimage.sobel(proj.as_array(), axis=0, mode='reflect', cval=0.0))

tilts = np.array([20.0, 22, 24, 26, 28, 28.5, 29, 29.2, 29.4, 29.6, 29.8, 30.0, 30.2, 30.4, 30.6, 30.8, 31., 31.5, 32, 34, 36., 38, 40.0 ])
# tilts = np.arange(28, 33, 2)


# recons_v = np.zeros((len(tilts),256,256))
# recons_v = DataContainer(recons_v, dimension_labels=('tilt', 'horizontal_y', 'horizontal_x'))
# recons_h = np.zeros((len(tilts),256,256))
# recons_h = DataContainer(recons_h, dimension_labels=('tilt', 'vertical', 'horizontal_x'))
# recons_v_filtered = np.zeros((len(tilts),256,256))
# recons_v_filtered = DataContainer(recons_v_filtered, dimension_labels=('tilt', 'horizontal_y', 'horizontal_x'))
# recons_h_filtered = np.zeros((len(tilts),256,256))
# recons_h_filtered = DataContainer(recons_h_filtered, dimension_labels=('tilt', 'vertical', 'horizontal_x'))

residuals = np.zeros((len(tilts),256,360,256))
residuals = DataContainer(residuals, dimension_labels=('tilt', 'vertical', 'angle', 'horizontal'))

residuals_filtered = np.zeros((len(tilts),256,360,256))
residuals_filtered = DataContainer(residuals_filtered, dimension_labels=('tilt', 'vertical', 'angle', 'horizontal')) 

for i, tilt in enumerate(tilts):
    ag_tilt = ag.copy()

    rotation_matrix = R.from_rotvec(np.radians(tilt) * tilt_direction)
    rotation_axis = rotation_matrix.apply(untilted_rotation_axis)
    ag_tilt.config.system.rotation_axis.direction = rotation_axis
    print(ag_tilt.config.system.rotation_axis.direction)

    ig = ag_tilt.get_ImageGeometry()
    # ig.voxel_num_z = 1 

    # Unfiltered
    fbp = FBP(ig, ag_tilt)
    fbp.set_input(proj)
    recon = fbp.get_output()
    # recons_v.array[i,:,:] = recon.array[128,:,:]
    # recons_h.array[i,:,:] = recon.array[:,128,:]
    # recon.apply_circular_mask(0.9)

    A = ProjectionOperator(ig, ag_tilt)
    reprojection = A.direct(recon)
    residual = reprojection - proj
    residuals.array[i,:,:,:] = residual.array[:,:,:]

    # show2D([data, reprojection, residual, residual**2], num_cols=4,
    #        title=['Data', 'Reprojection', 'Residual', 'L2 Norm Residual'], 
    #        fix_range=[(-0.3, 1), (-0.3, 1), (-0.5, 0), (0, 0.25)])
    
    loss = (residual ** 2).sum()
    losses.append(loss)

    # Filtered
    fbp = FBP(ig, ag_tilt)
    fbp.set_input(data_filtered)
    recon_filtered = fbp.get_output()
    # recons_v_filtered.array[i,:,:] = recon_filtered.array[128,:,:]
    # recons_h_filtered.array[i,:,:] = recon_filtered.array[:,128,:]
    # recon_filtered.apply_circular_mask(0.9)

    A = ProjectionOperator(ig, ag_tilt)
    reprojection = A.direct(recon_filtered)
    residual = reprojection - data_filtered

    residuals_filtered.array[i,:,:,:] = residual.array[:,:,:]

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
plt.plot(tilts, A-B)
plt.xlabel("Tilt angle (degrees)")
plt.ylabel("Filtered projection residual (sum of squares)")
plt.grid()
plt.plot(tilts[np.argmin(A-B)], np.min(A-B), 'rx')

# %% Look at reconstruction in the horizontal direction
islicer(recons_h)
# %%
show2D(recons_h, slice_list=[('tilt',0), ('tilt',11), ('tilt',22)],
       title=['tilt = 20','tilt = 30', 'tilt = 40'],
       num_cols=3)

# %% Look at reconstruction in the vertical direction
islicer(recons_v)
# %%
show2D(recons_v, slice_list=[('tilt',0), ('tilt',11), ('tilt',22)],
       title=['tilt = 20','tilt = 30', 'tilt = 40'],
       num_cols=3)

# %% Look at the filtered reconstruction in the horizontal direction
islicer(recons_h_filtered)
# %%
show2D(recons_h_filtered, slice_list=[('tilt',0), ('tilt',11), ('tilt',22)],
       title=['tilt = 20','tilt = 30', 'tilt = 40'],
       num_cols=3)
# %% Look at the filtered reconstruction in the vertical direction
islicer(recons_v_filtered)
# %%
show2D(recons_v_filtered, slice_list=[('tilt',0), ('tilt',11), ('tilt',22)],
       title=['tilt = 20','tilt = 30', 'tilt = 40'],
       num_cols=3)

# %% Look at the residuals
for i in [0, 11, 22]:
    tmp = DataContainer(residuals.array[i], dimension_labels = ('vertical', 'angle', 'horizontal'))
    show2D(tmp,
        slice_list=ag_slice_list,
        num_cols=5)     
# %% Look at the mean of the residuals
for i in [0, 11, 22]:
    tmp = DataContainer(residuals.array[i], dimension_labels = ('vertical', 'angle', 'horizontal'))
    show2D(tmp.mean(axis=1))
# %% Look at the 
for i in [0, 11, 22]:
    tmp = DataContainer(residuals.array[i]**2, dimension_labels = ('vertical', 'angle', 'horizontal'))
    show2D(tmp,
        slice_list=ag_slice_list,
        num_cols=5)
# %%
for i in [0, 11, 22]:
    tmp = DataContainer(residuals.array[i]**2, dimension_labels = ('vertical', 'angle', 'horizontal'))
    show2D(tmp.mean(axis=1))
# %%
for i in [0, 11, 22]:
    tmp = DataContainer(residuals_filtered.array[i], dimension_labels = ('vertical', 'angle', 'horizontal'))
    show2D(tmp,
        slice_list=ag_slice_list,
        num_cols=5)
# %%
for i in [0, 11, 22]:
    tmp = DataContainer(residuals_filtered.array[i], dimension_labels = ('vertical', 'angle', 'horizontal'))
    show2D(tmp.mean(axis=1))

# %%
residual = data_filtered - reprojection
loss1 = (residual ** 2).sum()
print(loss1)

from cil.optimisation.functions import L2NormSquared
F = L2NormSquared(b=reprojection)
loss2 = F(data_filtered)
print(loss2)

print(100*(loss1-loss2)/(loss1+loss2))

# %%
from scipy.optimize import minimize_scalar


def loss_function(tilt):

    ag_tilt = ag.copy()

    rotation_matrix = R.from_rotvec(np.radians(tilt) * tilt_direction)
    rotation_axis = rotation_matrix.apply(untilted_rotation_axis)
    ag_tilt.config.system.rotation_axis.direction = rotation_axis
    print(ag_tilt.config.system.rotation_axis.direction)

    ig = ag_tilt.get_ImageGeometry()
    # ig.voxel_num_z = 1 

    # Unfiltered
    fbp = FBP(ig, ag_tilt)
    fbp.set_input(proj)
    recon = fbp.get_output()

    A = ProjectionOperator(ig, ag_tilt)
    reprojection = A.direct(recon)
    residual = reprojection - proj
    loss_unfiltered = (residual ** 2).sum()
    
    # Filtered
    fbp = FBP(ig, ag_tilt)
    fbp.set_input(data_filtered)
    recon_filtered = fbp.get_output()

    A = ProjectionOperator(ig, ag_tilt)
    reprojection = A.direct(recon_filtered)
    residual = reprojection - data_filtered

    loss_filtered = (residual ** 2).sum()
    return loss_filtered


result = minimize_scalar(loss_function, bounds=(25, 35), method='bounded')


# %%
