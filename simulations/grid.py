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

def create_grid(shape= (64, 256, 256)):
    # Create a circular grid phantom in a volume with shape (Z, Y, X)
    
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

    return lines, ig
# %%
shape =  (64, 256, 256)
lines, ig = create_grid(shape)
show2D(lines,
       slice_list=[
           (0, shape[0]//2),  
           (1, 100),  
           (2, 100), 
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
s