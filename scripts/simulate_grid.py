import numpy as np
from scipy.spatial.transform import Rotation
from cil.framework import AcquisitionGeometry, ImageGeometry, ImageData
from cil.plugins.astra import ProjectionOperator
from cil.processors import Binner

def create_grid(tilt=0, flip=False, shape = (64, 256, 256)):
    # Create a circular grid phantom in a volume with shape (Y, X, Z)
    shape = (shape[0]*2, shape[1]*2, shape[2]*2)
    volume = np.zeros(shape, dtype=np.float32)

    margin = 32
    center_x, center_z = shape[1] // 2, shape[2] // 2
    radius = min(center_x, center_z) - margin
    edge_thickness = 1
    x_vals = np.arange(margin, shape[1] - margin)
    z_vals = np.arange(margin, shape[2] - margin)
    xx, zz = np.meshgrid(x_vals, z_vals, indexing='ij')  

    r2 = (xx - center_x)**2 + (zz - center_z)**2
    cylinder_mask = r2 <= radius**2
    cylinder_edge_mask = (r2 >= (radius - edge_thickness)**2) & (r2 <= (radius + edge_thickness)**2)

    grid_spacing = 32

    for xi, x in enumerate(x_vals):
        for zi, z in enumerate(z_vals):
            if not cylinder_mask[xi, zi]:
                continue
            if (x % grid_spacing == 0) or (z % grid_spacing == 0):
                volume[margin:shape[0]-margin, x, z] = 1.0

    for xi, x in enumerate(x_vals):
        for zi, z in enumerate(z_vals):
            if cylinder_edge_mask[xi, zi]:
                volume[margin:shape[0]-margin, x, z] = 1.0

    if flip:
        volume = np.rot90(volume, k=1, axes=(0, -1))
        ig = ImageGeometry(voxel_num_x=shape[0], voxel_num_y=shape[1], voxel_num_z=shape[2],
                        dimension_labels = ('horizontal_y', 'vertical', 'horizontal_x'))
        # ig.dimension_labels = ('horizontal_y', 'vertical', 'horizontal_x')
        grid = ImageData(volume, geometry=ig)
        grid.reorder('astra')
    else:
        ig = ImageGeometry(voxel_num_z=shape[0], voxel_num_x=shape[2], voxel_num_y=shape[1])
        ig.dimension_labels = ('vertical', 'horizontal_y', 'horizontal_x')
        grid = ImageData(volume, geometry=ig)

    tilt_rad = np.deg2rad(tilt)
    tilt_axis = np.array([1, 0, 0])
    rotation_axis = np.array([0, 0, 1]) # untilted rotation axis
    rotation_matrix = Rotation.from_rotvec(tilt_rad * tilt_axis)
    tilted_rotation_axis = rotation_matrix.apply(rotation_axis)

    ag = AcquisitionGeometry.create_Parallel3D(rotation_axis_direction=tilted_rotation_axis)\
        .set_angles(np.arange(0,360))\
        .set_panel([shape[1], shape[2]])
    ag.dimension_labels = ('vertical', 'angle','horizontal')

    A = ProjectionOperator(ig, ag)
    proj = A.direct(grid)
    proj.array += 0.01 * proj.max() * np.random.randn(*proj.shape).astype(np.float32)
    proj.array = np.clip(proj.array, 0, None)

    roi = {'horizontal': (None, None, 2),
            'vertical': (None, None, 2),
            'angle': (None, None, 2)}
    proj = Binner(roi=roi)(proj)

    roi = {'horizontal_x': (None, None, 2),
            'vertical': (None, None, 2),
            'horizontal_y': (None, None, 2)}
    grid = Binner(roi=roi)(grid)

    return grid, proj
