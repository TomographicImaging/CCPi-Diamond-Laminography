# Test creating a sphere (or cube) volume and forward projecting it with CIL
# Test what happens as you change the tilt angle
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


from cil.framework import ImageGeometry, ImageData, AcquisitionGeometry
from cil.plugins.astra.operators import ProjectionOperator
from cil.plugins.astra.processors import FBP
from cil.utilities.display import show1D, show2D, show_geometry
from cil.utilities.jupyter import islicer

import sys, os
sys.path.append(os.path.abspath('../utils'))
from sphere_fitting import find_circles, fit_circles

# %%
nx, ny, nz = (500, 500, 500)
dx, dy, dz = (1, 1, 1)

x = (np.arange(nx) - nx / 2 + 0.5) * dx
y = (np.arange(ny) - ny / 2 + 0.5) * dy
z = (np.arange(nz) - nz/2 + 0.5) * dz
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # shape (nx, ny)

theta = 11
x_rot = X * np.cos(theta) + Y * np.sin(theta)
y_rot = -X * np.sin(theta) + Y * np.cos(theta)
# %%
tilt = 0 # degrees

# ['vertical', 'horizontal_y', 'horizontal_x']
tilt_direction = np.array([1, 0, 0])
untilted_rotation_axis = np.array([0, 0, 1]) # detector up vector
beam_direction = np.array([0, 1, 0])


rotation_matrix = R.from_rotvec(np.radians(tilt) * tilt_direction)
tilted_rotation_axis = rotation_matrix.apply(untilted_rotation_axis)

tilt_the_sample = False # physically rotate the sample, different from if we tilt the rotation axis later
include_cube = False
include_sphere = True

sphere_radius = 80
# cube_half_edge = 60
cube_half_dims = np.array([30, 150, 150])

size = 500
sphere_offset = np.array([0, 0, 0])
cube_offset = np.array([-0, -0, 0])

centre = (size/2)

x, y, z = np.indices((size, size, size))
coords = np.stack((x, y, z), axis=-1).reshape(-1, 3)  
coords = coords - centre
if tilt_the_sample:
    coord_tilt_vector = np.array([0, 0, -1])
    coord_rotation_matrix = R.from_rotvec(np.radians(tilt) * coord_tilt_vector)
    coords = coords @ coord_rotation_matrix.as_matrix()

volume = np.zeros(coords.shape[0], dtype=np.float32)
if include_sphere:
    sphere_coords = coords - sphere_offset
    sphere_dist = np.linalg.norm(sphere_coords, axis=1)
    r = (3 / (4 * np.pi)) ** (1/3)  # sphere volume
    delta = np.clip(sphere_radius - sphere_dist, -r, r) / r
    sphere_mask = 0.5 + 0.5 * np.sin(0.5 * np.pi * delta)
    volume += sphere_mask

if include_cube:
    cube_coords = coords - cube_offset
    max_dist_inside = cube_half_dims - np.abs(cube_coords)
    min_inside = np.min(max_dist_inside, axis=1)

    r = (3 / (4 * np.pi)) ** (1/3)  # Approx. 0.620, effective smoothing range
    delta = np.clip(min_inside, -r, r) / r
    cube_mask = 0.5 + 0.5 * np.sin(0.5 * np.pi * delta)
    volume += cube_mask

volume = volume.reshape((size, size, size))

if include_cube:
    rotated_cube_center = rotation_matrix.apply(cube_offset)
    cube_plot_offsets = np.round(centre + rotated_cube_center).astype(np.int32)
    show2D(volume, slice_list = [(0, cube_plot_offsets[0]), (1, cube_plot_offsets[1]), (2, cube_plot_offsets[2])], num_cols=3)
if include_sphere:
    rotated_sphere_center = rotation_matrix.apply(sphere_offset)
    sphere_plot_offsets = np.round(centre + rotated_sphere_center).astype(np.int32)
    show2D(volume, slice_list = [(0, sphere_plot_offsets[0]), (1, sphere_plot_offsets[1]), (2, sphere_plot_offsets[2])], num_cols=3)
# %%
ig = ImageGeometry(size, size, size)
ig.set_labels(['vertical', 'horizontal_y', 'horizontal_x'])
image = ImageData(volume, geometry=ig)
ig = image.geometry

if include_cube:
    show2D(image, slice_list=[('horizontal_x', cube_plot_offsets[0]),
                    ('vertical', cube_plot_offsets[1]),
                    ('horizontal_y', cube_plot_offsets[2])],
                    num_cols=3)
if include_sphere:
    show2D(image, slice_list=[('horizontal_x', sphere_plot_offsets[0]),
                    ('vertical', sphere_plot_offsets[1]),
                    ('horizontal_y', sphere_plot_offsets[2])],
                    num_cols=3)

# %% 

ag = AcquisitionGeometry.create_Parallel3D(rotation_axis_direction=tilted_rotation_axis)
ag.set_panel([size,size],
             [(1/1000), (1/1000)])
ag.set_angles(np.arange(0, 360, 1))
ag.set_labels(['vertical','angle','horizontal'])
# show_geometry

ig = ag.get_ImageGeometry()
show_geometry(ag, ig)

A = ProjectionOperator(ig, ag)
projections = A.direct(image)
show2D(projections, slice_list=[('angle', 0), ('angle', 45), ('angle',90), ('angle', 135), ('angle',180)], num_cols=5)
# islicer(projections, direction='angle')
# %% Back project
recon = A.adjoint(projections)

if include_cube:
    show2D(recon, slice_list = [('horizontal_x', cube_plot_offsets[0]), ('vertical', cube_plot_offsets[1]), ('horizontal_y', cube_plot_offsets[2])], num_cols=3)
if include_sphere:
    show2D(recon, slice_list = [('horizontal_x', sphere_plot_offsets[0]), ('vertical', sphere_plot_offsets[1]), ('horizontal_y', sphere_plot_offsets[2])], num_cols=3)
# %%
from cil.recon import FBP
fbp = FBP(projections, ig, backend='astra')
# fbp.set_input(projections)
recon = fbp.run()

if include_cube:
    show2D(recon, slice_list = [('horizontal_x', cube_plot_offsets[0]), ('vertical', cube_plot_offsets[1]), ('horizontal_y', cube_plot_offsets[2])], num_cols=3)
if include_sphere:
    show2D(recon, slice_list = [('horizontal_x', sphere_plot_offsets[0]), ('vertical', sphere_plot_offsets[1]), ('horizontal_y', sphere_plot_offsets[2])], num_cols=3)
# %%

show1D(recon, slice_list=[('horizontal_x', cube_plot_offsets[0]), ('horizontal_y', cube_plot_offsets[2])])
# %% Fit spheres to calculate the tilt direction
# tilt direction is 100 so calculate from vertical and horizontal_y
centre_slice = recon.get_slice(horizontal_x=sphere_plot_offsets[0])
hough_res, hough_radii = find_circles(centre_slice.array,sigma=4, output=False)
res = fit_circles(centre_slice.array, hough_res, hough_radii, False)

vertical = res[1]
horizontal_y = res[0]
print('vertical = ' + str(res[1]))
print('horizontal_y = ' + str(res[0]))

# confirm
centre_slice = recon.get_slice(vertical=sphere_plot_offsets[1])
hough_res, hough_radii = find_circles(centre_slice.array,sigma=4, output=False)
res = fit_circles(centre_slice.array, hough_res, hough_radii, False)
print('horizontal_x = ' + str(res[0]))
print('horizontal_y = ' + str(res[1]))

centre_slice = recon.get_slice(horizontal_y=sphere_plot_offsets[2])
hough_res, hough_radii = find_circles(centre_slice.array,sigma=4, output=False)
res = fit_circles(centre_slice.array, hough_res, hough_radii, False)
print('vertical = ' + str(res[1]))
print('horizontal_x = ' + str(res[0]))
# %%
# tilt direction is 100 so calculate from vertical and horizontal_y
from numpy.linalg import norm
p0_yz = centre+sphere_offset[1:3]
p1_yz = np.array([vertical, horizontal_y])
v0 = p0_yz / norm(p0_yz)
v1 = p1_yz / norm(p1_yz)

cos_theta = np.dot(v0, v1)
angle_rad = np.arccos(cos_theta)
angle_deg = np.degrees(angle_rad)
print(angle_deg)



# %%
slice_img = recon.array[:, 250,:]
plt.imshow(slice_img)
mask = slice_img > 0.88
ys, xs = np.where(mask)  
points = np.column_stack([xs, ys])
plt.plot(xs, ys, '.k')
# %%
mask = slice_img > 0.88
ys, xs = np.where(mask)  
points = np.column_stack([xs, ys])
plt.plot(xs, ys, 'o')
params = fit_ellipse(xs, ys)
xc, yc, a, b, theta = params

ellipse = Ellipse((xc, yc), width=2*a, height=2*b, angle=np.degrees(theta),
                    edgecolor='red', facecolor='none', linewidth=2, label='Fitted ellipse', zorder=3)
plt.gca().add_patch(ellipse)
plt.gca().set_aspect('equal')

# %%
centre_x = np.zeros(projections.get_dimension_size('angle'))
centre_y = np.zeros(projections.get_dimension_size('angle'))
for i in np.arange(projections.get_dimension_size('angle')):
    proj = projections.array[:,i,:]
    hough_res, hough_radii = find_circles(proj,sigma=4, output=False)
    cx, cy, radii = fit_circles(proj, hough_res, hough_radii, False)
    # plt.imshow(proj)
    # plt.plot(cx[0], cy[0],'rx')
    centre_x[i] = cx[0]
    centre_y[i] = cy[0]
plt.plot(centre_x, centre_y, 'o')


params = fit_ellipse(centre_x, centre_y)
xc, yc, a, b, theta = params

plt.plot(centre_x, centre_y, 'o')

ellipse = Ellipse((xc, yc), width=2*a, height=2*b, angle=np.degrees(theta),
                    edgecolor='red', facecolor='none', linewidth=2, label='Fitted ellipse', zorder=3)
plt.gca().add_patch(ellipse)

# %% FBP
fbp = FBP(ig, ag)
fbp.set_input(projections)
recon = fbp.get_output()
# %%
show2D(recon, slice_list=[('horizontal_x',200,),
                   ('vertical',200),
                   ('horizontal_y', 250)],
                   num_cols=3)

# %% Back project
image2 = A.adjoint(projections)
show2D(image2)