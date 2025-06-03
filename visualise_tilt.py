# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
 #%%
 
def plot_tilt(tilt_deg, ax):

    tilt_rad = np.deg2rad(tilt_deg)

    ray_positions = np.linspace(-1/2, 1/2, num_rays)
    thetas = np.linspace(0, 2*np.pi, num_projections, endpoint=False)
    rot_axis = np.array([0, np.sin(tilt_rad), np.cos(tilt_rad)])


    for theta in thetas:
        Rx = R.from_rotvec(rot_axis*theta)
        ref_vec = np.array([1, 0, 0])
        if np.allclose(rot_axis, ref_vec / np.linalg.norm(ref_vec)):
            ref_vec = np.array([0, 1, 0])
        beam_dir = np.cross(rot_axis, ref_vec)
        beam_dir /= np.linalg.norm(beam_dir)

        detector_dir = np.cross(beam_dir, rot_axis)
        detector_dir /= np.linalg.norm(detector_dir)

        for offset in ray_positions:
            detector_offset = offset * detector_dir
            center_point = detector_offset

            p1 = center_point - 3 * beam_dir
            p2 = center_point + 3 * beam_dir

            p1_rot = Rx.apply(p1)
            p2_rot = Rx.apply(p2)

            ax.plot([p1_rot[0], p2_rot[0]], [p1_rot[2], p2_rot[2]], 'b-', alpha=0.5)

    scale = 2.5
    rot_axis_line = np.array([rot_axis * -scale, rot_axis * scale])
    ax.plot(rot_axis_line[:,0], rot_axis_line[:,2], 'k--', lw=2, label='Tilted rotation axis')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title(f'Rotation Axis Tilt {tilt_deg}°')
    ax.set_aspect('equal')
    ax.grid(True)
# %%

num_projections = 12
num_rays = 20
tilt_deg = 0


fig, axs = plt.subplots(3,1, figsize=(8,8))

for i, tilt_deg in enumerate([0, 30, 90]):

    plot_tilt(tilt_deg, axs[i])

plt.tight_layout()


# %% 

theta_deg = -30
theta = np.radians(theta_deg)

tilt_direction = np.array([1, 0, 0])

rotation_matrix = R.from_rotvec(theta * tilt_direction)
XY_vector = np.array([0, 0, 1])
XZ_vector = np.array([0, 1, 0])
YZ_vector = np.array([1, 0, 0])
tilted_XY_vector = rotation_matrix.apply(XY_vector)
tilted_XZ_vector = rotation_matrix.apply(XZ_vector)
tilted_YZ_vector = rotation_matrix.apply(YZ_vector)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.azim = 10

# axis_vector = np.array([1, 0, 0])
# tilted_axis_vector = np.array([0, np.sin(theta),  np.cos(theta)])
# print(tilted_axis_vector)

def get_plane(vector, tilted_vector):

    v1 = vector - np.dot(vector, tilted_vector) * tilted_vector
    v1 /= np.linalg.norm(v1)

    v2 = np.cross(tilted_vector, v1)
    return v1, v2

v1, v2 = get_plane(XY_vector, tilted_XY_vector)

s = np.linspace(-1, 1, 10)
t = np.linspace(-1, 1, 10)
S, T = np.meshgrid(s, t)
X = S * v1[0] + T * v2[0]
Y = S * v1[1] + T * v2[1]
Z = S * v1[2] + T * v2[2]
ax.plot_surface(X, Y, Z, alpha=0.3, color='cyan', edgecolor='gray')

v1, v2 = get_plane(XZ_vector, tilted_XZ_vector)
X = S * v1[0] + T * v2[0]
Y = S * v1[1] + T * v2[1]
Z = S * v1[2] + T * v2[2]
ax.plot_surface(X, Y, Z, alpha=0.3, color='cyan', edgecolor='gray')

X = S * 0 + T * 0
Y = S * 0 + T * 1
Z = S * 1 + T * 0
ax.plot_surface(X, Y, Z, alpha=0.3, color='cyan', edgecolor='gray')

origin = np.array([0, 0, 0])
ax.quiver(*origin, *tilted_XY_vector, length=1.5, color='red', linewidth=2)

u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)
radius = 0.8
sphere_x = radius*np.outer(np.cos(u), np.sin(v))
sphere_y = radius*np.outer(np.sin(u), np.sin(v))
sphere_z = 0.2*np.outer(np.ones_like(u), np.cos(v))

ax.plot_surface(sphere_x, sphere_y, sphere_z, color='orange', alpha=0.7, edgecolor='k')

angles = np.linspace(0, 2 * np.pi, 100)
x = (radius+0.1) * np.cos(angles)
y = (radius+0.1) * np.sin(angles)
zero = np.zeros(np.shape(x))

A = rotation_matrix.apply(np.array([x, y, zero]).T)



# ax.scatter(A[:,0], A[:,1], A[:,2], )

# ax.scatter(zero, x, y, c='r')
# ax.scatter(x, zero, y, c='r')
# ax.scatter(x, y, zero, c='r')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
ax.set_box_aspect([1, 1, 1]) 
origin = np.array([0, 0, 0])

