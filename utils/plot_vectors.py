# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_axes(ax, origin, directions, length=1.0, labels=['X', 'Y', 'Z'], color=['r', 'g', 'b']):
    for i in range(3):
        vec = directions[:, i] * length
        ax.quiver(*origin, *vec, color=color[i])
        ax.text(*(origin + vec), labels[i], color=color[i])

# Identity matrix for original axes
origin = np.array([0, 0, 0])
axes = np.eye(3)  # standard basis vectors (X, Y, Z)
# %%
def rotation_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2)
    b, c, d = -axis * np.sin(angle / 2)
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d),     2*(b*d + a*c)],
        [2*(b*c + a*d),     a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c),     2*(c*d + a*b),     a*a + d*d - b*b - c*c]
    ])

# Rotate around Z-axis by 45 degrees
rot_mat = rotation_matrix(np.array([0, 1, 0]), np.deg2rad(-30))
rotated_axes = rot_mat @ axes
# %%
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])
ax.set_box_aspect([1, 1, 1])

# Plot original axes
plot_axes(ax, origin, axes, length=1.0, labels=['X', 'Y', 'Z'], color=['r', 'g', 'b'])

ax = fig.add_subplot(122, projection='3d')
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])
ax.set_box_aspect([1, 1, 1])
# Plot rotated axes
plot_axes(ax, origin, rotated_axes, length=1.0, labels=["X'", "Y'", "Z'"], color=['m', 'c', 'y'])

plt.title("Original and Rotated Axes")
plt.show()
