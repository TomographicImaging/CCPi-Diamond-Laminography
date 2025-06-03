# %%
data_filtered = data_norm.copy()
data_filtered.fill(scipy.ndimage.sobel(data_norm.as_array(), axis=0, mode='reflect', cval=0.0))
# %%
data_norm.reorder('astra')
data_filtered.reorder('astra')
ag = data_norm.geometry
ig = ag.get_ImageGeometry()
# %%
from cil.plugins.astra.processors import FBP
offset = 0

recon_filtered_list = []
recon_list = []
tilts = np.arange(25, 35.05, 1)

evaluation = np.zeros(len(tilts))
evaluation_filtered = np.zeros(len(tilts))
for i, tilt in enumerate(tilts):
    ag_tilt = ag.copy()
    ag_tilt.set_centre_of_rotation(offset=offset)

    rotation_matrix = R.from_rotvec(np.radians(tilt) * axis_to_apply_tilt)
    rotation_axis = np.array(gvxr.getDetectorUpVector())
    rotation_axis = rotation_matrix.apply(rotation_axis)

    ag_tilt.config.system.rotation_axis.direction = rotation_axis
    ig_tilt = ag_tilt.get_ImageGeometry()
    ig_tilt.voxel_num_z = 1

    fbp = FBP(ig_tilt, ag_tilt)
    fbp.set_input(data_filtered)
    recon_filtered = fbp.get_output()
    recon_filtered.apply_circular_mask(0.9)
    recon_filtered_list.append(recon_filtered.array)
    evaluation_filtered[i] = (recon_filtered*recon_filtered).sum()

    fbp.set_input(data_norm)
    recon = fbp.get_output()
    recon.apply_circular_mask(0.9)
    recon_list.append(recon.array)

    evaluation[i] = (recon*recon).sum()
# %%
from cil.framework import DataContainer
DC_filtered = DataContainer(np.stack(recon_filtered_list, axis=0), dimension_labels=('Tilt',) + recon_filtered.geometry.dimension_labels)
DC_recon = DataContainer(np.stack(recon_list, axis=0), dimension_labels=('Tilt',) + recon.geometry.dimension_labels)
# %%
islicer(DC_filtered)
# %%
[fig, axs] = plt.subplots(1,2, figsize=(10,3))
ax = axs[0]
ax.plot(tilts, evaluation)
ax.grid()
ax.set_xlabel('Tilt')
ax.set_ylabel('Reconstruction sum of squares')

ax = axs[1]
ax.plot(tilts, evaluation_filtered)
ax.grid()
ax.set_xlabel('Tilt')
ax.set_ylabel('Filtered reconstruction sum of squares')

 # %%
# plt.plot(A)
# plt.plot(B)
A =  (evaluation - evaluation.min()) / (evaluation.max() - evaluation.min())
B =  (evaluation_filtered - evaluation_filtered.min()) / (evaluation_filtered.max() - evaluation_filtered.min())
# plt.plot(tilts, A)
# plt.plot(tilts, B)
plt.plot(tilts, A-B)
plt.plot(tilts[(A-B).argmax()], (A-B)[(A-B).argmax()], 'rx')