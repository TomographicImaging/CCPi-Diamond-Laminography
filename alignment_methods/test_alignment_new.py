# %% Imports
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize, differential_evolution
from scipy.ndimage import gaussian_filter, sobel
import matplotlib.pyplot as plt

from cil.io import NEXUSDataReader
from cil.plugins.astra.operators import ProjectionOperator
from cil.plugins.astra.processors import FBP
from cil.optimisation.algorithms import CGLS, SIRT
from cil.utilities.display import show2D
from cil.processors import Binner

# %%

def update_geometry(ag, tilt_deg, cor_pix, 
        beam_direction=np.array([0, 1, 0]),
        detector_x_direction=np.array([1, 0, 0]),
        detector_y_direction=np.array([0, 0, -1]),
        rotation_axis=np.array([0, 0, 1])
    ):

    tilt_rad = np.deg2rad(tilt_deg)
    rotation_matrix = R.from_rotvec(tilt_rad * detector_x_direction)
    tilted_rotation_axis = rotation_matrix.apply(rotation_axis)

    ag.set_centre_of_rotation(offset=cor_pix, distance_units='pixels')
    ag.config.system.rotation_axis.direction = tilted_rotation_axis
    

def reconstruct(data_in, ig, ag, method, voxel_num_z, x0, niter):
    
    if method.upper() == 'FBP':
        x = FBP(ig, ag)(data_in)
        
    elif method.upper() == 'SIRT':
        A = ProjectionOperator(ig, ag)
        alg = SIRT(initial=x0, operator=A, data=data_in)
        alg.run(niter)
        x = alg.x
    else:
        A = ProjectionOperator(ig, ag)
        alg = CGLS(initial=x0, operator=A, data=data_in)
        alg.run(niter)
        x = alg.x
    
    x.apply_circular_mask(0.9)
    return x

def sobel_2d(arr):
    gx = sobel(arr, axis=-1)
    gy = sobel(arr, axis=-2)
    return np.sqrt(gx**2 + gy**2)

def highpass_2d(arr, sigma=3.0):
    return arr - gaussian_filter(arr, sigma=sigma)

def loss_from_residual(residual,
                       kind='huber',
                       hp_sigma=3.0,
                       use_highpass=True,
                       use_sobel=True,
                       normalize_per_angle=False,
                       huber_delta=1.0):
    r = residual.as_array()

    if use_highpass:
        r = highpass_2d(r, sigma=hp_sigma)
    if use_sobel:
        r = sobel_2d(r)

    if normalize_per_angle and r.ndim >= 2:
        ang_axis = 0
        eps = 1e-8
        norms = np.sqrt(np.sum(r**2, axis=tuple(range(1, r.ndim)), keepdims=True)) + eps
        r = r / norms

    if kind.upper() == 'L2':
        return float(np.sum(r**2))
    if kind.upper() == 'L1':
        return float(np.sum(np.abs(r)))
    if kind.lower() == 'huber':
        a = np.abs(r)
        return float(np.sum(np.where(a <= huber_delta, 0.5*a*a, huber_delta*(a - 0.5*huber_delta))))
    raise ValueError(f"Unknown loss kind: {kind}")

def geom_loss(data, ag, tilt_deg, cor_pix, x0,
                     recon_method,
                     recon_iters, voxel_num_z,
                     loss_kind):
    update_geometry(ag, tilt_deg, cor_pix)
    
    ig = ag.get_ImageGeometry()
    ig.voxel_num_z = voxel_num_z
    x = reconstruct(data, ig, ag, recon_method, voxel_num_z, x0, recon_iters)
    
    A = ProjectionOperator(ig, ag)
    yhat = A.direct(x)
    r = yhat - data
    
    loss = loss_from_residual(r, kind=loss_kind)
    
    return loss, x

def linear_geometry_scan(data, ag, tilt_vals, cor_vals, binning,
                         voxel_num_z=256,
                         recon_method='FBP', recon_iters=None, 
                         loss_kind='L2', 
                         save_slices=False):

    ig = ag.get_ImageGeometry()
    
    losses = np.zeros((len(tilt_vals), len(cor_vals)))
    if save_slices:
        slices = np.zeros((len(tilt_vals), len(cor_vals), ig.shape[1], ig.shape[2]))
    
    binned_cor_vals = cor_vals/binning
    for i, tilt in enumerate(tilt_vals):
        for j, cor in enumerate(binned_cor_vals):
            x0 = ig.allocate(0)
            loss, x = geom_loss(data, ag, tilt, cor, x0, recon_method, recon_iters, voxel_num_z, loss_kind)
        
            losses[i, j] = loss
            
            if save_slices:
                slices[i, j, :, :]  = x.as_array()[x.shape[0]//2]
                
            print(f"[Scan] tilt: {tilt:.3f}, cor: {cor*binning:.3f}, loss: {loss:.6e}")
    
    if save_slices:
        return losses, slices
    else:
        return losses
    
def plot_losses(losses, tilt_vals, cor_vals):
    plt.figure(figsize=(6, 4))
    plt.imshow(losses, aspect='equal', origin='lower',
            extent=[cor_vals[0], cor_vals[-1], tilt_vals[0], tilt_vals[-1]])
    plt.xlabel('Centre of rotation (pixels)')
    plt.ylabel('Rotation axis tilt angle (degrees)')

    plt.colorbar(label='Filtered projection residual (sum of squares)')

def plot_losses_2D(losses, tilt_vals, cor_vals):

    i, j = np.unravel_index(np.argmin(losses), losses.shape)

    fig, axes = plt.subplots(1, 2, figsize=(8,4))
    ax = axes[0]
    ax.plot(cor_vals, losses[i,:] )
    ax.set_xlabel('Centre of rotation offset (pixels)')
    ax.set_ylabel('Loss function value')
    ax.grid()

    ax = axes[1]
    ax.plot(tilt_vals, losses[:,j] )
    ax.set_xlabel('Rotation axis tilt (degrees)')
    ax.set_ylabel('Loss function value')
    ax.grid()

    plt.tight_layout()

def plot_slice(slices, tilt_vals, cor_vals, tilt, cor):

    tilt_idx = np.argmin(np.abs(tilt_vals - tilt))
    cor_idx  = np.argmin(np.abs(cor_vals - cor))

    slice_array = slices[tilt_idx, cor_idx]
    plt.figure(figsize=(5,5))
    plt.imshow(slice_array, cmap='gray')
    plt.colorbar()
    title = f"Slice at tilt={tilt_vals[tilt_idx]}, cor={cor_vals[cor_idx]}"
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_slices(slices, tilt_vals, cor_vals):
    n_tilt = len(tilt_vals)
    n_cor = len(cor_vals)
    fig, axes = plt.subplots(n_tilt, n_cor, figsize=(2*n_cor, 2*n_tilt))

    vmin = slices.min()
    vmax = slices.max()

    for i, tilt in enumerate(tilt_vals):
        for j, cor in enumerate(cor_vals):
            ax = axes[i, j]
            ax.imshow(slices[i, j], cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(f"t={tilt:.1f}, c={cor:.1f}", fontsize=8)
            ax.axis('off')


# %% Load data
data = NEXUSDataReader(file_name='../output_data/cylinder_tilt_30_cor_offset_5.nxs').read()
ag = data.geometry
ig = ag.get_ImageGeometry()
data.reorder('astra')
fbp = FBP(ig, ag)
recon = fbp(data)

show2D(recon)

# %% subsample data
binning = 4
roi = {
        'horizontal': (None, None, binning),
        'vertical': (None, None, binning),
        'angle': (None, None, 1)
    }
data_binned = Binner(roi)(data)
ag_binned = data_binned.geometry
ig_binned = ag_binned.get_ImageGeometry()
recon = FBP(ig_binned, ag_binned)(data_binned)
# x_fixed.apply_circular_mask(0.9)

# %% Course linear search
tilt_vals = np.linspace(26, 34, 5)
cor_vals = np.linspace(1, 9, 5) # un-binned range
losses, slices = linear_geometry_scan(data_binned, ag_binned, tilt_vals, cor_vals, binning,
                         voxel_num_z=256,
                         recon_method='FBP', recon_iters=None, 
                         loss_kind='L2', 
                         save_slices=True)
# %%
plot_losses(losses, tilt_vals, cor_vals)
plot_losses_2D(losses, tilt_vals, cor_vals)
# %%
# plot all the slices
plot_slices(slices, tilt_vals, cor_vals)

# plot the best slice
i, j = np.unravel_index(np.argmin(losses), losses.shape)
tilt_centre = tilt_vals[i]
cor_centre = cor_vals[j]
plot_slice(slices, tilt_vals, cor_vals, tilt_centre, cor_centre)

# %% Fine linear search

# perhaps we can do something fancy to choose the fine search range?
grad_tilt = (losses[min(i+1, len(tilt_vals)-1), j] - losses[max(i-1, 0), j]) / (tilt_vals[min(i+1, len(tilt_vals)-1)] - tilt_vals[max(i-1, 0)])
grad_cor  = (losses[i, min(j+1, len(cor_vals)-1)] - losses[i, max(j-1, 0)]) / (cor_vals[min(j+1, len(cor_vals)-1)] - cor_vals[max(j-1, 0)])

# for now just define a range and check it isn't too long
tilt_precision = 0.01
tilt_range = 0.05 # plus or minus from course centre
tilt_vals = np.arange(tilt_centre - tilt_range, tilt_centre + tilt_range, tilt_precision)
print(len(tilt_vals))

cor_precision = 0.005
cor_range = 0.025 # plus or minus from course centre
cor_vals = np.arange(cor_centre - cor_range, cor_centre + cor_range, cor_precision)
print(len(cor_vals))
# %%
binning = 1
losses, slices = linear_geometry_scan(data, ag, tilt_vals, cor_vals, binning,
                         voxel_num_z=256,
                         recon_method='FBP', recon_iters=None, 
                         loss_kind='L2', 
                         save_slices=True)

# %%
plot_losses(losses, tilt_vals, cor_vals)
plot_losses_2D(losses, tilt_vals, cor_vals)
# %%
# plot all the slices
plot_slices(slices, tilt_vals, cor_vals)

# plot the best slice
i, j = np.unravel_index(np.argmin(losses), losses.shape)
tilt_centre = tilt_vals[i]
cor_centre = cor_vals[j]
plot_slice(slices, tilt_vals, cor_vals, tilt_centre, cor_centre)

# next adaptation: do linear search of tilt, then cor, then tilt
# %%

# %%
tilt = 30
cor = 1
update_geometry(ag_binned, tilt, cor)
ig = ag_binned.get_ImageGeometry()
ig.voxel_num_z = 256
recon = FBP(ig, ag_binned)(data_binned)
central_slice = recon.as_array()[recon.shape[0]//2]
show2D(central_slice, title=f"FBP with optimised geometry: tilt={tilt:.3f}, CoR={cor:.3f}")


# %%Run joint recon

# def joint_geometry_reconstruction(data, ag, p0=(30.0, 5.0), binning=1,
#                                   voxel_num_z=256,
#                                   recon_method ='FBP', recon_iters = None,
#                                   loss_kind='L2', min_method='Powell', maxiter=30,
#                                   bounds=[(0,100),(0,100)]):
#     ig = ag.get_ImageGeometry()
#     ig.voxel_num_z = voxel_num_z

#     x0 = [ig.allocate(0)]
#     p0_binned = (p0[0], p0[1]/binning)
#     bounds_binned = (bounds[0], (bounds[1][0]/binning, bounds[1][1]/binning))

#     def geom_loss_wrapper(p):
#         tilt, cor = p
#         loss, x = geom_loss(data, ag, tilt, cor, x0[0], recon_method, recon_iters, voxel_num_z, loss_kind)
        
#         x0[0] = x  # warm start for next iteration, if using iterative recon
        
#         print(f"tilt: {tilt:.3f}, cor: {cor*binning:.3f}, loss: {loss:.6e}")
#         return loss

#     res = minimize(geom_loss_wrapper, np.asarray(p0_binned, float),
#                    method=min_method,
#                    bounds=bounds_binned,
#                    options={'maxiter': maxiter, 'disp': True})
#     return res.x, res.fun
# bounds = [(25.0, 35.0), (4.0, 6.0)]

# p_opt, loss = joint_geometry_reconstruction(data_binned, ag_binned, binning=4,
#                                     p0=(30.0, 5.0),
#                                     recon_iters=15,
#                                     voxel_num_z=256,
#                                     bounds=bounds)
# print("Optimised tilt, CoR =", p_opt)

# %% 
# x_best, p_best = joint_recon(data, ag, x0=None, p0=(30.0, 5.0),
#                      loss_kind='L2', voxel_num_z=256, max_nfev=20)

# print("Final params:",  p_best)

# # %% Run alternating reconstruction
# x_best, p_best, hist = alternating_reconstruction(
#     data=data_binned,
#     ag=ag,
#     p0=(30.0, 5.0),
#     n_outer=6,
#     recon_iters=20,          
#     recon_method='CGLS',      
#     use_filtered=True,        
#     loss_kind='L2',
#     geom_method='powell',     
#     voxel_num_z=256
# )

# print("Final params:",  p_best)

# ag_final = update_geometry(ag, p_best[0], p_best[1])
# x_final, _ = reconstruct_iterative(data_binned, ag_final, niter=60, method='CGLS', voxel_num_z=512)
    
# show2D(x_final)


# plt.figure(); plt.plot(hist['loss'], '-o'); plt.xlabel('Outer iter'); plt.ylabel('Loss'); plt.grid(True)
