# %%
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize, differential_evolution, curve_fit
from scipy.ndimage import gaussian_filter, sobel
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

from cil.framework import Processor
from cil.io import NEXUSDataReader
from cil.plugins.astra.operators import ProjectionOperator
from cil.plugins.astra.processors import FBP
from cil.optimisation.algorithms import CGLS, SIRT
from cil.utilities.display import show2D
from cil.processors import Binner
from cil.framework import AcquisitionGeometry, AcquisitionData



# %%

def update_geometry(ag, tilt_deg, cor_pix, 
        tilt_direction_vector=np.array([1, 0, 0]),
        rotation_axis=np.array([0, 0, 1])
    ):

    tilt_rad = np.deg2rad(tilt_deg)
    rotation_matrix = R.from_rotvec(tilt_rad * tilt_direction_vector)
    tilted_rotation_axis = rotation_matrix.apply(rotation_axis)

    ag.set_centre_of_rotation(offset=cor_pix, distance_units='pixels')
    ag.config.system.rotation_axis.direction = tilted_rotation_axis
    

def reconstruct(data_in, ig, ag, method, voxel_num_z, niter):
    
    if method.upper() == 'FBP':
        x = FBP(ig, ag)(data_in)
        
    elif method.upper() == 'SIRT':
        x0 = ig.allocate(0)
        A = ProjectionOperator(ig, ag)
        alg = SIRT(initial=x0, operator=A, data=data_in)
        alg.run(niter)
        x = alg.x
    else:
        x0 = ig.allocate(0)
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

def gaussian(x, A, x0, sigma, y0):
    return (A-y0)* np.exp(-(x - x0)**2 / (2 * sigma**2)) + y0

def fit_gaussian(x, y, min_index):
    
    xx = np.linspace(x[0], x[-1], 100)
    x0 = x[min_index]
    y0 = np.max(y)
    A = -(np.max(y)-np.min(y))
    sigma = (x[-1] - x[0])/4

    popt, _ = curve_fit(gaussian, x, y, p0=[A, x0, sigma, y0])
    curve = gaussian(xx, *popt)
    
    return xx, curve

def fit_cubic(x, y):
    xx = np.linspace(x[0], x[-1], 100)
    coeffs = np.polyfit(x, y, 3)
    poly = np.poly1d(coeffs)
    yy = poly(xx)

    return xx, yy

def cubic_interp(x, y):
    cubic_interp = interp1d(x, y, kind='cubic')
    xx = np.linspace(x[0], x[-1], 200)
    yy = cubic_interp(xx)
    
    return xx, yy

def compare_fits(x, y):
    fig, axes = plt.subplots(1, 3, figsize=(9,3))

    ax = axes[0]
    ax.plot(x, y)
    xx, yy = cubic_interp(x, y)
    ax.plot(xx, yy, '--r')
    ax.set_xlabel('Centre of rotation offset (pixels)')
    ax.set_ylabel('Loss function value')
    ax.set_title(f'Cubic interpolation:  {xx[np.argmin(yy)]:.6f}')
    ax.grid()

    ax = axes[1]
    ax.plot(x, y)
    xx, yy = fit_cubic(x, y)
    ax.plot(xx, yy, '--r')
    ax.set_xlabel('Centre of rotation offset (pixels)')
    ax.set_ylabel('Loss function value')
    ax.set_title(f'Cubic fit: {xx[np.argmin(yy)]:.6f}')
    ax.grid()

    ax = axes[2]
    ax.plot(x, y)
    xx, yy = fit_gaussian(x, y, np.argmin(y))
    ax.plot(xx, yy, '--r')
    ax.set_xlabel('Centre of rotation offset (pixels)')
    ax.set_ylabel('Loss function value')
    ax.set_title(f'Gaussian fit: {xx[np.argmin(yy)]:.6f}')
    ax.grid()

    plt.tight_layout()

def get_min( offsets, values, ind):
    #calculate quadratic from 3 points around ind  (-1,0,1)
    a = (values[ind+1] + values[ind-1] - 2*values[ind]) * 0.5
    b = a + values[ind] - values[ind-1]
    ind_centre = -b / (2*a)+ind

    ind0 = int(ind_centre)
    w1 = ind_centre - ind0
    return (1.0 - w1) * offsets[ind0] + w1 * offsets[ind0+1]

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

def geom_loss(data, ag, tilt_deg, cor_pix,
                     recon_method,
                     recon_iters, voxel_num_z,
                     loss_kind):
    update_geometry(ag, tilt_deg, cor_pix)
    
    ig = ag.get_ImageGeometry()
    ig.voxel_num_z = voxel_num_z
    x = reconstruct(data, ig, ag, recon_method, voxel_num_z, recon_iters)
    
    A = ProjectionOperator(ig, ag)
    yhat = A.direct(x)
    r = yhat - data
    
    loss = loss_from_residual(r, kind=loss_kind)
    
    return loss, x

def geometry_scan_2D(data, ag, tilt_vals, cor_vals, binning,
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
            loss, x = geom_loss(data, ag, tilt, cor, recon_method, recon_iters, voxel_num_z, loss_kind)
        
            losses[i, j] = loss
            
            if save_slices:
                slices[i, j, :, :]  = x.as_array()[x.shape[0]//2]
                
            print(f"[Scan] tilt: {tilt:.3f}, cor: {cor*binning:.3f}, loss: {loss:.6e}")
    
    if save_slices:
        return losses, slices
    else:
        return losses
    
def geometry_scan_linear(data, ag, tilt_initial, tilt_vals, cor_vals, binning,
                         voxel_num_z=256,
                         recon_method='FBP', recon_iters=None, 
                         loss_kind='L2', 
                         save_slices=False):

    ig = ag.get_ImageGeometry()
    
    
    if save_slices:
        slices = np.zeros((len(cor_vals)+len(tilt_vals)+len(cor_vals), ig.shape[1], ig.shape[2]))
        
    # scan COR
    print(f"Fix tilt = {tilt_initial:.3f}")
    losses1, slices1, cor_min = linear_cor_scan(data, ag, tilt_initial, cor_vals, voxel_num_z, recon_method, recon_iters, loss_kind, save_slices)
    
    # scan tilt
    print(f"Fix COR = {cor_min:.3f}")
    losses2, slices2, tilt_min = linear_tilt_scan(data, ag, tilt_vals, cor_min/binning, voxel_num_z, recon_method, recon_iters, loss_kind, save_slices)

    # scan COR again
    print(f"Fix tilt = {tilt_min:.3f}")
    losses3, slices3, cor_min = linear_cor_scan(data, ag, tilt_min, cor_vals, voxel_num_z, recon_method, recon_iters, loss_kind, save_slices)

    if save_slices:
        return cor_min, tilt_min, losses1, losses2, losses3, slices1, slices2, slices3
    else:
        return cor_min, tilt_min, losses1, losses2, losses3
    
def joint_geometry_reconstruction(data, ag, p0=(30.0, 5.0), binning=1,
                                  voxel_num_z=256,
                                  recon_method ='FBP', recon_iters = None,
                                  loss_kind='L2', min_method='Powell', maxiter=30,
                                  bounds=[(0,100),(0,100)], xtol=(1e-2, 1e-2),
                                  save_slices=False):
    ig = ag.get_ImageGeometry()
    ig.voxel_num_z = voxel_num_z


    p0_binned = (p0[0], p0[1]/binning)
    bounds_binned = (bounds[0], (bounds[1][0]/binning, bounds[1][1]/binning))
    evaluations = []
    slices = []
    def geom_loss_wrapper(p):
        tilt, cor = p

        tilt = p[0] * xtol[0]
        cor  = p[1] * xtol[1]
        
        loss, x = geom_loss(data, ag, tilt, cor, recon_method, recon_iters, voxel_num_z, loss_kind)
        
        if save_slices:
            slices.append(x.as_array()[x.shape[0]//2])

        evaluations.append((tilt, cor * binning, loss))  
        print(f"tilt: {tilt:.6f}, cor: {cor*binning:.6f}, loss: {loss:.6e}")
        return loss

    p0_scaled = np.array([p0_binned[0] / xtol[0],
                          p0_binned[1] / xtol[1]], dtype=float)
    
    bounds_scaled = [(bounds_binned[0][0] / xtol[0], bounds_binned[0][1] / xtol[0]),
                     (bounds_binned[1][0] / xtol[1],  bounds_binned[1][1] / xtol[1])]
    
    res_scaled = minimize(geom_loss_wrapper, np.asarray(p0_scaled, float),
                   method=min_method,
                   bounds=bounds_scaled,
                   options={'maxiter': maxiter, 'disp': True, 'xtol': 1.0})
    
    res_real = res_scaled
    res_real.x = np.array([res_scaled.x[0] * xtol[0],
                           res_scaled.x[1] * xtol[1]])

    if save_slices:
        return res_real, evaluations, slices
    else:
        return res_real, evaluations
    
def linear_cor_scan(data, ag, tilt_fixed, cor_vals, voxel_num_z=256, recon_method='FBP', recon_iters=None, loss_kind='L2', save_slices=False):
    losses = np.zeros(len(cor_vals))
    binned_cor_vals = cor_vals/binning
    
    if save_slices:
        slices = np.zeros((len(binned_cor_vals), ig.shape[1], ig.shape[2]))
    
    for i, cor in enumerate(binned_cor_vals):

        loss, x = geom_loss(data, ag, tilt_fixed, cor, recon_method, recon_iters, voxel_num_z, loss_kind)
        losses[i] = loss
        
        if save_slices:
            slices[i, :, :]  = x.as_array()[x.shape[0]//2]
        else:
            slices = None
            
        print(f"\t[Scan cor] cor: {cor*binning:.3f}, loss: {loss:.6e}. {i+1}/{len(cor_vals)}")
    
    min_cor = get_min(cor_vals, losses, ind)

    return losses, slices, min_cor

    
def linear_tilt_scan(data, ag, tilt_vals, cor_fixed, voxel_num_z=256, recon_method='FBP', recon_iters=None, loss_kind='L2', save_slices=False):
    losses = np.zeros(len(tilt_vals))
    
    if save_slices:
        slices = np.zeros((len(tilt_vals), ig.shape[1], ig.shape[2]))
    
    for i, tilt in enumerate(tilt_vals):
        
        loss, x  = geom_loss(data, ag, tilt, cor_fixed, recon_method, recon_iters, voxel_num_z, loss_kind)
        losses[i] = loss
        
        if save_slices:
            slices[i, :, :]  = x.as_array()[x.shape[0]//2]
        else:
            slices = None
            
        print(f"\t[Scan tilt] tilt: {tilt:.3f}, loss: {loss:.6e}. {i+1}/{len(tilt_vals)}")

    min_tilt = get_min(tilt_vals, losses, ind)

    return losses, slices, min_tilt

    
def plot_losses2D(losses, tilt_vals, cor_vals):
    plt.figure(figsize=(6, 4))
    plt.imshow(losses, aspect='equal', origin='lower',
            extent=[cor_vals[0], cor_vals[-1], tilt_vals[0], tilt_vals[-1]])
    plt.xlabel('Centre of rotation (pixels)')
    plt.ylabel('Rotation axis tilt angle (degrees)')

    plt.colorbar(label='Filtered projection residual (sum of squares)')

def plot_losses_slice2D(losses, tilt_vals, cor_vals):

    i, j = np.unravel_index(np.argmin(losses), losses.shape)

    fig, axes = plt.subplots(1, 2, figsize=(8,4))
    ax = axes[0]
    ax.plot(cor_vals, losses[i,:] )
    
    cor_min = get_min(cor_vals, losses[i, :], j)
    ax.plot([cor_min, cor_min], ax.get_ylim()) 
    # xx, yy  = fit_cubic(cor_vals, losses[i,:])
    # ax.plot(xx, yy, '--r')
    # cor_min = xx[np.argmin(yy)]
    
    ax.set_xlabel('Centre of rotation offset (pixels)')
    ax.set_ylabel('Loss function value')
    ax.grid()

    ax = axes[1]
    ax.plot(tilt_vals, losses[:,j] )
    
    tilt_min = get_min(tilt_vals, losses[:,j], i)
    ax.plot([tilt_min, tilt_min], ax.get_ylim())   
    # xx, yy  = fit_cubic(tilt_vals, losses[:, j])
    # ax.plot(xx, yy, '--r')
    # tilt_min = xx[np.argmin(yy)]

    ax.set_xlabel('Rotation axis tilt (degrees)')
    ax.set_ylabel('Loss function value')
    ax.grid()

    plt.tight_layout()

    return cor_min, tilt_min

def plot_scipy_loss(evaluations):
    tilt_array, cor_array, loss_array = zip(*evaluations)
    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(tilt_array, cor_array, c=loss_array, cmap='viridis', s=100, edgecolors='k')
    plt.colorbar(scatter, label='Loss value')
    plt.xlabel('Tilt')
    plt.ylabel('Cor')
    plt.grid()
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
angle_binning = 1
ag = data.geometry
ig = ag.get_ImageGeometry()
data.reorder('astra')
fbp = FBP(ig, ag)
recon = fbp(data)

show2D(recon)

# %%
file_path = "data/k11-54286.nxs"
data, image_key, angles = load_data(file_path, verbose=False)
data, angles = preprocess_data(data, image_key, angles, verbose=False)
tilt = 35
detector_pixel_size=0.54
angle_binning = 4
geometry = create_geometry(
    angles, tilt,
    data.shape[2], data.shape[1],
    detector_pixel_size,
)

data = AcquisitionData(data, geometry=geometry)
data.reorder('astra')
# %%
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
        'angle': (None, None, angle_binning)
    }
data_binned = Binner(roi)(data)
ag_binned = data_binned.geometry
ig_binned = ag_binned.get_ImageGeometry()
recon = FBP(ig_binned, ag_binned)(data_binned)
show2D(recon)
# x_fixed.apply_circular_mask(0.9)

# %% coarse 2D search
tilt_vals = np.arange(33, 37.1, 0.5) # np.linspace(27, 35, 5)
cor_vals = np.arange(1, 4, 0.5) # np.linspace(2, 10, 5) # un-binned range
losses, slices = geometry_scan_2D(data_binned, ag_binned, tilt_vals, cor_vals, binning,
                         voxel_num_z=256,
                         recon_method='FBP', recon_iters=None, 
                         loss_kind='L2', 
                         save_slices=True)
# %%
plot_losses2D(losses, tilt_vals, cor_vals)
cor_centre, tilt_centre = plot_losses_slice2D(losses, tilt_vals, cor_vals)
# %%
# plot all the slices
plot_slices(slices, tilt_vals, cor_vals)
plot_slice(slices, tilt_vals, cor_vals, tilt_centre, cor_centre)

# %% Choose fine range

# perhaps we can do something fancy to choose the fine search range?
# grad_tilt = (losses[min(i+1, len(tilt_vals)-1), j] - losses[max(i-1, 0), j]) / (tilt_vals[min(i+1, len(tilt_vals)-1)] - tilt_vals[max(i-1, 0)])
# grad_cor  = (losses[i, min(j+1, len(cor_vals)-1)] - losses[i, max(j-1, 0)]) / (cor_vals[min(j+1, len(cor_vals)-1)] - cor_vals[max(j-1, 0)])

# for now just define a range and check it isn't too long
tilt_precision = 0.01
tilt_range = 0.05 # plus or minus from coarse centre
tilt_vals = np.arange(tilt_centre - tilt_range, tilt_centre + tilt_range, tilt_precision)
print(tilt_vals)

cor_precision = 0.005
cor_range = 0.025 # plus or minus from coarse centre
cor_vals = np.arange(cor_centre - cor_range, cor_centre + cor_range, cor_precision)
print(cor_vals)
print('2D scan length: ' + str(len(cor_vals)*len(tilt_vals)))
print('Linear scan length: ' + str(len(cor_vals)+len(tilt_vals)+len(cor_vals)))

# %% Fine grid search
binning = 1
roi = {
        'horizontal': (None, None, binning),
        'vertical': (None, None, binning),
        'angle': (None, None, angle_binning)
    }
data_binned = Binner(roi)(data)
ag_binned = data_binned.geometry
ig_binned = ag_binned.get_ImageGeometry()

losses, slices = geometry_scan_2D(data_binned, ag_binned, tilt_vals, cor_vals, binning,
                         voxel_num_z=256,
                         recon_method='FBP', recon_iters=None, 
                         loss_kind='L2', 
                         save_slices=True)
# %%
plot_losses2D(losses, tilt_vals, cor_vals)
cor_centre, tilt_centre = plot_losses_slice2D(losses, tilt_vals, cor_vals)
# %%
# plot all the slices
plot_slices(slices, tilt_vals, cor_vals)
plot_slice(slices, tilt_vals, cor_vals, tilt_centre, cor_centre)
# %% Fine linear search

binning = 1
roi = {
        'horizontal': (None, None, binning),
        'vertical': (None, None, binning),
        'angle': (None, None, angle_binning)
    }
data_binned = Binner(roi)(data)
ag_binned = data_binned.geometry
ig_binned = ag_binned.get_ImageGeometry()
# %%
losses, slices = geometry_scan_linear(data_binned, ag_binned, tilt_centre, tilt_vals, cor_vals, binning,
                         voxel_num_z=128,
                         recon_method='FBP', recon_iters=None, 
                         loss_kind='L2', 
                         save_slices=True)

# %%Run SciPy minimiser
binning = 4
roi = {
        'horizontal': (None, None, binning),
        'vertical': (None, None, binning),
        'angle': (None, None, angle_binning)
    }
data_binned = Binner(roi)(data)
ag_binned = data_binned.geometry
ig_binned = ag_binned.get_ImageGeometry()

# bounds = [(28.0, 32.0), (4.0, 6.0)]
bounds = [(33.0, 37.0), (1.0, 3.0)]
# p0 = (30.0, 5.0)
p0 = (35.0, 2.0)
res, evaluations = joint_geometry_reconstruction(data_binned, ag_binned, binning=binning,
                                    p0=p0,
                                    recon_iters=15,
                                    voxel_num_z=256,
                                    bounds=bounds,
                                    xtol = (0.5, 0.2))
tilt_min = res.x[0]
cor_min = res.x[1]*binning
print("Optimised tilt, CoR =", tilt_min, cor_min)

# %%
binning = 1
roi = {
        'horizontal': (None, None, binning),
        'vertical': (None, None, binning),
        'angle': (None, None, angle_binning)
    }
data_binned = Binner(roi)(data)
ag_binned = data_binned.geometry
ig_binned = ag_binned.get_ImageGeometry()

bounds = [(tilt_min-0.5, tilt_min+0.5), (cor_min-0.5, cor_min+0.5)]
res, evaluations = joint_geometry_reconstruction(data_binned, ag_binned, binning=binning,
                                    p0=(tilt_min, cor_min),
                                    recon_iters=15,
                                    voxel_num_z=256,
                                    bounds=bounds,
                                    xtol = (0.02,0.01))

tilt_min = res.x[0]
cor_min = res.x[1]*binning

print("Optimised tilt, CoR =", tilt_min, cor_min)


# %%
update_geometry(ag, tilt_min, cor_min)
recon = FBP(ig, ag)(data)
show2D(recon)

# %%


# %% Checking what the filters look like
binning = 1
roi = {
        'horizontal': (None, None, binning),
        'vertical': (None, None, binning),
        'angle': (None, None, angle_binning)
    }
data_binned = Binner(roi)(data)
ag_binned = data_binned.geometry
ig_binned = ag_binned.get_ImageGeometry()

tilt = 34.87
cor = 2.455
voxel_num_z = 256

kind = 'L2'

update_geometry(ag_binned, tilt, cor)
ig_binned = ag_binned.get_ImageGeometry()
ig_binned.voxel_num_z = voxel_num_z
x = reconstruct(data_binned, ig_binned, ag_binned, 'FBP', voxel_num_z, None)

A = ProjectionOperator(ig_binned, ag_binned)
yhat = A.direct(x)
r = yhat - data_binned

r = r.as_array()

h = r - gaussian_filter(r, sigma=3)
s = sobel_2d(h)

loss_r = float(np.sum(r**2))
loss_h = float(np.sum(h**2))
loss_s = float(np.sum(s**2))


show2D([r[:, 0, :], h[:, 0, :], s[:, 0, :]],
       ['Residual ' + str(loss_r), 'High pass filter ' + str(loss_h), 'Sobel filter ' + str(loss_s)],
       num_cols=3)


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


# %%
import cil
from cil.framework import DataContainer
def load_data(file_path, verbose=True):
    if verbose:
        print(f"Reading data from {file_path}")
    
    data = cil.io.utilities.HDF5_utilities.read(file_path, '/entry/imaging/data')
    image_key = cil.io.utilities.HDF5_utilities.read(file_path, '/entry/instrument/EtherCAT/image_key')
    angles = cil.io.utilities.HDF5_utilities.read(file_path, '/entry/imaging_sum/smaract_zrot')
    
    if verbose:
        print(f"Data loaded: {data.shape}, Image key: {image_key.shape}, Angles: {angles.shape}")
        unique_keys, counts = np.unique(image_key, return_counts=True)
        for key, count in zip(unique_keys, counts):
            key_type = {0: "Tomography", 1: "Flat field", 2: "Dark field"}.get(key, f"Unknown ({key})")
            print(f"  {key_type} images: {count}")
    
    return data, image_key, angles


def preprocess_data(data, image_key, angles, verbose=True):
    flat_indices = np.where(image_key == 1)[0]
    dark_indices = np.where(image_key == 2)[0]
    proj_indices = np.where(image_key == 0)[0]
    
    if len(flat_indices) == 0:
        raise ValueError("No flat field images found!")
    if len(dark_indices) == 0:
        raise ValueError("No dark field images found!")
    
    flat_fields = data[flat_indices]
    dark_fields = data[dark_indices]
    projections = data[proj_indices]
    projection_angles = angles[proj_indices]
    
    flat_mean = np.mean(flat_fields, axis=0)
    dark_mean = np.mean(dark_fields, axis=0)
    
    normalized_data = np.zeros_like(projections, dtype=np.float32)
    for i in range(projections.shape[0]):
        a = (projections[i] - dark_mean)
        b = (flat_mean - dark_mean)
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(a, b)
            c[~np.isfinite(c)] = 1e-5
        normalized_data[i] = c.astype(np.float32)
    
    clipped = np.clip(normalized_data, 1e-3, None)
    absorption_data = -np.log(clipped)
    
    if verbose:
        print(f"Preprocessed: {absorption_data.shape} projections")
    
    data_container = DataContainer(absorption_data, dtype=np.float32, 
                                  dimension_labels=['angle', 'vertical', 'horizontal'])
    
    return data_container, projection_angles

def create_geometry(projection_angles, tilt_angle, num_pixels_x, num_pixels_y,
                   detector_pixel_size):
    tilt = np.radians(tilt_angle)
    
    ag = AcquisitionGeometry.create_Parallel3D(
        rotation_axis_direction=[0, -np.sin(tilt), np.cos(tilt)],
        units="microns"
    )
    
    ag.set_panel(
        num_pixels=[num_pixels_x, num_pixels_y],
        origin='top-left',
        pixel_size=detector_pixel_size
    )
    
    ag.set_angles(projection_angles)
    return ag