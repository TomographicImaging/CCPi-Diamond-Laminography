# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import gc
import time
from datetime import datetime
from functools import partial

from cil.framework import AcquisitionData, AcquisitionGeometry, DataContainer
from cil.recon import FBP
from cil.processors import Binner
from cil.plugins.astra import ProjectionOperator
import cil.io.utilities
from skimage.filters import sobel
from scipy.ndimage import laplace
from scipy.stats import kurtosis



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


def apply_binning(data, binning_factor):
    if binning_factor <= 1:
        return data
    
    arr = data.as_array()
    
    binned_arr = arr[:, ::binning_factor, ::binning_factor]
    
    return DataContainer(binned_arr, dtype=np.float32, 
                        dimension_labels=['angle', 'vertical', 'horizontal'])


def create_geometry(projection_angles, tilt_angle, num_pixels_x, num_pixels_y,
                   detector_pixel_size, detector_binning=1):
    tilt = np.radians(tilt_angle)
    effective_pixel_size = detector_pixel_size * detector_binning
    
    ag = AcquisitionGeometry.create_Parallel3D(
        rotation_axis_direction=[0, -np.sin(tilt), np.cos(tilt)],
        units="microns"
    )
    
    ag.set_panel(
        num_pixels=[num_pixels_x, num_pixels_y],
        origin='top-left',
        pixel_size=effective_pixel_size
    )
    
    ag.set_angles(projection_angles)
    return ag


def reconstruct(acquisition_data, cor_offset=0):
    if cor_offset != 0:
        acquisition_data.geometry.set_centre_of_rotation(cor_offset, distance_units='pixels')
    
    acquisition_data.reorder('astra')
    reconstructor = FBP(acquisition_data, backend='astra')
    return reconstructor.run()


def validate_parameters(tilt_range, cor_range, metrics_list):
    valid_metrics = ['gradient', 'projection_consistency', 'contrast', 'focus', 'total_variation', 'kurtosis']
    
    # Handle 'all' or None to enable all metrics
    if metrics_list is None or metrics_list == 'all':
        metrics_list = valid_metrics.copy()
    
    for metric in metrics_list:
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric '{metric}'. Valid metrics: {valid_metrics}")
    
    if isinstance(tilt_range, tuple) and len(tilt_range) == 3:
        if tilt_range[2] <= 0:
            raise ValueError("Tilt step size must be positive")
    elif not isinstance(tilt_range, (int, float, tuple)):
        raise ValueError("tilt_range must be a number or (start, stop, step) tuple")
        
    if isinstance(cor_range, tuple) and len(cor_range) == 3:
        if cor_range[2] <= 0:
            raise ValueError("COR step size must be positive")
    elif not isinstance(cor_range, (int, float, tuple)):
        raise ValueError("cor_range must be a number or (start, stop, step) tuple")
    
    return metrics_list


def calculate_metrics(volume_3d, metrics_list, slice_sampling=10, reconstruction=None, acquisition_data=None):
    nz, ny, nx = volume_3d.shape
    
    # Sample slices, avoiding edges and including middle
    margin = max(int(nz * 0.05), 3)
    slice_indices = list(range(margin, nz - margin, slice_sampling))
    if nz // 2 not in slice_indices:
        slice_indices.append(nz // 2)
    slice_indices.sort()
    
    metrics = {}
    metric_sums = {metric: 0.0 for metric in metrics_list if metric != 'projection_consistency'}
    
    # Calculate slice-based metrics
    for idx in slice_indices:
        slice_2d = volume_3d[idx]
        
        for metric in metrics_list:
            if metric == 'projection_consistency':
                continue  # Handle separately below
            elif metric == 'gradient':
                value = np.mean(sobel(slice_2d))
            elif metric == 'contrast':
                p2, p98 = np.percentile(slice_2d, (2, 98))
                value = p98 - p2
            elif metric == 'focus':
                value = np.var(laplace(slice_2d))
            elif metric == 'total_variation':
                dx = np.diff(slice_2d, axis=0)
                dy = np.diff(slice_2d, axis=1)
                value = -(np.sum(np.abs(dx)) + np.sum(np.abs(dy)))
            elif metric == 'kurtosis':
                value = kurtosis(slice_2d.flatten())
            else:
                continue
                
            metric_sums[metric] += value
    
    # Average slice-based metrics
    num_slices = len(slice_indices)
    for metric in metric_sums:
        metrics[metric] = metric_sums[metric] / num_slices
    
    if 'projection_consistency' in metrics_list:
        if reconstruction is not None and acquisition_data is not None:
            try:
                projector = ProjectionOperator(reconstruction.geometry, acquisition_data.geometry)
                forward_proj = projector.direct(reconstruction)
                orig_arr = acquisition_data.as_array()
                fproj_arr = forward_proj.as_array()
                
                # Sample every 10th projection
                indices = np.arange(0, min(orig_arr.shape[0], fproj_arr.shape[0]), 10)
                
                # nrmalised cross-correlation
                sampled_orig = orig_arr[indices]
                sampled_fproj = fproj_arr[indices]
                orig_norm = (sampled_orig - np.mean(sampled_orig)) / (np.std(sampled_orig) + 1e-10)
                fproj_norm = (sampled_fproj - np.mean(sampled_fproj)) / (np.std(sampled_fproj) + 1e-10)
                correlation = np.mean(orig_norm * fproj_norm)
                metrics['projection_consistency'] = correlation
                
                del forward_proj
                gc.collect()
                
            except Exception as e:
                print(f"  Warning: Projection consistency failed: {e}")
                metrics['projection_consistency'] = np.nan
        else:
            metrics['projection_consistency'] = np.nan
            
    return metrics


def reconstruct_and_evaluate(params):
    tilt = params['tilt']
    cor = params['cor']
    processed_data = params['processed_data']
    projection_angles = params['projection_angles']
    detector_pixel_size = params['detector_pixel_size']
    detector_binning = params['detector_binning']
    metrics_list = params['metrics_list']
    slice_sampling = params['slice_sampling']
    job_id = params['job_id']
    total_jobs = params['total_jobs']
    
    print(f"Job {job_id}/{total_jobs}: Reconstructing tilt={tilt:.2f}°, COR={cor:.1f}")
    
    try:
        geometry = create_geometry(
            projection_angles, tilt,
            processed_data.shape[2], processed_data.shape[1],
            detector_pixel_size, detector_binning
        )
        
        if cor != 0:
            geometry.set_centre_of_rotation(cor, distance_units='pixels')
        
        acquisition_data = AcquisitionData(processed_data, geometry=geometry)
        acquisition_data.reorder('astra')
        
        reconstruction = reconstruct(acquisition_data)
        volume_array = reconstruction.as_array()
        
        metrics = calculate_metrics(volume_array, metrics_list, slice_sampling, 
                                   reconstruction=reconstruction, acquisition_data=acquisition_data)
        
        result = {'tilt': tilt, 'cor': cor}
        result.update(metrics)
        
        print(f"Job {job_id}/{total_jobs}: Complete. Metrics: {list(metrics.keys())}")
        
        del reconstruction, acquisition_data, volume_array
        gc.collect()
        
        return result
        
    except Exception as e:
        print(f"Job {job_id}/{total_jobs}: Failed with error: {e}")
        result = {'tilt': tilt, 'cor': cor}
        for metric in metrics_list:
            result[metric] = np.nan
        return result


def grid_search(file_path, tilt_range, cor_range, metrics_list,
                slice_sampling=10, max_workers=4, output_dir=None,
                detector_pixel_size=0.54, detector_binning=4, binning=1,
                verbose=True):
    
    start_time = time.time()
    
    metrics_list = validate_parameters(tilt_range, cor_range, metrics_list)
    
    if isinstance(tilt_range, (int, float)):
        tilt_values = np.array([float(tilt_range)])
    else:
        tilt_values = np.arange(*tilt_range)
        
    if isinstance(cor_range, (int, float)):
        cor_values = np.array([float(cor_range)])
    else:
        cor_values = np.arange(*cor_range)
    
    total_jobs = len(tilt_values) * len(cor_values)
    
    if verbose:
        print("=" * 60)
        print("Grid Search Configuration:")
        print(f"- Tilt values: {len(tilt_values)} ({tilt_values[0]:.2f} to {tilt_values[-1]:.2f})")
        print(f"- COR values: {len(cor_values)} ({cor_values[0]:.2f} to {cor_values[-1]:.2f})")
        print(f"- Total reconstructions: {total_jobs}")
        print(f"- Metrics: {metrics_list}")
        print(f"- Slice sampling: every {slice_sampling} slices")
        print(f"- Parallel workers: {max_workers}")
        print("=" * 60)
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tilt_str = f"{tilt_values[0]:.1f}" if len(tilt_values) == 1 else f"{tilt_values[0]:.1f}-{tilt_values[-1]:.1f}"
        cor_str = f"{cor_values[0]:.1f}" if len(cor_values) == 1 else f"{cor_values[0]:.1f}-{cor_values[-1]:.1f}"
        output_dir = f"simple_grid_search_tilt{tilt_str}_cor{cor_str}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    if verbose:
        print("Loading and preprocessing data...")
    data, image_key, angles = load_data(file_path, verbose=False)
    processed_data, projection_angles = preprocess_data(data, image_key, angles, verbose=False)
    
    if binning > 1:
        processed_data = apply_binning(processed_data, binning)
    
    jobs = []
    job_id = 1
    for tilt in tilt_values:
        for cor in cor_values:
            job = {
                'tilt': tilt,
                'cor': cor,
                'processed_data': processed_data,
                'projection_angles': projection_angles,
                'detector_pixel_size': detector_pixel_size,
                'detector_binning': detector_binning,
                'metrics_list': metrics_list,
                'slice_sampling': slice_sampling,
                'job_id': job_id,
                'total_jobs': total_jobs
            }
            jobs.append(job)
            job_id += 1
    
    if verbose:
        print(f"Starting {total_jobs} reconstructions with {max_workers} workers...")
    
    results = []
    completed = 0
    
    def update_progress(result):
        nonlocal completed
        completed += 1
        results.append(result)
        if verbose:
            print(f"Progress: {completed}/{total_jobs} reconstructions completed ({100*completed/total_jobs:.1f}%)")
    
    with Pool(max_workers) as pool:
        async_results = []
        for job in jobs:
            async_result = pool.apply_async(reconstruct_and_evaluate, (job,), callback=update_progress)
            async_results.append(async_result)
        
        for async_result in async_results:
            async_result.wait()
    
    df = pd.DataFrame(results)
    
    csv_path = os.path.join(output_dir, 'grid_search_results.csv')
    df.to_csv(csv_path, index=False)
    
    plot_results(df, output_dir, tilt_values, cor_values, metrics_list)
    
    elapsed = time.time() - start_time
    if verbose:
        print("=" * 60)
        print(f"Grid search completed in {elapsed/60:.1f} minutes")
        print(f"Results saved to: {output_dir}")
        print(f"CSV file: {csv_path}")
        
        print("\nBest parameters for each metric:")
        for metric in metrics_list:
            if metric in df.columns:
                best_idx = df[metric].idxmax()
                if not pd.isna(df.loc[best_idx, metric]):
                    best_tilt = df.loc[best_idx, 'tilt']
                    best_cor = df.loc[best_idx, 'cor']
                    best_value = df.loc[best_idx, metric]
                    print(f"  {metric}: tilt={best_tilt:.2f}°, cor={best_cor:.2f}, value={best_value:.4f}")
        print("=" * 60)
    
    return df


def plot_results(df, output_dir, tilt_values, cor_values, metrics_list):
    if len(tilt_values) == 1 and len(cor_values) == 1:
        # Single point - bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        metric_vals = [df[metric].iloc[0] for metric in metrics_list if not pd.isna(df[metric].iloc[0])]
        metric_names = [m for m in metrics_list if not pd.isna(df[m].iloc[0])]
        ax.bar(metric_names, metric_vals)
        ax.set_title(f'Metrics for Tilt={tilt_values[0]:.2f}°, COR={cor_values[0]:.2f}')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_bar.png'), dpi=300)
        plt.close()
        
    elif len(tilt_values) == 1:
        # Fixed tilt, varying COR - line plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_list[:6]):
            ax = axes[i]
            ax.plot(cor_values, df[metric], 'o-')
            ax.set_xlabel('Center of Rotation (pixels)')
            ax.set_ylabel(metric.title())
            ax.set_title(f'{metric.title()} vs COR')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_vs_cor.png'), dpi=300)
        plt.close()
        
    elif len(cor_values) == 1:
        # Fixed COR, varying tilt - line plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_list[:6]):
            ax = axes[i]
            ax.plot(tilt_values, df[metric], 'o-')
            ax.set_xlabel('Tilt Angle (degrees)')
            ax.set_ylabel(metric.title())
            ax.set_title(f'{metric.title()} vs Tilt')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_vs_tilt.png'), dpi=300)
        plt.close()
        
    else:
        # 2D grid - heatmaps
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_list[:6]):
            ax = axes[i]
            
            grid = np.zeros((len(tilt_values), len(cor_values)))
            for j, tilt in enumerate(tilt_values):
                for k, cor in enumerate(cor_values):
                    mask = (df['tilt'] == tilt) & (df['cor'] == cor)
                    if mask.any():
                        grid[j, k] = df.loc[mask, metric].iloc[0]
                    else:
                        grid[j, k] = np.nan
            
            im = ax.imshow(grid, aspect='auto', cmap='viridis',
                          extent=[cor_values[0], cor_values[-1], tilt_values[-1], tilt_values[0]])
            ax.set_xlabel('COR (pixels)')
            ax.set_ylabel('Tilt (degrees)')
            ax.set_title(metric.title())
            plt.colorbar(im, ax=ax)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_heatmaps.png'), dpi=300)
        plt.close()

# %%
if __name__ == "__main__":
    # File to reconstruct:
    file_path = "data/k11-54286.nxs"

    tilt_range = (34, 37.5, 0.5)  # (start, stop, step) in degrees
    cor_range = (-2, 2.5, 0.5)     # (start, stop, step) in pixels

    # metrics_list = ['gradient', 'entropy', 'contrast', 'focus', 'total_variation']
    metrics_list = None  # or None to calculates all metrics

    # Run search
    results = grid_search(
        file_path=file_path,
        tilt_range=tilt_range,
        cor_range=cor_range,
        metrics_list=metrics_list,
        slice_sampling=10,          # Sample every nth slice
        max_workers=8,              # Parallel processes
        detector_pixel_size=0.54,   # microns
        detector_binning=4,         # hardware binning set for scan
        binning=4,                  # additional software binning
        verbose=True                # useful(?) print statements 
    )
        
    print("Grid search completed")

# %%
file_path = "data/k11-54286.nxs"

tilt_range = (34, 37.5, 0.5)  # (start, stop, step) in degrees
cor_range = (-2, 2.5, 0.5)     # (start, stop, step) in pixels

# metrics_list = ['gradient', 'entropy', 'contrast', 'focus', 'total_variation']
metrics_list = None  # or None to calculates all metrics
metrics_list = validate_parameters(tilt_range, cor_range, metrics_list)
    
if isinstance(tilt_range, (int, float)):
    tilt_values = np.array([float(tilt_range)])
else:
    tilt_values = np.arange(*tilt_range)
    
if isinstance(cor_range, (int, float)):
    cor_values = np.array([float(cor_range)])
else:
    cor_values = np.arange(*cor_range)

total_jobs = len(tilt_values) * len(cor_values)
# %%
slice_sampling=10 
max_workers=4
output_dir=None
detector_pixel_size=0.54
detector_binning=4
binning=1

if output_dir is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tilt_str = f"{tilt_values[0]:.1f}" if len(tilt_values) == 1 else f"{tilt_values[0]:.1f}-{tilt_values[-1]:.1f}"
    cor_str = f"{cor_values[0]:.1f}" if len(cor_values) == 1 else f"{cor_values[0]:.1f}-{cor_values[-1]:.1f}"
    output_dir = f"simple_grid_search_tilt{tilt_str}_cor{cor_str}_{timestamp}"

os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Input file not found: {file_path}")

data, image_key, angles = load_data(file_path, verbose=False)
processed_data, projection_angles = preprocess_data(data, image_key, angles, verbose=False)

if binning > 1:
    processed_data = apply_binning(processed_data, binning)
# %%
from cil.utilities.display import show2D
show2D([data, processed_data])
# %%
tilt = 35
cor = 0
job_id = 0
params = {      'tilt': tilt,
                'cor': cor,
                'processed_data': processed_data,
                'projection_angles': projection_angles,
                'detector_pixel_size': detector_pixel_size,
                'detector_binning': detector_binning,
                'metrics_list': metrics_list,
                'slice_sampling': slice_sampling,
                'job_id': job_id,
                'total_jobs': total_jobs
            }
tilt = params['tilt']
cor = params['cor']
processed_data = params['processed_data']
projection_angles = params['projection_angles']
detector_pixel_size = params['detector_pixel_size']
detector_binning = params['detector_binning']
metrics_list = params['metrics_list']
slice_sampling = params['slice_sampling']
job_id = params['job_id']
total_jobs = params['total_jobs']

print(f"Job {job_id}/{total_jobs}: Reconstructing tilt={tilt:.2f}°, COR={cor:.1f}")


geometry = create_geometry(
    projection_angles, tilt,
    processed_data.shape[2], processed_data.shape[1],
    detector_pixel_size, detector_binning
)

if cor != 0:
    geometry.set_centre_of_rotation(cor, distance_units='pixels')

acquisition_data = AcquisitionData(processed_data, geometry=geometry)
acquisition_data.reorder('astra')

reconstruction = reconstruct(acquisition_data)

show2D(reconstruction)
# %%
volume_array = reconstruction.as_array()

metrics = calculate_metrics(volume_array, metrics_list, slice_sampling, 
                            reconstruction=reconstruction, acquisition_data=acquisition_data)

result = {'tilt': tilt, 'cor': cor}
result.update(metrics)

print(f"Job {job_id}/{total_jobs}: Complete. Metrics: {list(metrics.items()) }")
# %%
del reconstruction, acquisition_data, volume_array
        

        

# %%
show2D(recon)
# %%