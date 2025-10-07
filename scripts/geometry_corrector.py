# %%
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter, sobel
import matplotlib.pyplot as plt
import numpy as np

from cil.framework import Processor
from cil.plugins.astra.operators import ProjectionOperator
from cil.plugins.astra.processors import FBP
from cil.processors import Binner, Slicer
from cil.framework import AcquisitionData
from cil.framework.labels import AcquisitionType, AcquisitionDimension

from shrink_volume import VolumeShrinker
import logging
log = logging.getLogger(__name__)



class GeometryCorrector(Processor):

    def __init__(self, initial_parameters=(30.0, 0.0), parameter_bounds=[(25, 45),(-5, 5)], parameter_tolerance=1e-6, 
                 initial_binning=None, angle_binning = None, reduced_volume = None):
        """
        
        """
        kwargs = {
                    'initial_parameters'  : initial_parameters,
                    'parameter_bounds' : parameter_bounds,
                    'parameter_tolerance' : parameter_tolerance,
                    'reduced_volume' : reduced_volume,
                    'initial_binning' : initial_binning,
                    'angle_binning' : angle_binning,
                    'evaluations' : []
                    }
        super(GeometryCorrector, self).__init__(**kwargs)

    def check_input(self, dataset):
        if dataset.geometry.geom_type & AcquisitionType.CONE:
            raise NotImplementedError("GeometryCorrector does not yet support CONE data")
        
        if not AcquisitionDimension.check_order_for_engine('astra', dataset.geometry):
            raise ValueError("GeometryCorrector must be used with astra data order, try `data.reorder('astra')`")

        return True

    def update_geometry(self, ag, tilt_deg, cor_pix, 
                        tilt_direction_vector = np.array([1, 0, 0]), 
                        original_rotation_axis=np.array([0, 0, 1])):

        tilt_rad = np.deg2rad(tilt_deg)
        rotation_matrix = R.from_rotvec(tilt_rad * tilt_direction_vector)
        tilted_rotation_axis = rotation_matrix.apply(original_rotation_axis)

        ag.set_centre_of_rotation(offset=cor_pix, distance_units='pixels')
        ag.config.system.rotation_axis.direction = tilted_rotation_axis

        return ag
   
    def sobel_2d(self, arr):
        gx = sobel(arr, axis=0)
        gy = sobel(arr, axis=2)
        return np.sqrt(gx**2 + gy**2)

    def highpass_2d(self, arr, sigma=3.0):
        return arr - gaussian_filter(arr, sigma=sigma)
    
    def get_min(self, offsets, values, ind):
        #calculate quadratic from 3 points around ind  (-1,0,1)
        a = (values[ind+1] + values[ind-1] - 2*values[ind]) * 0.5
        b = a + values[ind] - values[ind-1]
        ind_centre = -b / (2*a)+ind

        ind0 = int(ind_centre)
        w1 = ind_centre - ind0
        return (1.0 - w1) * offsets[ind0] + w1 * offsets[ind0+1]
    
    def loss_from_residual(self, residual,
                       hp_sigma=3.0,
                       use_highpass=True,
                       use_sobel=True,
                       normalize_per_angle=False):
        
        r = residual.as_array()

        if use_highpass:
            r = self.highpass_2d(r, sigma=hp_sigma)
        if use_sobel:
            r = self.sobel_2d(r)

        return float(np.sum(r**2))
    
    def projection_reprojection(self, data, ig, ag, ag_ref, y_ref, tilt_deg, cor_pix):
        
        ag = self.update_geometry(ag, tilt_deg, cor_pix)
        recon = FBP(ig, ag)(data)
        recon.apply_circular_mask(0.9)

        # ag = Slicer(roi={'angle':(None, None, divider)})(ag)
        ag_ref = self.update_geometry(ag_ref, tilt_deg, cor_pix)
        A = ProjectionOperator(ig, ag_ref)
        
        yhat = A.direct(recon)
        r = yhat - y_ref
        
        loss = self.loss_from_residual(r)
        
        return loss, recon
    
   
    def minimise_geometry(self, data, binning, p0, bounds):
        

        current_run_evaluations = []
        xtol = self.parameter_tolerance
        xtol_binned = (xtol[0], binning*xtol[1])
        p0_binned = (p0[0], p0[1]/binning)
        bounds_binned = (bounds[0], (bounds[1][0]/binning, bounds[1][1]/binning))

        p0_scaled = np.array([p0_binned[0] / xtol[0],
                              p0_binned[1] / xtol[1]], dtype=float)
        
        bounds_scaled = [(bounds_binned[0][0] / xtol[0], bounds_binned[0][1] / xtol[0]),
                         (bounds_binned[1][0] / xtol[1],  bounds_binned[1][1] / xtol[1])]
        
        print(f"Tilt bounds : ({bounds[0][0]:.3f}:{bounds[0][1]:.3f}), CoR bounds : ({bounds[1][0]:.3f}:{bounds[1][1]:.3f})")

        target = max(np.ceil(data.get_dimension_size('angle') / 10), 36)
        divider = np.floor(data.get_dimension_size('angle') / target)
        y_ref = Slicer(roi={'angle':(None, None, divider)})(data)

        ag = data.geometry.copy()
        ag_ref = Slicer(roi={'angle':(None, None, divider)})(ag)

        ig = ag.get_ImageGeometry()
        if self.reduced_volume is not None:
            ig.voxel_num_z = self.reduced_volume.voxel_num_z//binning
            ig.voxel_num_x = self.reduced_volume.voxel_num_x//binning
            ig.voxel_num_y = self.reduced_volume.voxel_num_y//binning
            ig.center_x = self.reduced_volume.center_x//binning
            ig.center_y = self.reduced_volume.center_y//binning
            ig.center_z = self.reduced_volume.center_z//binning

        loss_at_p0, _ = self.projection_reprojection(data, ig, ag, ag_ref, y_ref, p0_binned[0], p0_binned[1])
        ftol = self.ftol_from_bounds_and_xtol(loss_at_p0, 1, bounds_scaled)
        
        def loss_function_wrapper(p):
            tilt = p[0] * xtol[0]
            cor  = p[1] * xtol[1]

            loss, recon = self.projection_reprojection(data, ig, ag, ag_ref, y_ref, tilt, cor)
            
            current_run_evaluations.append((tilt, cor * binning, loss))

            print(f"tilt: {tilt:.3f}, CoR: {cor*binning:.3f}, loss: {loss:.3e}")

            return loss

        res_scaled = minimize(loss_function_wrapper, p0_scaled,
                    method='Powell',
                    bounds=bounds_scaled,
                    options={'maxiter': 5, 'disp': True, 'xtol': 1.0, 'ftol':ftol})
        
        res_real = res_scaled
        res_real.x = np.array([res_scaled.x[0] * xtol[0],
                           res_scaled.x[1] * xtol[1] * binning])
        
        self.evaluations.append({
            "p0": p0,
            "bounds": bounds,
            "binning": binning,
            "xtol": xtol,
            "result": res_real,
            "evaluations": current_run_evaluations
        })

        return res_real
    
    def ftol_from_bounds_and_xtol(self, loss_at_p0, xtol, bounds, min_abs_fatol=1e-6):

        xtol = np.asarray(xtol, dtype=float)
        ranges = np.array([b[1] - b[0] for b in bounds], dtype=float)
        tau = np.min(xtol / ranges)
        # rel_fatol = base_rel*tau
        ftol = max(min_abs_fatol, tau * abs(loss_at_p0))
        return ftol
    
    def process(self, out=None):
        data = self.get_input()

        if self.initial_binning is None:
            self.initial_binning = min(int(np.ceil(data.geometry.config.panel.num_pixels[0] / 128)),16)
        binning = self.initial_binning
        if self.angle_binning is None:
            self.angle_binning = np.ceil(data.get_dimension_size('angle')/(data.get_dimension_size('horizontal')*(np.pi/2)))
        roi = {
                'horizontal': (None, None, binning),
                'vertical': (None, None, binning),
                'angle': (None, None, self.angle_binning*binning)
            }
        #gaussian filter data
        # data_binned = data.copy()
        # data_binned.fill(gaussian_filter(data.as_array(), [binning//2, 0, binning//2]))
        data_binned = Binner(roi)(data)
        
        coarse_tolerance = (self.parameter_tolerance[0], self.parameter_tolerance[1])
        res = self.minimise_geometry(data_binned, binning=binning, 
                                                  p0=self.initial_parameters, 
                                                  bounds=self.parameter_bounds)
        
        tilt_min = res.x[0]
        cor_min = res.x[1]
        print(f"Coarse scan optimised tilt = {tilt_min:.3f}, CoR = {cor_min:.3f}")

        binning = 1
        roi = {
                'horizontal': (None, None, binning),
                'vertical': (None, None, binning),
                'angle': (None, None, self.angle_binning)
            }
        #gaussian filter data
        # data_binned = data.copy()
        # data_binned.fill(gaussian_filter(data.as_array(), [binning//2, 0, binning//2]))
        data_binned = Binner(roi)(data)

        m = 3 # how many coarse-tolerance units to extend on either side of the coarse optimum to set the fine search bounds
        
        fine_bounds_tilt = (tilt_min - m * coarse_tolerance[0], tilt_min + m * coarse_tolerance[0])
        fine_bounds_cor = (cor_min - m * coarse_tolerance[1], cor_min + m * coarse_tolerance[1])

        res = self.minimise_geometry(data_binned, binning=binning,
                                            p0=(tilt_min, cor_min), bounds=[fine_bounds_tilt, fine_bounds_cor])
        tilt_min = res.x[0]
        cor_min = res.x[1]
        print(f"Fine scan optimised tilt = {tilt_min:.3f}, CoR ={cor_min:.3f}")
        
        if log.isEnabledFor(logging.DEBUG):
            self.plot_evaluations()

        new_geometry = data.geometry.copy()
        self.update_geometry(new_geometry, tilt_min, cor_min)

        if out is None:
            return AcquisitionData(array=data.as_array(), deep_copy=True, geometry=new_geometry)
        else:
            out.geometry = new_geometry
            return out
        
    def plot_evaluations(self):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
        for i in [0, 1]:
            eval = self.evaluations[i]
            tilts = [t[0] for t in eval['evaluations']]
            cors = [t[1] for t in eval['evaluations']]
            losses = [t[2] for t in eval['evaluations']]
            
            ax = axs[i]
            scatter = ax.scatter(tilts, cors, c=losses, cmap='viridis', s=100, edgecolors='k')
            fig.colorbar(scatter, label='Loss value', ax=ax)
            ax.set_xlabel('Tilt')
            ax.set_ylabel('Cor')
            ax.set_title('bounds = ({:.2f}:{:.2f}), ({:.2f}:{:.2f}), binning = {}, xtol = ({}, {}) \n result = ({:.3f}, {:.3f})'
                        .format(*eval['bounds'][0], *eval['bounds'][1], eval['binning'], *eval['xtol'], eval['result'].x[0], eval['result'].x[1]))
            ax.grid()
        plt.tight_layout()


# %%
from cil.io.utilities import HDF5_utilities
from cil.framework import AcquisitionData, AcquisitionGeometry
from cil.processors import TransmissionAbsorptionConverter, Normaliser
from cil.utilities.display import show2D, show_geometry
from cil.utilities.jupyter import islicer
from cil.plugins.astra.processors import FBP

import numpy as np
from scipy.spatial.transform import Rotation as R
import time

file_path = "../alignment_methods/data/k11-54286.nxs"
detector_pixel_size=0.54 # um - get this from the file in future

data = HDF5_utilities.read(file_path, '/entry/imaging/data')
image_key = HDF5_utilities.read(file_path, '/entry/instrument/EtherCAT/image_key')
angles = HDF5_utilities.read(file_path, '/entry/imaging_sum/smaract_zrot')

unique_keys, counts = np.unique(image_key, return_counts=True)
for key, count in zip(unique_keys, counts):
    key_type = {0: "Tomography", 1: "Flat field", 2: "Dark field"}.get(key, f"Unknown ({key})")
    print(f"  {key_type} images: {count}")

flat_fields = data[np.where(image_key == 1)[0]]
dark_fields = data[np.where(image_key == 2)[0]]
projections = data[np.where(image_key == 0)[0]]
projection_angles = angles[np.where(image_key == 0)[0]]

cor = 2 # pix
tilt = 31 # deg
ag = AcquisitionGeometry.create_Parallel3D(units="microns")
ag.set_panel(num_pixels=[projections.shape[2], projections.shape[1]],
        origin='top-left',
        pixel_size=detector_pixel_size)
ag.set_angles(projection_angles)

ag.set_centre_of_rotation(offset=cor, distance_units='pixels')

tilt_direction_vector=np.array([1, 0, 0])
original_rotation_axis=np.array([0, 0, 1])

rotation_matrix = R.from_rotvec(np.deg2rad(tilt) * tilt_direction_vector)
tilted_rotation_axis = rotation_matrix.apply(original_rotation_axis)

ag.config.system.rotation_axis.direction = tilted_rotation_axis

acq_data = AcquisitionData(projections, deep_copy=False, geometry=ag)
acq_data = Normaliser(np.mean(flat_fields, axis=0), np.mean(dark_fields, axis=0))(acq_data)
acq_data = TransmissionAbsorptionConverter(min_intensity=1e-6)(acq_data)
acq_data.reorder('astra')
data = acq_data
# %%
# ag = acq_data.geometry
# ig = ag.get_ImageGeometry()
# fbp = FBP(ig, ag)
# recon = fbp(acq_data)
# show2D(recon)
# %%

binning = min(int(np.ceil(data.geometry.config.panel.num_pixels[0] / 128)),16)
binning = 2
angle_binning = np.ceil(data.get_dimension_size('angle')/(data.get_dimension_size('horizontal')*(np.pi/2)))
roi = {
        'horizontal': (None, None, binning),
        'vertical': (None, None, binning),
        'angle': (None, None, angle_binning*binning)
    }
data_binned = Binner(roi)(data)

# %%
vs = VolumeShrinker()
ig_reduced = vs.run(data_binned)

# %%
cor = 2 # pix
tilt = 34.5 # deg

optimise_geometry = True
cor_bounds = (-10, 10) # pixels
tilt_bounds = (30, 40) # deg
tilt_tol = 0.01 # deg
cor_tol = 0.01 # pixels

if optimise_geometry:
    
    t0 = time.time()

    processor = GeometryCorrector(initial_parameters=(tilt, cor), parameter_bounds=(tilt_bounds, cor_bounds), parameter_tolerance=(tilt_tol, cor_tol),
                                  reduced_volume=ig_reduced)

    processor.set_input(data_binned)
    data_corrected = processor.get_output()
    print((time.time()-t0)/60)
else:
    data_corrected = data

processor.plot_evaluations()
# %%
ag = data_corrected.geometry
ig = ag.get_ImageGeometry()
fbp = FBP(ig, ag)
recon = fbp(data_corrected)
show2D(recon)

# %%