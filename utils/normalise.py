# %%
import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from tifffile import imwrite

from cil.processors import TransmissionAbsorptionConverter
from cil.utilities.display import show_geometry, show2D
from cil.utilities.jupyter import islicer
from cil.processors import FluxNormaliser, Normaliser
from cil.framework import AcquisitionData, AcquisitionGeometry
from cil.io import TIFFWriter

# %%
output_path = 'output_data'
json_fname = os.path.join(output_path, 'cylinder_simulation_360.json')
with open(json_fname, 'r') as file:
    params = json.load(file)

ag = AcquisitionGeometry.create_Parallel3D(ray_direction = params['Detector']['BeamDirection'],
                                      detector_position = params['Detector']['Position'],
                                      detector_direction_x = params['Detector']['RightVector'],
                                      detector_direction_y = params['Detector']['UpVector'],
                                      rotation_axis_position = params['Scan']['CentreOfRotation'],
                                      rotation_axis_direction = params['Scan']['RotationAxis'])                                 
ag.set_angles(np.arange(params['Scan']['StartAngle'], params['Scan']['FinalAngle'], params['Scan']['AngleStep']))
ag.set_panel(params['Detector']['NumberOfPixels'],params['Detector']['PixelSize'])
show_geometry(ag)
# %%
from cil.io import TIFFStackReader
data_path = os.path.join(output_path, params['Scan']['OutFolder'])
data = TIFFStackReader(data_path).read()
data = AcquisitionData(data, geometry=ag)

islicer(data)
# %%
# Normaliser
max = data.max()
min = data.min()
data_norm = Normaliser(flat_field=max*np.ones((data.shape[1],data.shape[2])),
                       dark_field=(min-0.1*(max-min))*np.ones((data.shape[1],data.shape[2])))(data)
print(data_norm.min())
print(data_norm.max())
islicer(data_norm)

# %%
# Apply Beer-Lambert law
data_norm = TransmissionAbsorptionConverter(white_level=1.01)(data_norm)
print(data_norm.min())
print(data_norm.max())
islicer(data_norm)

# %%
# FluxNormaliser
data_norm = FluxNormaliser(flux=data_norm.max(), target=1)(data_norm)
print(data_norm.min())
print(data_norm.max())
islicer(data_norm)

# %%
for i in np.arange(data_norm.get_dimension_size('angle')):
    fname = os.path.join('test_path', "cylinder_simulation_" + str(i).zfill(4) + ".tif")
    imwrite(fname, data_norm.array[i].astype(np.float32), photometric='minisblack')
