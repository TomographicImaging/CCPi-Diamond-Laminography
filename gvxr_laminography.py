# %%
import os
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'serif',
         'size'   : 15
       }
matplotlib.rc('font', **font)
from gvxrPython3 import gvxr

from gvxrPython3 import gvxr2json
from gvxrPython3.JSON2gVXRDataReader import *
from cil.recon import FBP
from cil.plugins.astra.processors import FBP as astra_FBP
from cil.processors import TransmissionAbsorptionConverter
from cil.utilities.display import show_geometry, show2D
from cil.utilities.jupyter import islicer
from scipy.spatial.transform import Rotation as R

# %%
tilt = 30
output_path = "output_data"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# %% Create the experiment geometry
# Set up the source
print("Create an OpenGL context")
gvxr.createOpenGLContext()
print("Set up the beam")
gvxr.setSourcePosition(0.0,  -40.0, 0.0, "mm")
gvxr.setMonoChromatic(500, "keV", 1000)
gvxr.useParallelBeam()

# Set up the detector
print("Set up the detector")
gvxr.setDetectorPosition(0.0, 40.0, 0.0, "mm")
gvxr.setDetectorUpVector(0, 0, 1)
gvxr.setDetectorNumberOfPixels(300, 300)
gvxr.setDetectorPixelSize(0.5, 0.5, "mm")

# %%
# Locate the sample STL file
fname =  "Turtle_Singlecolor.stl"

# Load the sample data
if not os.path.exists(fname):
    raise IOError(fname)

print("Load the mesh data from", fname)
gvxr.loadMeshFile("Turtle", fname, "mm")

print("Move ", "Turtle", " to the centre")
gvxr.moveToCentre("Turtle")
gvxr.applyCurrentLocalTransformation("Turtle")

# %%
# Chosse a density for the sample
# Carbon (Z number: 6, symbol: C)
gvxr.setElement("Turtle", 6)
gvxr.setElement("Turtle", "C")

# Compute an X-ray image
print("Compute an X-ray image")
gvxr.displayScene()
x_ray_image = np.array(gvxr.computeXRayImage()) / gvxr.getTotalEnergyWithDetectorResponse()
show2D(x_ray_image)
# %%
# Tilt the turtle and compute an x-ray image
gvxr.rotateNode("Turtle", tilt, 1, 0, 0)

# Compute an X-ray image
print("Compute an X-ray image")
x_ray_image = np.array(gvxr.computeXRayImage()) / gvxr.getTotalEnergyWithDetectorResponse()
show2D(x_ray_image)

# %% Check we can still see the turtle as we rotate around the CT rotation axis
gvxr.rotateNode("Turtle", 90, 0, 0, 1)
# Compute an X-ray image
print("Compute an X-ray image")
x_ray_image = np.array(gvxr.computeXRayImage()) / gvxr.getTotalEnergyWithDetectorResponse()
show2D(x_ray_image)

# %% Calcualte the rotated axis
axis = np.array([1, 0, 0])
axis = axis / np.linalg.norm(axis)
rotation = R.from_rotvec(np.radians(tilt) * axis)
vector = np.array(gvxr.getDetectorUpVector())
rotated_vector = rotation.apply(vector)
print("Rotated vector:", rotated_vector)
# %% Create CT projections
number_of_projections = int(300*(np.pi/2))
gvxr.computeCTAcquisition(os.path.join(output_path, "projections-" + str(number_of_projections)), # the path where the X-ray projections will be saved.
                                                                    # If the path is empty, the data will be stored in the main memory, but not saved on the disk.
                                                                    # If the path is provided, the data will be saved on the disk, and the main memory released.
                          os.path.join(output_path, "screenshots-" + str(number_of_projections)), # the path where the screenshots will be saved.
                                                                    # If kept empty, not screenshot will be saved.
                          number_of_projections, # The total number of projections to simulate.
                          0, # The rotation angle corresponding to the first projection.
                          True, # A boolean flag to include or exclude the last angle. It is used to calculate the angular step between successive projections.
                          360,
                          0, # The number of white images used to perform the flat-field correction. If zero, then no correction will be performed.
                          *gvxr.getCentreOfRotationPositionCT("mm"), # The location of the rotation centre.
                          "mm", # The corresponding unit of length.
                          *rotated_vector, # The rotation axis
                          True # If true the energy fluence is returned, otherwise the number of photons is returned
                               # (default value: true)
)
# %% Save the current simulation states in a JSON file.
json_fname = os.path.join(output_path, "simulation-" + str(number_of_projections) + "_" +str(tilt)+ "_degrees.json")
gvxr2json.saveJSON(json_fname)
# %% Read the simulated data with CIL.
from gvxrPython3.JSON2gVXRDataReader import *
reader = JSON2gVXRDataReader(json_fname)
data_tilt = reader.read()
# %% Apply Beer-Lambert law
data_tilt = TransmissionAbsorptionConverter()(data_tilt)
# %% Check the data and geometry look right in CIL
data_tilt.reorder('tigre')
show_geometry(data_tilt.geometry)
print(data_tilt.geometry)
islicer(data_tilt)
# %% Compare CIL recon FBP with astra
from cil.recon import FBP
data_tilt.reorder('astra')
fbp = FBP(data_tilt, backend='astra')
recon_tilt = fbp.run()
recon_tilt.apply_circular_mask(0.9)
show2D(recon_tilt)
# %% and astra plugin
from cil.plugins.astra.processors import FBP as astra_FBP
fbp = astra_FBP(data_tilt.geometry.get_ImageGeometry(), data_tilt.geometry)
recon_astra = fbp(data_tilt)
recon_astra.apply_circular_mask(0.9)
show2D([recon_tilt,recon_astra]) 
# %% scroll through the reconstruction
islicer(recon_astra)