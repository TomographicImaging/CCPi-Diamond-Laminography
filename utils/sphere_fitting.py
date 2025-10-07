from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
import matplotlib.pyplot as plt
import numpy as np
from cil.utilities.display import show2D
def circle(x, y, x0, y0, r):
    return (x-x0)**2 + (y-y0)**2 - r**2 

from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
def find_circles(data_abs,sigma, output=False):
    edges = canny(data_abs, sigma=sigma)
    min_radius = 76
    max_radius = 84
    radius_step = 2
    hough_radii = np.arange(min_radius, max_radius, radius_step)
    hough_res = hough_circle(edges, hough_radii)
    if output:
        show2D([data_abs, edges, hough_res], num_cols=3)
    return hough_res, hough_radii

def fit_circles(data_abs, hough_res, hough_radii, output=False):
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
    if output:
        print(cx, cy, radii)
        plt.figure()
        plt.imshow(data_abs)
        x, y = np.meshgrid(range(data_abs.shape[0]),range(data_abs.shape[1]))
        plt.contour(x,y,circle(x,y,cx[0],cy[0],radii[0]),[0])
    return cx, cy, radii

# x and y are your input coordinate arrays
def fit_ellipse(x, y):
    coords = np.column_stack((x, y))  # combine into Nx2 array
    model = EllipseModel()
    model.estimate(coords)
    return model.params  # (xc, yc, a, b, theta)