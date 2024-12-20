import rawpy
import numpy as np
import glob
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

def plot_overlayed_hist(data,loc,sensitivity,size):
    """
    plot the histogram (PDF) of pixel intensities for each sensitivity setting on the camera.
    
    Hint: How to make nice histograms ?!
    1. Use the plt.his function
    2. Use alpha around 0.8
    3. you can use the option ec="k"
    4. You can use density=True
    
    args:
        data(np.ndarray): (H, W, #colors, #images, #sensitivity) array of captured pixel intensities
        loc(np.ndarray): (y,x) 2D location of pixel to plot histogram
        sensitivity(np.ndarray): (#sensitivity) array of camera sensitivity settings
        size(np.ndarray): (h,w) of pixels to include in histogram 
        
    output:
        void, but you should plot the graphs! hint: try looking at plt.hist
    """
    # print(data.shape)
    alpha = 0.8
    ec = "k"
    density = True
    bins = 10
    H, W, colors, images, _ = data.shape
    y, x = loc
    h, w = size
    
    #iterate over the sensitivity settings, not rgb
    for i in range(len(sensitivity)):
        plt.hist(data[y:y+h, x:x+w, :, :, i].ravel().T, bins=bins, alpha=alpha, ec=ec, density=density,label='Sensitivity: {} '.format(sensitivity[i]))
        # plt.title('Sensitivity: ' + str(sensitivity[i]))
        plt.xlabel('Pixel Intensity')
        plt.ylabel('PDF')
    
    plt.legend()

    return None


def get_pixel_location(img_shape,N_x,N_y):
    """
    
    Takes the shape of an image and number of to be gridded points in X and Y direction 
    to sample equally spaced points on the 2D-grid
    
    We want to exclude points at the boundaries.
    
    E.g., if our image has 100 x 100 and we want to sample 3 x 4 points we would do the following 
    
    25 50 75 for the x-coordinate
    and
    20 40 60 80 for the y-coordinate
    
    Those 2 vectors then need to converted into 2 matrices for X and Y positions (use meshgrid)
    
    the following numpy functions can come in handy to develop this function:
    
    1. np.arange
    2. np.meshgrid
    3. np.round to cast to integer values 
    4. np.astype(np.uint16) as we want integer values for the coordinates
    
    Input:
    
    Output:
    
    """
    
    H, W, _, _, _ = img_shape
    x = np.round(np.linspace(0, W, N_x+2)[1:-1]).astype(np.uint16)
    y = np.round(np.linspace(0, H, N_y+2)[1:-1]).astype(np.uint16)
    X, Y = np.meshgrid(x, y)

    return X.T, Y.T