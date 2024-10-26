import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from src.gsolve import gsolve
from src import loaddata
from src import recradmap
from skimage.color import rgb2lab, rgb2yuv

def clip_hdr(hdr):
    """
    Clip the hdr image so that it displays correctly
    
    We implement this function for you. However, this might not work perfectly depending
    how you data looks like an you might to adapt this slightly, if the results don't work.
    
    """
    counts, bins = np.histogram(hdr.ravel(),bins=255);
    bins = bins[0:-1]
    cumsum = np.cumsum(counts)
    cumsum = cumsum/cumsum.max()
    idxs = np.where(cumsum>0.99)
    val = bins[idxs[0][0]]


    hdr = hdr / val

    hdr[hdr>1] = 1
    
    return hdr,bins,cumsum


def scaleBrightness(E):
    """
    Brightness scaling function, which will scale the values on the radiance map to between 0 and 1

    Args:
        E: An m*n*3 array. m*n is the size of your radiance map, and 3 represents R, G and B channel. It is your plotted Radiance map (don't forget to use np.exp function to get it back from logorithm of radiance!)
    Returns:
        ENomrMap: An m*n*3 array. Normalized radiance map, whose value should between 0 and 1
    """
    ENormMap = np.zeros_like(E)
    ENormMap[:,:,0] = (E[:,:,0] - np.min(E[:,:,0])) / (np.max(E[:,:,0]) - np.min(E[:,:,0]))
    ENormMap[:,:,1] = (E[:,:,1] - np.min(E[:,:,1])) / (np.max(E[:,:,1]) - np.min(E[:,:,1]))
    ENormMap[:,:,2] = (E[:,:,2] - np.min(E[:,:,2])) / (np.max(E[:,:,2]) - np.min(E[:,:,2]))
    
    return ENormMap
    
    

def apply_gamma_curve(E, gamma= 0.4):
    """
    apply gamma to the curve through raising E to the gamma.

    Args:
        E: An m*n*3 array. m*n is the size of your radiance map, and 3 represents R, G and B channel. It is your plotted Radiance map (don't forget to use np.exp function to get it back from logorithm of radiance!)
        gamma: a float value that is representative of the power to raise all E to.
    Returns:
        E_gamma: E modified by raising it to gamma.
    """
    E_gamma = E**gamma
    
    return E_gamma
    
    
def convert_rgb2gray(rgb):
    """
    TODO: IMPLEMENT ME
    
    rgb2gray converts RGB values to grayscale values by forming a weighted
    sum of the R, G, and B components:

    0.2989 * R + 0.5870 * G + 0.1140 * B 
    
    
    These values come from the BT.601 standard for use in colour video encoding,
    where they are used to compute luminance from an RGB-signal.
    
    Find more information here:
    https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.601-7-201103-I!!PDF-E.pdf

    Args:
        input (nd.array): 3-dimensional RGB where third dimension is ordered as RGB
    Returns:
        output (np.ndarray): Gray scale image of RGB weighted by weighting function from above
    """
    
    output = np.zeros((rgb.shape[0], rgb.shape[1]))
    output = 0.2989 * rgb[:,:,0] + 0.5870 * rgb[:,:,1] + 0.1140 * rgb[:,:,2]
    
    return output

def tone_mapping(radiance, a= 0.1):
    """
    This function completes tone mapping using global tone mapping operator from Reinhard'02 paper.
    
    Args:
        ENormMap: An m*n*3 array. Normalized Radiance map, whose value should between 0 and 1.
        a: A scalar value. You can pick different values and see different results.
    
    Returns:
        MMat: An m*n array. Calculated M matrix, which is used to generate final HDR image.
    """
    output = np.zeros_like(radiance)
    graymap = convert_rgb2gray(radiance)
    log_avg_exp = np.exp(np.mean(np.log(graymap)))
    T = (a / log_avg_exp)*graymap
    L_tone = (T*(1+(T/(T.max()**2))))/ (1+T)
    M = L_tone/graymap


    return M

def compute_hdr_image(rawImg, expTime, lam = 100, gamma = 0.35, a = 0.2):
    """
    putting all the functions together, this funtion creates the hdr image 
    using all the previous functions.

    Args:
        rawImg: the raw img
        expTime: the exposure time
        lam: lambda
        gamma: the gamma value to apply to image.
        a: A scalar value. You can pick different values and see different results.

    Returns:
        hdr: the hdr image
        radiance_gamma: the E_gamma, gamma curve applied to image through apply_gamma_curve
    """

    brightness = loaddata.create_measured_Z_values(rawImg)
    
    log_exposure_time = np.log(expTime)
    solveG = np.zeros((256,3))
    log_exposure = np.zeros((1000,3))
    
    for k in range(3):
        solveG[:,k], log_exposure[:,k] = gsolve(brightness[:,:,k], log_exposure_time, lam)
    
    log_radiance = recradmap.get_log_radiance_map(rawImg, log_exposure_time, solveG)
    
    radiance = np.exp(log_radiance)
    radiance_normalized = scaleBrightness(radiance)
    
    radiance_gamma = apply_gamma_curve(radiance_normalized, gamma=gamma)

    M = tone_mapping(radiance_normalized,a=a)
    
    hdr =  radiance_normalized * M[:,:,None]

    hdr,_,_ = clip_hdr(hdr);

            
        
    
    return hdr, radiance_gamma