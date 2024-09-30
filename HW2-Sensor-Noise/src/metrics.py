import rawpy
import numpy as np
import glob
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

def calc_mean(imgs):
    """
    calculates the mean across all time stamps of the images with a specific filter
    args:
        imgs(np.ndarray): the images separated into rgb vals, whos means you are trying to 
        calculate.
    output:
        mean_imgs(np.ndarray): the mean value of images in relation to their bayer pattern
        filters. size should be (x dimension * y dimension * r g b)
    """
    
    return np.mean(imgs, axis=3)
    

def calc_var(imgs):
    """
    calculates the variance across all time stamps of the images with a specific filter
    args:
        imgs(np.ndarray): the images separated into rgb vals, whos variance you are trying to 
        calculate.
    output:
        var_imgs(np.ndarray): the variance value of images in relation to their bayer pattern
        filters. size should be (x dimension * y dimension * r g b)
    """
    #ddof = 1 to get the unbiased estimator
    ddof = 1
    return np.var(imgs, axis=3, ddof=ddof)

def fit_linear_polynom_to_variance_mean(mean, var,th=200):
    """
    finds the polyfit between mean and variance which you calculate in the previous functions, 
    mean and var.
    
    mean(np.ndarray): the mean of the img filtered into rgb values - #(M, N, Num_channel, Num_gain)
    var(np.ndarray): the variance of the img filtered into rgb values - #(M, N, Num_channel, Num_gain)
    
    output:
          gain(nd.array): the slope of the polynomial fit. Should be of shape (Num_channel,Num_gain) for our data
          delta(nd.array): the y-intercept of the polynomial fit. Should be of shape (Num_channel,Num_gain) for our data
    """
    
    M, N, Num_channel, Num_gain = mean.shape
    gain = np.zeros((Num_channel,Num_gain))
    delta = np.zeros((Num_channel,Num_gain))
    
    for i in range(Num_channel):
        for j in range(Num_gain):
            good_idx = np.where(mean[:,:,i,j] < th)
            gain[i,j], delta[i,j] = np.polyfit(mean[good_idx[0],good_idx[1],i,j], var[good_idx[0],good_idx[1],i,j], 1)
    
    return gain, delta
            

def fit_linear_polynom_to_read_noise(delta, gain):
    """
    finds the polyfit between mean and variance which you calculate in the previous functions, 
    mean and var.
    
    sigma(np.ndarray): the total read noise filtered into rgb values - #(Num_Channel,Num_gain)
    gain(np.ndarray): the estimated camera gain filtered into rgb values - #(Num_Channel,Num_gain)
    
    output:
          sigma_read(np.ndarray): the slope of the linear fit - #(Num_Channel)
          sigma_ADC(np.ndarray): the y-intercept of the linear fit - #(Num_Channel)
    """
    Num_Channel, Num_gain = delta.shape
    
    sigma_read = np.zeros(Num_Channel)
    sigma_ADC = np.zeros(Num_Channel)
    
    for i in range(Num_Channel):
        gain_squared = gain[i,:]**2
        sigma_read[i], sigma_ADC[i] = np.polyfit(gain_squared, delta[i,:], 1)
    
    return sigma_read, sigma_ADC
    
    
    
def calc_SNR_for_specific_gain(mean,var):

    """
    Calculate the SNR (mean / stddev) vs. the mean pixel intensity for a specific gain setting. You will need to bin the mean values into the range [0,255] so that you can compute SNR for a discrete set of values. 
    
    mean(np.ndarray): the mean of the img filtered into rgb values - #(M, N, Num_gain)
    var(np.ndarray): the variance of the img filtered into rgb values - #(M, N, Num_gain)
    
    output:
          SNR(np.ndarray): the computed SNR vs. mean of the captured image dataset - #(255, Num_gain)
    """
    
    M, N = mean.shape
    SNR = np.zeros((255))

    binned_means_indexes = np.digitize(mean, bins=np.arange(0, 256))
    
    
    for j in range(255):
        indices = np.where(mean == j)
        if len(indices[0]) > 0: 
            std_devs = np.sqrt(var[indices])  
            avg_std_dev = np.mean(std_devs) 
            
            if avg_std_dev > 0:
                SNR[j] = j / avg_std_dev  
            else:
                SNR[j] = 0  
        else:
           SNR[j] = 0  

    return SNR