import rawpy
import numpy as np
import glob
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_with_colorbar(img,vmax=0):
    """
    args:
        vmax: The maximal value to be plotted
    """
    ax = plt.gca()
    if(vmax == 0):
        im = ax.imshow(img, cmap= 'gray')
    else:
        im = ax.imshow(img, cmap= 'gray',vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    

def plot_input_histogram(imgs,sensitivity):
    """
    
    The imgs variable consists of 1 image captured per different camera sensitivity (ISO) settings. plot_input_histogram
    visualize the histograms for each image in a subplot fashion

    
    args:
        imgs(np.ndarray): 3-dimensional array containing one image per intensity setting (not all the 200)
    
    """
    print(imgs.shape)
    num_histograms = imgs.shape[2]
    index = 0
    
    #we should iterate here bc we have rows we want to set up
    num_rows = (num_histograms + 1) // 2
    plt.figure(figsize=(15, 5 * num_rows))

    for i in range(num_histograms):
        plt.subplot(num_rows, 2, i + 1)
        plt.hist(imgs[:, :, i].ravel(), bins=256, alpha=0.8, ec="k")
        plt.title('Sensitivity: ' + str(sensitivity[i]))
        plt.xlabel('Intensity')
        plt.ylabel('Count')

    plt.tight_layout()
    
    plt.show()
    return
    
        
def plot_histograms_channels(img,sensitivity):
    """
    
    Plots the histogram for each channel in a subplot (1 row, 3 cols)
    
    args:
        img(np.ndarray): The RGB image
        sensitivity(float): The gain settings of the img series
    
    """
    

    fig, ax = plt.subplots(1,3,figsize=(15,5))
    plt.suptitle("Histogram for Sensitivity Level: {}".format(sensitivity))
    
    flattened = img[:, :, 0].flatten()
    ax[0].hist(flattened, bins=80, alpha=0.5, range=(0, 255))
    ax[0].set_title('Sensitivity for Red Channel')
    ax[0].set_ylabel('Count')
    
    flattened_2 = img[:, :, 1].flatten()
    ax[1].hist(flattened_2, bins=80, alpha=0.5, range=(0, 255))
    ax[1].set_title('Sensitivity for Green Channel')
    ax[1].set_ylabel('Count')
    
    flattened_3 = img[:, :, 2].flatten()
    ax[2].hist(flattened_3, bins=80, alpha=0.5, range=(0, 255))
    ax[2].set_title('Sensitivity for Blue Channel')
    ax[2].set_ylabel('Count')
    
    

        
def plot_input_images(imgs,sensitivity):
    """
    
    The dataset consists of 1 image captured per different camera sensitivity (ISO) settings. Lets visualize a single image taken at each different sensitivity setting
    
    Hint: Use plot_with_colorbar. Use the vmax argument to have a scale to 255
    (if you don't use the vmax argument)
    
    args:
        imgs(np.ndarray): 3-dimensional array containing one image per intensity setting (not all the 200)
        sensitivity(np.ndarray): The sensitivy (gain) vector for the image database
    
    """
    
    num_rows = (len(sensitivity) + 1) // 2
    plt.figure(figsize=(15, 5 * num_rows))

    for i in range(len(sensitivity)):
        plt.subplot(num_rows, 2, i + 1)
        plot_with_colorbar(imgs[:, :, i], vmax=255)
        plt.title('Sensitivity Lvl {}'.format(sensitivity[i]))

    plt.tight_layout()
    plt.show()
        
    

def plot_rgb_channel(img, sensitivity):
    
    """
    Plots the RGB channels of an image in separate subplots.

    args:
        img(np.ndarray): The RGB image
        sensitivity(float): The gain settings of the img series
    """
    fig,ax = plt.subplots(1,3,figsize=(25,15))
    fig.suptitle("Sensitivity {}".format(sensitivity))
    
    ax[0].set_title('Red')
    plt.sca(ax[0])
    plot_with_colorbar(img[:,:,0],vmax=img[:, :, 0].max())
    

    ax[1].set_title('Blue')
    plt.sca(ax[1])
    plot_with_colorbar(img[:,:,2],vmax=img[:, :, 2].max())

    
    ax[2].set_title('Green')
    plt.sca(ax[2])
    plot_with_colorbar(img[:,:,1],vmax=img[:, :, 1].max())  
    plt.show()

    
    
def plot_images(data, sensitivity, statistic,color_channel):
    """
    this function should plot all 3 filters of your data, given a
    statistic (either mean or variance in this case!)

    args:

        data(np.ndarray): this should be the images, which are already
        f    iltered into a numpy array.

        statsistic(str): a string of either mean or variance (used for
        titling your graph mostly.)

    returns:

        void, but show the plots!

    """
    map_word_to_color = {0: 'Red', 1: 'Green', 2: 'Blue'}
    fig, ax = plt.subplots(2,3,figsize=(20,15))
    plt.suptitle("{} - Image | Color Channel : {}".format(statistic, map_word_to_color[color_channel]))

    counter = 0
    for i in range(2):
        for j in range(3):
            # ax[i,j].imshow(data[:,:,color_channel, counter],cmap='gray')
            plt.sca(ax[i,j])
            plot_with_colorbar(data[:,:,color_channel, counter])
            ax[i,j].set_title('{} for {}'.format(statistic, sensitivity[counter]))
            counter += 1
    
    plt.tight_layout()
        
        
    
def plot_relations(means, variances, skip_pixel, sensitivity, color_idx):
    """
    this function plots the relationship between means and variance. 
    Because this data is so large, it is recommended that you skip
    some pixels to help see the pixels.

    args:
        means: contains the mean values with shape (200x300x3x6)
        variances: variance of the images (200x300x3x6)
        skip_pixel: amount of pixel skipped for visualization
        sensitivity: sensitivity array with 1x6
        color_idx: the color index (0 for red, 1 green, 2 for blue)

    returns:
        void, but show plots!
    """
    fig, ax = plt.subplots(2,3,figsize=(15,5))
    counter = 0
    for i in range(2):
        for j in range(3):
            ax[i,j].scatter(means[::skip_pixel,::skip_pixel,color_idx,counter],variances[::skip_pixel,::skip_pixel,color_idx,counter].flatten())
            ax[i,j].set_title('Mean vs. Variance for {}'.format(sensitivity[counter]))
            ax[i,j].set_xlabel('Mean')
            ax[i,j].set_ylabel('Variance')
            counter += 1
    
        
def plot_mean_variance_with_linear_fit(gain,delta,means,variances,skip_points=50,color_channel=0):
    """
        this function should plot the linear fit of mean vs. variance against a scatter plot of the data used for the fitting 
        
        args:
        gain (np.ndarray): the estimated slopes of the linear fits for each color channel and camera sensitivity

        delta (np.ndarray): the estimated bias/intercept of the linear fits for each color channel and camera sensitivity

        means (np.ndarray): the means of your data in the form of 
        a numpy array that has the means of each filter.

        variances (np.ndarray): the variances of your data in the form of 
        a numpy array that has the variances of each filter.
        
        skip_points: how many points to skip so the scatter plot isn't too dense
        
        color_channel: which color channel to plot

    returns:
        void, but show plots!
    """
    map_word_to_color = {0: 'Red', 1: 'Green', 2: 'Blue'}
    fig, ax = plt.subplots(2,3,figsize=(15,5))
    plt.suptitle('Mean vs. Variance for Color Channel {}'.format(map_word_to_color[color_channel]))
    counter = 0
    for i in range(2):
        for j in range(3):
            print(gain[counter],delta[counter])
            ax[i,j].scatter(means[::skip_points,::skip_points,color_channel,counter],variances[::skip_points,::skip_points,color_channel,counter].flatten())
            ax[i,j].plot(means[::skip_points,::skip_points,color_channel,counter].ravel(),gain[color_channel,counter]*means[::skip_points,::skip_points,color_channel,counter].ravel() + delta[color_channel,counter],color='red', label='Line of Best Fit')
            ax[i,j].set_title('Mean vs. Variance for {}'.format(counter+1))
            ax[i,j].set_xlabel('Mean Intensity')
            ax[i,j].set_ylabel('Variance')
            counter += 1
    

    
def plot_read_noise_fit(sigma_read, sigma_ADC, gain, delta, color_channel=0):
    """
        this function should plot the linear fit of read noise delta vs. gain plotted against the data used for the fitting 
        
        args:
        sigma_read (np.ndarray): the estimated gain-depdenent read noise for each color channel of the sensor 

        sigma_ADC (np.ndarray): the estimated gain-independent read noise for each color channel of the sensor

        gain (np.ndarray): the estimated slopes of the linear fits of mean vs. variance for each color channel and camera sensitivity

        delta (np.ndarray): the estimated bias/intercept of the linear fits of mean vs. variance for each color channel and camera sensitivity

        color_channel: which color channel to plot
        
    returns:
        void, but show plots!
    """
    
    plt.figure(figsize=(15,5))
    plt.scatter(gain[color_channel,:],delta[color_channel,:])
    plt.plot(gain[color_channel,:],sigma_read[color_channel]*gain[color_channel,:]**2 + sigma_ADC[color_channel],color='red')
    
    plt.xlabel('Gain')
    plt.ylabel('Noise')
    
    plt.title('Read Noise vs. Gain')
    plt.show()
 