import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import re
from os.path import join
from glob import glob

savedir = 'output'

def save_fig_as_png(figtitle):
    '''
    Saves the current figure into the output folder
    The figtitle should not contain the ".png".
    This helper function shoudl be easy to use and should help you create the figures 
    needed for the report
    
    The directory where images are saved are taken from savedir in "Code.py" 
    and should be included in this function.
    
    Hint: The plt.gcf() might come in handy
    Hint 2: read about this to crop white borders
    https://stackoverflow.com/questions/8218608/scipy-savefig-without-frames-axes-only-content
    
    '''
    output_folder = join(savedir, figtitle + '.png')
    plt.gcf()
    plt.savefig(output_folder, bbox_inches='tight', pad_inches=0)
    
    

def loadImage(imgFolder):
    """
    This function will read the images and save all the input images into an array. The exposure times in seconds must be parsed from the filenames. 
    
    Input
        imgFolder(str): The location of the folder of input images.
    Output
        rawImg(np.array) - uint8): m*n*3*k array. m, n is the size of each image, and k is the number of input images. 3 represents k images in three different channels.
        expTime(1D-array): k array of exposure times extracted from the filenames
        
    Please note that when using OpenCV to read input images, it will generate an m*n*3 array, and the sequence is BGR CHANNEL instead of RGB! 
    Hint: There is an openCV function to convert from BGR to RGB (google it for more information)
    
    """
    list_of_pngs = glob(imgFolder + '/*.png')
    rawImg = []
    expTime = []
    
    for img in list_of_pngs:
        image = cv2.imread(img)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rawImg.append(rgb)
        split_string = img.split('/')[-1].split('.png')[0]
        expTime.append(float(split_string))
    
    rawImg = np.array(rawImg)
    expTime = np.array(expTime)
    
    rawImg = rawImg.transpose(1,2,3,0)
    
    #order the flat array of exposure times
    order = np.argsort(expTime)
    expTime = expTime[order]
    rawImg = rawImg[:,:,:,order]
    
    return rawImg, expTime
    
    
    
    

def loadExposureTime(imgFolder):
    """
    This function will load the exposure time using the name of input images within the folder, then calculate the logorithm of loaded exposuretime.
    The name of each image is the expusure time in nanosecond. 
    So before calculating the log exposure time, please read the image, convert it to number from string, then convert it into second (1 second = 1e9 nanosecond).
    Input
        imgFolder:  A string. The location of the folder of input images.
    Output 
        expTime:  An (k,) array. k is the number of images. The exposure time.
        logExpTime: just the log of expTime
    """
    
    list_of_pngs = glob(imgFolder + '/*.png')
    expTime = []
    
    for img in list_of_pngs:
        split_string = img.split('/')[-1].split('.png')[0]
        expTime.append(float(split_string))
    
    expTime = np.array(expTime)
    logExpTime = np.log(expTime)
    
    return expTime, logExpTime
        

def create_measured_Z_values(rawImg, numSample= 1000, low = 0, high = 245):
    """
    This function will sample a subset of the Z values from the captured images dataset. 
    Output of this function will be the input of the gsolve function to calculate the camera response curve.
    Note: In order to accelerate the speed of the program, DO NOT USE ALL PIXELS IN THE IMAGE!
    In this function, you need to sample some of the pixels within the input images randomly. You can sample several hunderd or several thousand pixels.

    Input
        rawImg: m*n*3*k array. m, n is the size of each image, and k is the number of input images.
        numSample: number of sampled pixels. Default is 1000, you can use your own values.
        low: the lowest intensity value to trust before considering the pixel underexposed 
        high: the highest intensity value to trust before considering the pixel overexposed

    Output
        zValues: numSample*k*3 array. numSample is the number of sampled pixels, k is the number of images, and 3 represents RGB channels. It's the matrix of pixel values, 
    """
    
    m, n, _, k = rawImg.shape
    zValues = np.zeros((numSample, k, 3))
    count = 0
    while count < numSample:
        x = np.random.randint(0, m)
        y = np.random.randint(0, n)
        if np.all(rawImg[x, y, :, :] >= low) and np.all(rawImg[x, y, :, :] <= high):
            zValues[count] = rawImg[x, y, :, :].T
            count += 1
        else:
            continue

    return zValues
    
   

