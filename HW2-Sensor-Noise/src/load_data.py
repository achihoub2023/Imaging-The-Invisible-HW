import numpy as np
import glob
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

def convert_uint16_to_uint8(img):
    img_new = (img.astype(np.float32)*4)/2**8
    return np.uint8(img_new)

def load_data(folder,numImages=200,height=600,width=800):
    """
    this function should load your images as a numpy array

    The images are stored as "Raw" files. There are many ways how to load a raw images.
    Do a google research, there are many resources that will help you.
    
    You can e.g. use np.fromfile. However, this gives only a 1D-array back, hence you have to
    resize it after
    
    All information that you need for the images hrer are the following:
    
    Size of images that are used in this assignemnt are: 600 x 800
    Dtype of Raw images: np.uint8
    
    You can use e.g. the glob function with the wildcard (*) parameter to get a list of
    only the raw files. Then you only have to iterate through this list and load each image and
    store it.
    
    
    args:
        folder: Only the foldername inside 'pic//HW2_data//. Should not include the complete path!
        numImages: number of images you want to load (might come in handy for debugging your code)
        height: height of the image in pixels
        width: width of the images in pixels
          
    returns:
        imgs (np.array): The 200 images as monochromatic images in uint8 type format
    """
    imgs = []
    full_path = 'pic//HW2_Data//' + folder + '/*.Raw'
    list_of_files = glob.glob(full_path)
    # print(os.path.isdir('./pic//HW2_data//'+folder))
    for file in list_of_files:
        img = np.fromfile(file, dtype=np.uint8)
        img = img.reshape(height,width)
        imgs.append(img)
    
    images_as_array = np.array(imgs)
    transposed = np.transpose(images_as_array, (1,2,0))
    
    return transposed
    
def load_dataset():
    """
    Should load all data into 2 large numpy arrays
    
    You probably need about 2-3 GB of free RAM on your computer
    
    The sensitivies in the folders are [0,1,3,9,14,18]
    
    returns:
        dark (np.array): A 4-dimensional numpy array with dim = (600, 800, 200, 6) for the dark images
        imgs (np.array): A 4-dimensional numpy array with dim = (600, 800, 200, 6) for the white images
        sensitivy (np.array): A numpy array containing [0,1,3,9,14,18]
    
    """
    sensitivity = np.array([0,1,3,9,14,18])
    dark = []
    imgs = []
    
    #generate all possilbe combinations of the strings and make a list of the paths
    for s in sensitivity:
        dark_path = "dark" + str(s)
        gain_path = "gain" + str(s)
        dark.append(load_data(dark_path))
        imgs.append(load_data(gain_path))    
    
    dark = np.array(dark)
    imgs = np.array(imgs)
    
    dark = np.transpose(dark, (1,2,3,0))
    imgs = np.transpose(imgs, (1,2,3,0))
    
    
    return dark,imgs,sensitivity
    