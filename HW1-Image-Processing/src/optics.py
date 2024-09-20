import numpy as np
import scipy

def create_radial_distance_map(N_img):
    """  
    This should create 2-Dimensional matrix which calculates
    the radial distance from the center. I.e. the value 
    in the center should be 0 and then the values increase
    radial symmetric around the center.
    
    Helpful function are np.linspace, np.meshgrid
    and the np.sqrt function
    
    You can use Pythagorean theorem to calculat the radial distances.
    
    Args:
        N_img: Number of pixels of the distance map
    Return:
        R (float): Radial distance map as 2D array
    """
    
    i_indices, j_indices = np.indices((N_img, N_img))
    
    center = (N_img / 2) - 1
    
    return np.float32(np.sqrt((i_indices - center) ** 2 + (j_indices - center) ** 2))
 
    
    

def gaussian_psf(R,sigma):
    """  
    Gaussian PSFs creates a 2D matrix which follows
    a Gaussian Point-spread-function.
    
    Find more information on what a PSF is here:
    https://en.wikipedia.org/wiki/Point_spread_function
    
    Calculating this is quite simple and you don't
    require any additional packages.
    
    Simply apply the gaussian function 
    (https://en.wikipedia.org/wiki/Gaussian_function)
    to your radial distance map.
    
    The only parameter you'll need is sigma. Simply
    set the mean to 0.
    
    Args:
        R: radial distance map
        sigma: standard deviation of the gaussian PSF
    Return:
        PSF (float): A 2D-array with the gaussian PSF
    """
    intermediate_product = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-R**2/(2*sigma**2))
    normalized_product = intermediate_product/np.sum(intermediate_product)
    return normalized_product



def calc_angular_field_of_view(sensor_size_mm,focal_length):
    """
    Calcualte Angle of View
    
    According to: https://shuttermuse.com/calculate-field-of-view-camera-lens/

    Args:
        sensor_size_mm (float): sensor size of camera
        focal_length (float): focal length of camera
    Returns:
        angle of view of specific camera
    """
    return 2*np.arctan(sensor_size_mm/(2*focal_length))*(180/np.pi)


def calc_field_of_view(sensor_size_mm,o_obj,focal_length):
    """
    Calculate linear field of view at specific distance away from the lens
   
    
    You have to transform the equation given in https://en.wikipedia.org/wiki/Magnification#Photography

    Args:
        sensor_size_mm (float): sensor size of camera
        o_obj (float): distance where object is located
        focal_length (float): focal length of objetive lens
    Returns:
        linear of view of specific camera for both dimensions in mm
    """
    magnification = focal_length/(o_obj-focal_length)
    
    
    return magnification*sensor_size_mm
    
    


def calc_blur_radius(f,D,o_foc,o_obj):
    """
    Calcualte the blur radius according to
    lecture 1 - image formation

    Args:
        sensor_size_mm (float): sensor size of camera
        o_obj (float): distance where object is located
        focal_length (float): focal length of objetive lens
    Returns:
        angle of view of specific camera
    """
    blur_radius = D*np.abs((f*(o_foc-o_obj))/(o_obj*(o_foc-f)))
    
    return blur_radius

def crop_background_image_sensor_ratio(sensor_size_mm,img):
    """"
    
    This functions crops an image of arbitrary to size to a specific aspect ratio defined 
    by the sensor size of the camera
    
    1. Calculate aspect ratio of sensor: e.g., 24/36 = 2/3
    2. Calculate aspect ratio r of input image:
        Two cases: r > 2/3 or r<2/3
    Depending on those cases, you have to crop either width or height (think about which one)
    3. Depending on result of (2), calculate how much you have to crop
    4. Crop only either width or height dimension, depending on what you've calculated in 3
    5. You're done. I would use only numpy for this. Nothing else is needed

    
    Input:
    sensor_size (float) : array containing the height and width of the image sensor which defines the aspect ratio
    img (int,float) : the image that should be cropped to the specific aspect ratio
    Output:
    resized_img (int,loat): The cropped image that now has the correct aspect ratio
    """
    aspect_ratio_of_sensor = sensor_size_mm[0]/sensor_size_mm[1]
    aspect_ratio_of_image = img.shape[0]/img.shape[1]
    
    
    if aspect_ratio_of_image > aspect_ratio_of_sensor:
        needed_size = int(((img.shape[1])*sensor_size_mm[0]))
        crop = needed_size
        print(crop)     
        new_img = img[:crop,:]
    else:
        needed_size = int(((img.shape[0])*(1/aspect_ratio_of_sensor)))
        crop = needed_size
        print(crop)
        new_img = img[:,:crop]
    
    # print(new_img.shape)
    new_aspect_ratio = new_img.shape[0]/new_img.shape[1]
    # print(new_aspect_ratio)
    assert np.abs(new_aspect_ratio-aspect_ratio_of_sensor)<0.01
    
    return new_img

def convolve_image(img,PSF):
    """
    
    Convolve_image blurs the image with a PSF. Because we've defined very large blur 
    kernels (100x100 instead of small ones like 5x5) these Convolutions can take
    very long, especially when performed in spatial domain.
    
    The trick is to perform this convolution in frequency domain using the
    convolution theorem: https://en.wikipedia.org/wiki/Convolution_theorem
    
    Luckily, there are already python package which perform convolution
    in the frequency domain. 
    
    You can look e.g. at scipy.signal.fftconvolve for such a method that you
    can use in this functon.
    
    Input:
    img (float) : RGB array of the image to be blurred
    psf (float) : the point spread function we are blurring the image with
    Output:
    img_filtered (foat): The filtered image of same size as img
    """
    
    #apply the kernel to the image
    PSF_with_axis = PSF[:,:, np.newaxis] 
    img_filtered = scipy.signal.fftconvolve(img, PSF_with_axis, mode='same')
    return img_filtered
