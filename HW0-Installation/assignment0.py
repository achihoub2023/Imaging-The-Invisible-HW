# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Homework 0 - Environment Setup and Introduction to Python

# %% [markdown]
# Welcome to your first homework assignment! In this notebook, you'll get familiar with the Jupyter Lab environment and the assignment structure for this course. You'll also have the opportunity to practice some basic Python and NumPy operations.
#
# Please go through this Jupyter notebook and implement the missing functions in the `src/code.py` file. After implementing the functions, add, commit, and push your changes to your GitHub repository.
#
# Once you've completed the assignment, prepare your first report using the provided templates. Remember to submit this report to Canvas as well.

# %%
# Importing packages

import numpy as np  # NumPy for linear algebra and mathematical operations
import random  # Random for creating random numbers

# Information on autoreload which will be important to update your modules
# when you change them so that they work in the Jupyter notebook
# https://ipython.org/ipython-doc/3/config/extensions/autoreload.html
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# Let's check if the other required packages have been installed successfully. If any of these packages cannot be imported, please install them manually using PIP or CONDA. The easiest way to do this is to search "Install package name pip python" on Google, look for the installation guide, and install it through a simple command in your shell.

# %%
import rawpy
import numpy as np
import glob
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# %% [markdown]
# # Import the functions that you have to implement

# %%
import src.code as code

# %% [markdown]
# # Implement the functions and test if they work

# %%
code.sum_numbers(1, 2)

# %%
code.multiply_numbers(2, 3)

# %%
code.create_add_matrix(np.ones((3, 3)))

# %%
code.indexing_aggregation([1, 2, 4, 4, 5], 3)

# %%
A = np.array([[4, 7], [2, 6]])

# %%
A_inv = code.matrix_inverse(A)
print(A_inv)

# %%
# Verify that this is the actual inverse
np.matmul(A, A_inv)
# The result should be a matrix with only ones on the diagonal
# %% [markdown]
# ## Additional Python and NumPy Practice

# %% [markdown]
# In this section, you'll have the opportunity to showcase your Python and NumPy skills by writing your own code. Feel free to use any concepts or techniques you've learned previously in school or college. If you need inspiration, consider searching online for interesting Python or NumPy examples.
#
# Your task is to write 5-10 minutes worth of Python and/or NumPy code. This can be a single coherent example or a series of smaller, unrelated examples. The goal is to demonstrate your understanding of the language and its libraries.
#
# Some ideas to get you started:
# - Implement a classic algorithm like the Fibonacci sequence or the Sieve of Eratosthenes
# - Create a function that calculates the factorial of a number
# - Write a script that generates a random password of a specified length
# - Use NumPy to create and manipulate a 3D array
# - Implement a basic linear regression model using NumPy
#
# Remember to include comments in your code to explain what each part does. If you use any external resources (like StackOverflow or Python documentation), be sure to cite them in your comments.

# %% [markdown]
# # Your code here

def manipulate_random_3d_array():
    new_array = np.random.rand(3, 3, 3)
    
    #add laplacian noise to this 3D array
    new_array = new_array + np.random.laplace(0, 1, (3, 3, 3))
    
    #get the sum of its columns
    column_sum = np.sum(new_array, axis=0)
    
    #get the sum of its rows
    row_sum = np.sum(new_array, axis=1)
    
    #get the sum of its depth
    depth_sum = np.sum(new_array, axis=2)
    
    #save the array to a file
    np.save('3d_array.npy', new_array)
    
    return column_sum, row_sum, depth_sum
    
    

# %% [markdown]
# Briefly explain what your code does and how it works. If you used any external resources, cite them here.
# This code generates a 3D array of random numbers and adds laplacian noise to it. It then calculates the sum of the columns, rows, and depth of the array. Finally, it saves the array to a file to perserve it for future use. The output is three variables containing the sums in questions.
# (Write 2-3 sentences)

# %% [markdown]
# -------------
# -------------

# %% [markdown]
# ## <span style="color:indigo">(R) Questions: Report</span>
#
# Answer each of these questions in *your own* words:
#
# You can revisit the first lecture or use Google to find information, but don't forget to cite your sources.
#
# (Try to write less than 5 sentences for each question)
#
# 1. What is Computational Photography?
# Computational photography is a field that involves the use of digital processes to refine and generate traditional photographs . It involves the use of sensing techniques, algorithms, and image processing to enhance the quality of images. 
# 2. Why are scientists/engineers/industry interested in Computational Photography? Can you name a few examples?
# Computational photography is interesting to multiple parties since images are an important modality to convey visual information. It is useful in remote sensing, medical imaging, and computer vision.
# 3. What do you think are the differences between Computational Photography, Computational Imaging, Optics, Computer Graphics, and Computer Vision?
# The differences between computational photography, computational imaging, optics, graphics, and vision is that computational photography focuses on methods to enhance traditional photographs, computational imaging focuses on the use of algorithms to improve and collect images, optics focuses on the physics of light, graphics focuses on the creation of images, and vision focuses on the interpretation of images.
# 4. What do you think are important skills/tools to have when you work in Computational Photography?
# Important skills that are useful in computational photography are python programming, traditional image formation knowledge, and a good understanding of linear algebra.
# 5. Why did you decide to take Computational Photography? Are you a photographer? Do you have experience in Computer Vision? Do you like imaging?
# I took this class since I am interested in imaging, photography, and computer vision. I do research in computer vision, so I think I am well-suited for this class. I am not a professional photographer, but I do enjoy taking pictures.
# 6. To familiarize yourself with including equations (in LaTeX/Word), include 2 equations from previous courses (Optics/Engineering/Signal Processing) that you might think become important for Computational Photography. Describe the formula in 1-2 sentences and make sure to cite the source.
#    - These links might be useful to refresh your memory: [Link 1](https://www.newport.com/n/optics-formulas), [Link 2](https://www.dummies.com/education/science/physics/optics-for-dummies-cheat-sheet/)

"""
\begin{equation}
Snell's Law: n_1sin(\theta_1) = n_2sin(\theta_2)
\end{equation}
This equation describes the relationship between the angles of incidence and refraction when light passes through different mediums. It is an important formula in optics and is used to calculate the refractive index of a medium.
"""

"""
\begin{equation}
\mathbf{P} = \mathbf{E} \times \mathbf{H}
\end{equation}
This equation describes the relationship between the electric field and magnetic field in an electromagnetic wave, which is shown through the right hand rule.

"""

# %% [markdown]
# ## <span style="color:orange">(T) Questions: Think!</span>
# 1. This type of question will be a thought experiment or a question whose answer will guide you in the homework. Don't include the answers to these in the report. Think questions typically appear in between the programming assignments.

# %% [markdown]
# ## <span style="color:indigo">(R) Results: Report</span>
#
# 1. Please include some images (screenshots) to show that you've successfully installed Git, Anaconda, and JupyterLab. Follow the tips in the report template to include well-labeled and captioned images.
# See attached zip file
