import numpy as np
from scipy import linalg

def sum_numbers(x, y):
    """
    TODO: IMPLEMENT ME

    Sum two numbers.

    Args:
        x (int, float): first number in sum
        y (int, float): second number in sum

    Returns:
        Sum of x and y.
    """
    # replace the following line with an actual implementation that returns something
    return x + y

def multiply_numbers(x, y):
    """
    TODO: IMPLEMENT ME

    Multiply two numbers.

    Args:
        x (int, float): first number in product
        y (int, float): second number in product

    Returns:
        Product of x and y.
    """
    # replace the following line with an actual implementation that returns something
    return x*y

def create_add_matrix(x):
    """
    TODO: IMPLEMENT ME

    Step 1. Create a 3x3 numpy array whose elements are all ones.
    Step 2. Sum the array and the input array x.
    Step 3. Return the result

    Args:
        x (np.ndarray): a 2D numpy array

    Returns:
        output (np.ndarray): the operation result
    """
    # replace the following line with an actual implementation that returns something
    three_by_three = np.ones((3, 3))
    
    try:
        sum = x + three_by_three
    except ValueError:
        raise ValueError('Matrix must have the same shape or broacastable shape for addition')
    
    return sum
    

def indexing_aggregation(x, n):
    """
    TODO: IMPLEMENT ME

    Return the mean value of the first n elements of the input array x.

    Args:
        x (np.ndarray): a 1D numpy array
        n (int): number of elements to include in the mean calculation

    Returns:
        output (float): the operation result
    """
    # replace the following line with an actual implementation that returns something
    return np.mean(x[:n]).item()

def matrix_inverse(A):
    """
    TODO: IMPLEMENT ME

    Return the inverse of Matrix A.
    Checks for dimension mismatch.

    Args:
        A (np.ndarray): a 2D numpy array

    Returns:
        output (np.ndarray): the operation result
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError('Matrix must be square for inversion')
    
    
    return np.linalg.inv(A)

def element_wise_multiplication(A, B):
    """
    TODO: IMPLEMENT ME

    Perform element-wise multiplication of two matrices A and B.
    Checks for dimension mismatch.

    Args:
        A (np.ndarray): a 2D numpy array
        B (np.ndarray): a 2D numpy array

    Returns:
        output (np.ndarray): the operation result
    """
    # replace the following line with an actual implementation that returns something
    if A.shape != B.shape:
        raise ValueError('Matrix must have the same shape for element-wise multiplication')
    
    return A * B
 

def matrix_transpose(A):
    """
    TODO: IMPLEMENT ME

    Return the transpose of Matrix A.

    Args:
        A (np.ndarray): a 2D numpy array

    Returns:
        output (np.ndarray): the operation result
    """
    # replace the following line with an actual implementation that returns something
    
    return A.T