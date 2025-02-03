#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 10:43:54 2022

@author: marcelo
"""
import numpy as np
from numba import jit, prange
from time import time
import numpy.typing as npt

@jit(nopython=True, parallel=False)
def fast_argmax(array_2d):
    matches = np.nonzero((array_2d == np.max(array_2d)))
    row, col= matches
    return row[0], col[0]

#@jit(nopython=True, parallel=False)
def my_unravel_index(index, shape: npt):
    sizes = np.zeros(len(shape), dtype=np.int64)
    result = np.zeros((1,len(shape)), dtype=np.int64)
    sizes[-1] = 1
    for i in range(len(shape) - 2, -1, -1):
        sizes[i] = sizes[i + 1] * shape[i + 1]
    remainder = index
    for i in range(len(shape)):
        result[0][i] = remainder // sizes[i]
        remainder %= sizes[i]
    result2=(result[0][0],result[0][1])
    return result2


if __name__== '__main__':
    array=np.random.rand(10,10)
    start=time()
    result =my_unravel_index(np.argmax(array, axis=None), array.shape)
    end = time()    
    print(f'It took {(end - start):,.4f} seconds!')
    
    start=time()
    result2= np.unravel_index(np.argmax(array, axis=None), array.shape)
    end = time()    
    print(f'It took {(end - start):,.4f} seconds!')
    