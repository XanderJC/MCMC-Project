# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import numpy as np
#import pandas as pd 
from scipy import stats
#import matplotlib.pyplot as plt
#import seaborn as sns 

def normsampler(x):
    samples = []
    while len(samples) < x:
        
        z = stats.cauchy.rvs(size = 1)
        y = stats.uniform.rvs(size = 1) * 2 * stats.cauchy.pdf(z)
        
        if y < stats.norm.pdf(z):
            samples.append(z)
            
    return samples


