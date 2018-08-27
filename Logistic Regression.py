#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 20:12:46 2018

@author: alex
"""


from __future__ import division
import numpy as np
from scipy import stats
import scipy

def generate_data(samples=10, dim=1):
    
    def logistic(theta, x):
        
        y = np.exp(-np.dot(theta,x))
        z = y / (1 + y)
        
        return z
    
    thetatrue = np.ones(dim)
    designX = np.zeros(shape=(samples,dim))
    Y = []
    
    for i in range(samples):
        
        xi = []
        
        for j in range(dim):
            
            xij = (stats.binom.rvs(1,p=0.5)*2)-1
            xi.append(xij)
            
        npxi = np.array(xi)
        norm = np.sqrt(np.sum(npxi**2))
        normxi = npxi / norm
        designX[i] = normxi
        
        yp = logistic(thetatrue, normxi)
        yi = stats.binom.rvs(1, p=yp)
        Y.append(yi)
    
    Y = np.array(Y)
    
    return designX, Y


X,Y = generate_data(3,2)

YtX = np.dot(Y,X)

xGram = np.zeros(shape=[len(X[0]),len(X[0])])

for i in range(len(X[:,0])):
    
    xxt = np.outer(X[i],X[i])
    xGram = xGram + xxt

xGram = xGram / len(X[:,0])

lamda = (3*len(X[0])) / (np.pi**2)

w,v = np.linalg.eig(xGram)

M = (lamda + (0.25*len(X[:,0]))) * np.amax(w)
m = lamda * np.amin(w)

def logregd(theta):

    logs = np.log(1 + np.exp(-np.dot(X,theta)))
    logsum = np.sum(logs)
    xgramtheta = np.dot(xGram,theta)
    pdf = (- np.dot(YtX, theta) - logsum - 
            (((3*len(X[0]))/(2*np.pi**2)) 
            *(np.dot(theta,xgramtheta))))
    return pdf

def negative(theta):
    
    pdf = logregd(theta)
    
    return -pdf
  
    
    
    
    
    
    
    
    
    
    


