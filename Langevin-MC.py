#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 12:03:22 2018

@author: alex
"""
from __future__ import division
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class langevin:
    
    def __init__(self, n, tune, nchains, dim):
        
        self.n = n
        self.tune = tune
        self.nchains = nchains
        self.dim = dim
        self.samples = np.zeros(shape=(n,nchains,dim))
        self.start = []
        self.noise = stats.multivariate_normal.rvs(size = (self.n*self.nchains),
                mean=np.zeros(self.dim), cov=np.identity(self.dim))
        
        XtY = np.dot(X.T,Y)
        
        thetastar = scipy.optimize.minimize(negative,np.zeros(self.dim))
        sigma = (1/M) * np.identity(self.dim)
        for i in range(self.nchains):
            self.start.append(stats.multivariate_normal.rvs(size=1, 
                                        mean=thetastar.x, cov=sigma))
        
        def nabla_f(theta):
            
            logsum = np.zeros(self.dim)
            
            for i in range(len(X[:,0])):
                logs = X[i] / (1 + np.exp(np.dot(X[i],theta)))
                logsum = logsum + logs
                
            grad = (XtY + ((3*len(X[0])/(np.pi**2)) * np.dot(xGram,theta)) -
                    logsum)
            
            return grad
        
        for j in range(self.nchains):
            
            for i in range(self.n):
                
                if i == 0:
                    
                    self.samples[0,j] = self.start[j]
                    
                else:
                    
                    i2 = i - 1
                    xn = self.samples[i2,j]
                    
                    xnplus = xn - (tune*nabla_f(xn)) + (np.sqrt(2*tune) * 
                                               self.noise[((j*self.n) + i)])
                    
                    self.samples[i,j] = xnplus
      
        
    def trace(self, dim=0, save=0):
        
        for j in range(self.nchains):
            plt.plot(self.samples[:,j,dim])
        if save == 1:
            plt.savefig('traceall.png')
        
    def trace2d(self, save=0):
        
        cmap = {0:'b',1:'g',2:'r',3:'c',4:'m',5:'y',6:'k',7:'w'}
        for j in range(self.nchains):
            plt.plot(self.samples[:,j,0],self.samples[:,j,1],
                     'C3', color=cmap[j])
        if save == 1:
            plt.savefig('trace2d.png')        
    
    def kde(self, n=0):
        
        sns.jointplot(x = self.samples[:,n,0], y=self.samples[:,n,1], 
                      kind = "kde")    
    
    def diagnose(self):
                
        means = np.zeros(shape=[self.nchains,self.dim])
        
        for j in range(self.nchains):
            
            chain = np.array(self.samples[:,j,:])
            ave = sum(chain) / self.n 
            means[j,:] = ave
            
        within = np.zeros(shape=[self.dim,self.dim])
        
        for j in range(self.nchains):
            
            for i in range(self.n):
                
                dif = self.samples[i,j,:] - means[j,:]
                sqdif = np.outer(dif, dif.transpose())
                
                within = within + sqdif
            
        wvar = (1/(self.nchains * (self.n - 1))) * within
        
        tmean = sum(means) / self.nchains
        
        tss = np.zeros(shape=[self.dim,self.dim])
        
        for j in range(self.nchains):
            
            dif = means[j] - tmean
            sqdif = np.outer(dif,dif.transpose())
            
            tss = tss + sqdif
        
        bvar = (1/(self.nchains - 1)) * tss
            
        pdmatrix = np.dot(np.linalg.inv(wvar),bvar)
        
        w,v = np.linalg.eig(pdmatrix)
        
        lamda = np.amax(w)
        
        Rhat = ((self.n-1)/self.n)+(((self.nchains+1)/self.nchains)*lamda)

        print("PSFR value: ") + str(Rhat)
        
        chainesjdns=[]
        
        for j in range(self.nchains):
            
            chain = np.array(self.samples[(int(self.n/2)):,j,:])
            length = int(chain.size / self.dim) 

            ex = chain[:(length - 1),:] 
            ex1 = chain[1:,:]

            dif = ex - ex1
            sqdif = np.zeros(length - 1)
        
            for i in range(self.dim):
            
                sqdif = sqdif + (dif[:,i] ** 2)
        
            esjdn = sqdif.sum() / (length - 1)
            chainesjdns.append(esjdn)
            
        npesjd = np.array(chainesjdns)
        esjd = npesjd.sum() / npesjd.size
        self.esjd = esjd
        
        print("ESJD: ") + str(esjd)























