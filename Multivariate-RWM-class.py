#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 18:04:31 2018

@author: alex
"""

from __future__ import division
import numpy as np
from scipy import stats
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns


class mcmc:
    
    def __init__(self, n, tune=1, start='generate', nchains=1, 
                 target='normal', dim=2):
        self.n = n
        self.tune = tune
        self.start = start
        self.nchains = nchains
        self.dim = dim
        self.samples = np.zeros(shape=(n,nchains,dim))
        self.rejections = 0
        self.acceptance = 0
        self.esjd = 0
        self.rmean = np.zeros(shape=(n,nchains,dim))
       
    ########################################################################

        proposalcov = np.identity(self.dim) * self.tune

        def proposald(x, centre):
    
            pdf = stats.multivariate_normal.logpdf(x, mean=centre, 
                                                cov=proposalcov)
    
            return pdf 

        def proposalr(centre):
    
            rvs = stats.multivariate_normal.rvs(size=1, mean=centre, 
                                                cov=proposalcov)
    
            return rvs

    ########################################################################

        def mnormd(x):
    
            centre = np.zeros(self.dim)
            sigma = np.identity(self.dim)
            pdf = stats.multivariate_normal.logpdf(x, mean=centre, cov=sigma)
    
            return pdf
        
        def gaussmixd(x):
            
            a = np.zeros(self.dim)
            pdf = ( (1 / (2*((2*np.pi)**(self.dim/2)))) *
                   (np.exp(-0.5*np.dot((x-a),(x-a)))) + 
                   (np.exp(-0.5*np.dot((x+a),(x+a)))))
            
            return pdf         
 
        def logregd(theta):

            logs = np.log(1 + np.exp(-np.dot(X,theta)))
            logsum = np.sum(logs)
            xgramtheta = np.dot(xGram,theta)
            pdf = (- np.dot(YtX, theta) - logsum - 
                   (((3*len(X[0]))/(2*np.pi**2)) 
                   *(np.dot(theta,xgramtheta))))

            return pdf
        
        def bananad(x):
            
            pdf = -((x[0]**2)/200) -(((x[1]-(0.1*((x[0]**2) - 100)))**2)/2)
            
            return pdf 
        
        def flowerd(x):
            
            pdf = -((np.sqrt((x[0]**2)+(x[1]**2)) - 10 -
                    (6*np.cos(6*np.arctan2(x[1],x[0])))) / 2)
            
            return pdf
        
        
        targetdict = {'normal':mnormd, 'gaussian_mixture':gaussmixd,
                      'logistic':logregd, 'banana':bananad,
                      'flower':flowerd}
        
        targetd = targetdict[target]
        
        def negativetarget(x):
            
            pdf = targetd(x)
            
            return -pdf
   
    ########################################################################             

        if self.start == 'generate':
            
            self.start = np.zeros(shape =[self.nchains, self.dim])
            
            nsamp = self.nchains * 50
            init = stats.multivariate_normal.rvs(size = nsamp, 
                    mean=np.zeros(self.dim), cov=(5*np.identity(self.dim)))
            means=[]
            covs=[]
            weights=[]
            
            for i in range(nsamp):
                
                opt = optimize.minimize(negativetarget, init[i])
                mean = opt.x
                means.append(mean)
                hess_inv = np.matrix(opt.hess_inv)
                sigma = hess_inv.I
                covs.append(sigma)
                weights.append(-opt.fun)
            
            optstartm = []
            optstartcov = []
            
            for i in range(self.nchains):
                
                npweights = np.array(weights)
                tot = npweights.sum()
            
                normweights = weights / tot 
                pick = stats.multinomial.rvs(n=1, p=normweights)
                pickn = np.dot(pick,range(len(pick)))
                
                optstartm.append(means[pickn])
                optstartcov.append(covs[pickn])
                
                del weights[pickn]
                del means[pickn]
                del covs[pickn]
                
            newstart = []
            chisq = stats.chi2.rvs(df=2, size=self.nchains)
                
            for i in range(self.nchains):
                
                norm = stats.multivariate_normal.rvs(size = 1, 
                            mean = optstartm[i], cov = optstartcov[i])
                tdist = norm / chisq[i]
                newstart.append(tdist)
                
            self.start = newstart 
            
    ########################################################################
                
        def logaccept(x,y):
    
            logprob = (targetd(y) + proposald(x, centre=y) -
                       targetd(x) - proposald(y, centre=x))
    
            acc = (np.log(stats.uniform.rvs(size=1)) < logprob)
    
            return acc

        for j in range(self.nchains): 
        
            for i in range(self.n):
                
                if i == 0:
                    
                    self.samples[0,j] = self.start[j]
                    self.rmean[0,j] = self.start[j]
                
                else:
                    
                    i2 = i - 1
                    xn = self.samples[i2,j]
                    y1 = proposalr(centre=xn)
                
                    if logaccept(xn,y1):
                        self.samples[i,j] = y1
                    else:
                        self.samples[i,j] = xn
                        self.rejections += 1
                        
                    meani = ((((i-1)/i)*self.rmean[(i-1),j]) + 
                             (self.samples[i,j]/i))
                    self.rmean[i,j] = meani
        
    ########################################################################        
    
    def trace(self, n=0, dim=0, save=0):
        
        plt.plot(self.samples[:,n,dim])
        if save == 1:
            plt.savefig('trace.png')
        
    def traceall(self, dim=0, save=0):
        
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
        
    def plotmeans(self, save=0):
        
        for j in range(self.nchains):
            plt.plot(self.rmean[:,j,0])        
        if save == 1:
            plt.savefig('plotmeans.png')
    
    def scatter(self, n=0, save=0):
        
        plt.scatter(x=self.samples[:,n,0], y=self.samples[:,n,1])
        if save == 1:
            plt.savefig('scatter.png')
    
    def kde(self, n=0):
        
        sns.jointplot(x = self.samples[:,n,0], y=self.samples[:,n,1], 
                      kind = "kde")
    
    def hist(self, n=0):
        
        sns.distplot(self.samples[:,n]) 
    
    
    def diagnose(self):
        
        ratio = (((self.n * self.nchains) - self.rejections) / 
                 (self.n * self.nchains))
        self.acceptance = ratio
        print("Acceptance ratio: ") + str(ratio)
        
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

        print("PSRF value: ") + str(Rhat)
        
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
        
    
    
    
    
    
    
    
    
    
    