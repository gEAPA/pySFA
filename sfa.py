# Stochastic Frontier Analysis with maximum likelihood estimation (SFA-MLE)
# Reference: Bogetoft, P. and L. Otto (2011). Benchmarking with DEA, SFA, and R, Springer.
# Author: Sheng Dai, Turku School of Economics, Finland
# Date: 4 May 2023

import numpy as np
from sklearn.linear_model import LinearRegression
from math import sqrt, pi, log 
from scipy.stats import norm
import scipy.optimize as opt

def resfun(x,y,beta):
      return y - beta[0] - np.dot(x , beta[1:])

def sfa(x, y, lamda0=1):
      
      # initial OLS regression
      reg = LinearRegression().fit(X=x, y=y)
      beta0 = np.concatenate(([reg.intercept_], reg.coef_), axis=0)
      parm = np.concatenate((beta0, [lamda0]), axis=0)

      # Maximum Likelihood Estimation  
      def loglik(parm):
            '''
            Log-likelihood function for the SFA model
            '''

            N, K = x.shape[0], x.shape[1] + 1
            beta0, lamda0 = parm[0:K], parm[K]
            e = resfun(x,y,beta0)  
            s = np.sum(e**2)/N
            z = -lamda0*e/sqrt(s)
            pz = np.maximum(norm.cdf(z), 1e-323) 
      
            return N/2*log(pi/2) + N/2*log(s) - np.sum(np.log(pz)) + N/2.0 
      
      fit = opt.minimize(loglik, parm, method='BFGS').x

      # beta, residuals, lambda, sigma^2
      K = x.shape[1] + 1
      beta = fit[0:K]
      e = resfun(x,y,beta)  
      lamda = fit[K]
      sigma2 = np.sum(e ** 2)/e.shape[0]

      # sigma_u^2, sigma_v^2
      s2u = lamda**2 / (1+lamda**2) * sigma2
      s2v = sigma2 / (1+lamda**2)

      return beta, e, lamda, sigma2, s2u, s2v

def teJ(residuals, lamda, sigma2):
      '''
      Efficiencies estimates using the conditional mean approach 
      Jondrow et al. (1982, 235)
      '''

      sign = 1
      ustar = - sign * residuals*lamda**2/(1+lamda**2)
      sstar = lamda/(1+lamda**2)*sqrt(sigma2)

      return np.exp(-ustar -sstar*( norm.pdf(ustar/sstar)/norm.cdf(ustar/sstar) ) )

def te(residuals, lamda, sigma2):
      '''
      Efficiencies estimated by minimizing the mean square error; 
      Eq. (7.21) in Bogetoft and Otto (2011, 219) and Battese and Coelli (1988, 392)
      '''

      sign = 1
      ustar = - sign * residuals*lamda**2/(1+lamda**2)
      sstar = lamda/(1+lamda**2)*sqrt(sigma2)

      return norm.cdf(ustar/sstar - sstar)/norm.cdf(ustar/sstar) * np.exp(sstar**2/2 -ustar)

def teMod(residuals, lamda):
      '''
      Efficiencies estimates using the conditional mode approach; 
      Bogetoft and Otto (2011, 219), Jondrow et al. (1982, 235)
      '''

      sign = 1
      ustar = - sign * residuals*lamda**2/(1+lamda**2)

      return np.exp(np.minimum(0, -ustar))
