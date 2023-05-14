import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from math import sqrt, pi, log
from scipy.stats import norm, t
from scipy.optimize import minimize
from .constant import FUN_PROD, FUN_COST, TE_teJ, TE_te, TE_teMod
from .utils import tools


class SFA:
    """Stochastic frontier analysis (SFA) 
    """

    def __init__(self, y, x, fun=FUN_PROD, intercept=True, lamda0=1, method=TE_teJ):
        """SFA model

          Args:
              y (float) : output variable. 
              x (float) : input variables.
              intercept (bool, optional): whether to include intercept. Defaults to True.
              lamda0 (float, optional): initial value of lambda. Defaults to 1.
              fun (String, optional): FUN_PROD (production function) or FUN_COST (cost function). Defaults to FUN_PROD.
              method (String, optional): TE_teJ, TE_te, or TE_teMod. Defaults to TE_teJ.
          """
        self.fun, self.intercept, self.lamda0, self.method = fun, intercept, lamda0, method
        self.y, self.x = tools.assert_valid_basic_data(y, x, self.fun)

        if self.fun == FUN_COST:
            self.sign = -1
        else:
            self.sign = 1

    def optimize(self):

        # initial OLS regression
        if self.intercept == False:
            reg = LinearRegression(fit_intercept=False).fit(X=self.x, y=self.y)
            parm = np.concatenate((reg.coef_, [self.lamda0]), axis=0)
        elif self.intercept == True:
            reg = LinearRegression().fit(X=self.x, y=self.y)
            parm = np.concatenate(
                ([reg.intercept_], reg.coef_, [self.lamda0]), axis=0)

        # Maximum Likelihood Estimation
        def __loglik(parm):
            ''' Log-likelihood function'''
            N = len(self.x)
            if self.intercept == False:
                K = len(self.x[0])
            elif self.intercept == True:
                K = len(self.x[0]) + 1
            beta0, lamda0 = parm[0:K], parm[K]
            res = self.__resfun(beta0)
            sig2 = np.sum(res**2)/N
            z = -lamda0*res/sqrt(sig2)
            pz = np.maximum(norm.cdf(z), 1e-323)
            return N/2*log(pi/2) + N/2*log(sig2) - np.sum(np.log(pz)) + N/2.0

        return minimize(__loglik, parm, method='BFGS')

    def __resfun(self, beta):
        ''' Residual function'''
        if self.intercept == False:
            return self.y - np.dot(self.x, beta[0:])
        elif self.intercept == True:
            return self.y - beta[0] - np.dot(self.x, beta[1:])

    def get_beta(self):
        '''Return the estimated coefficients'''
        return self.optimize().x[0:-1]

    def get_lambda(self):
        '''Return the estimated lambda'''
        return self.optimize().x[-1]

    def get_residuals(self):
        '''Return the estimated residuals'''
        return self.__resfun(self.optimize().x[0:-1])

    def get_sigma2(self):
        '''Return the estimated sigma2'''
        return np.sum(self.get_residuals()**2)/len(self.x)

    def get_sigmau2(self):
        '''Return the estimated sigmau2'''
        return self.get_lambda()**2 / (1 + self.get_lambda()**2) * self.get_sigma2()

    def get_sigmav2(self):
        '''Return the estimated sigmav2'''
        return self.get_sigma2()/(1 + self.get_lambda()**2)

    def get_std_err(self):
        '''Return the standard errors'''
        return np.sqrt(np.diag(self.optimize().hess_inv))

    def get_tvalue(self):
        '''Return the t-values'''
        return self.optimize().x/self.get_std_err()

    def get_pvalue(self):
        '''Return the p-values'''
        if self.intercept == False:
            K = len(self.x[0])
        elif self.intercept == True:
            K = len(self.x[0]) + 1
        return np.around(2*t.sf(np.abs(self.get_tvalue()), len(self.x) - K), decimals=3)

    def __teJ(self):
        '''Efficiencies estimates using the conditional mean approach 
            Jondrow et al. (1982, 235)'''

        self.ustar = - self.sign * self.get_residuals() * \
            self.get_lambda()**2/(1+self.get_lambda()**2)
        self.sstar = self.get_lambda()/(1+self.get_lambda()**2)*sqrt(self.get_sigma2())
        return np.exp(-self.ustar - self.sstar *
                      (norm.pdf(self.ustar/self.sstar)/norm.cdf(self.ustar/self.sstar)))

    def __te(self):
        '''Efficiencies estimated by minimizing the mean square error; 
            Eq. (7.21) in Bogetoft and Otto (2011, 219) and Battese and Coelli (1988, 392)'''

        self.ustar = - self.sign * self.get_residuals() * \
            self.get_lambda()**2/(1+self.get_lambda()**2)
        self.sstar = self.get_lambda()/(1+self.get_lambda()**2)*sqrt(self.get_sigma2())
        return norm.cdf(self.ustar/self.sstar - self.sstar) / \
            norm.cdf(self.ustar/self.sstar) * \
            np.exp(self.sstar**2/2 - self.ustar)

    def __teMod(self):
        '''Efficiencies estimates using the conditional mode approach;
            Bogetoft and Otto (2011, 219), Jondrow et al. (1982, 235)'''

        self.ustar = - self.sign * self.get_residuals() * \
            self.get_lambda()**2/(1+self.get_lambda()**2)
        return np.exp(np.minimum(0, -self.ustar))

    def get_technical_efficiency(self):
        '''Return the technical efficiency estimates'''

        if self.method == TE_teJ:
            return self.__teJ()
        elif self.method == TE_te:
            return self.__te()
        elif self.method == TE_teMod:
            return self.__teMod()
        else:
            raise ValueError("Undefined decomposition technique.")

    def summary(self):
        '''Print the summary of the estimation results'''

        if self.intercept == False:
            self.names = ['x'+str(i+1)
                          for i in range(len(self.x[0]))] + ['lambda']
        elif self.intercept == True:
            self.names = ['(Intercept)'] + ['x'+str(i+1)
                                            for i in range(len(self.x[0]))] + ['lambda']

        output = np.vstack(
            (np.round(self.optimize().x, decimals=5), np.round(self.get_std_err(), decimals=6), np.round(self.get_tvalue(), decimals=3), self.get_pvalue()))
        index = ['Parameters', 'Std.err', 't-value', 'Pr(>|t|)']
        re = pd.DataFrame(output, index=index, columns=self.names).T
        print(re)
        print('sigma2: ', self.get_sigma2().round(5))
        print('sigmav2: ', self.get_sigmav2().round(5),
              '; sigmau2: ', self.get_sigmau2().round(5))
        print('log likelihood: ', round(self.optimize().fun, 5))
