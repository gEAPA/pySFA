import numpy as np
from sklearn.linear_model import LinearRegression
from math import sqrt, pi, log
from scipy.stats import norm
import scipy.optimize as opt
from .constant import FUN_PROD, FUN_COST, TE_teJ, TE_te, TE_teMod, LOG_hnormal, LOG_tnormal, LOG_exp
from .utils import tools


class SFA:
    """Stochastic frontier analysis (SFA) 
    """

    def __init__(self, y, x, loglik=LOG_hnormal, fun=FUN_PROD, lamda0=1, method=TE_teJ):
        """SFA model

          Args:
              y (float) : output variable. 
              x (float) : input variables.
              fun (String, optional): FUN_PROD (production function) or FUN_COST (cost function). Defaults to FUN_PROD.
          """
        self.y, self.x = tools.assert_valid_basic_data(y, x, fun)
        self.fun, self.lamda0, self.method = fun, lamda0, method
        self.loglik = loglik

    def __mle(self):

        # initial OLS regression
        reg = LinearRegression().fit(X=self.x, y=self.y)
        beta0 = np.concatenate(([reg.intercept_], reg.coef_), axis=0)
        parm = np.concatenate((beta0, [self.lamda0]), axis=0)

        # Maximum Likelihood Estimation
        def __loglik_hnormal(parm):
            ''' Log-likelihood function normal/half-normal distribution'''
            N, K = len(self.x), len(self.x[0]) + 1
            beta0, lamda0 = parm[0:K], parm[K]
            res_ols = self.__resfun(beta0)
            sig2 = np.sum(res_ols**2)/N
            z  = -lamda0*res_ols/sqrt(sig2)
            pz = np.maximum(norm.cdf(z), 1e-323)
            return N/2*log(pi/2) + N/2*log(sig2) - np.sum(np.log(pz)) + N/2.0
        
        def __loglik_exp(parm):
            ''' Log-likelihood function normal/exponential distribution'''
            N, K = len(self.x), len(self.x[0]) + 1
            beta0 = parm[0:K]
            res_ols = self.__resfun(beta0)
            sigu2 = np.var(res_ols)*(1-2/pi)
            sigv2 = np.var(res_ols)
            z = (-res_ols-(sigv2/np.sqrt(sigu2))) / np.sqrt(sigv2)
            pz = np.maximum(norm.cdf(z), 1e-323)
            return N/2*log(sigu2) - N/2*(sigv2/sigu2) - np.sum(np.log(pz)) - np.sum(res_ols)/np.sqrt(sigu2)

        if self.loglik == LOG_hnormal:
            maxlik_ = __loglik_hnormal
        elif self.loglik == LOG_exp:
            maxlik_ = __loglik_exp

        fit = opt.minimize(maxlik_, parm, method='BFGS').x

        # beta, residuals, lambda, sigma^2
        K = len(self.x[0]) + 1
        self.beta = fit[0:K]
        self.residuals = self.__resfun(self.beta)
        self.lamda = fit[K]
        self.sigma2 = np.sum(self.residuals ** 2)/self.residuals.shape[0]

        # sigma_u^2, sigma_v^2
        self.s2u = self.lamda**2 / (1+self.lamda**2) * self.sigma2
        self.s2v = self.sigma2 / (1+self.lamda**2)

        return self.beta, self.residuals, self.lamda, self.sigma2, self.s2u, self.s2v

    def __resfun(self, beta):
        return self.y - beta[0] - np.dot(self.x, beta[1:])

    def __teJ(self):
        '''Efficiencies estimates using the conditional mean approach 
            Jondrow et al. (1982, 235)'''

        if self.fun == FUN_COST:
            self.sign == -1
        else:
            self.sign = 1
        self.ustar = - self.sign * self.residuals * \
            self.lamda**2/(1+self.lamda**2)
        self.sstar = self.lamda/(1+self.lamda**2)*sqrt(self.sigma2)
        return np.exp(-self.ustar - self.sstar *
                      (norm.pdf(self.ustar/self.sstar)/norm.cdf(self.ustar/self.sstar)))

    def __te(self):
        '''Efficiencies estimated by minimizing the mean square error; 
            Eq. (7.21) in Bogetoft and Otto (2011, 219) and Battese and Coelli (1988, 392)'''

        if self.fun == FUN_COST:
            self.sign == -1
        else:
            self.sign = 1
        self.ustar = - self.sign * self.residuals * \
            self.lamda**2/(1+self.lamda**2)
        self.sstar = self.lamda/(1+self.lamda**2)*sqrt(self.sigma2)
        return norm.cdf(self.ustar/self.sstar - self.sstar) / \
            norm.cdf(self.ustar/self.sstar) * \
            np.exp(self.sstar**2/2 - self.ustar)

    def __teMod(self):
        '''Efficiencies estimates using the conditional mode approach;
            Bogetoft and Otto (2011, 219), Jondrow et al. (1982, 235)'''

        if self.fun == FUN_COST:
            self.sign == -1
        else:
            self.sign = 1
        self.ustar = - self.sign * self.residuals * \
            self.lamda**2/(1+self.lamda**2)
        return np.exp(np.minimum(0, -self.ustar))

    def get_technical_efficiency(self):
        """
        Args:
              method (String, optional): TE_teJ, TE_te, or TE_teMod. Defaults to TE_teJ.

        calculate technical efficiency
        """
        self.__mle()
        if self.method == TE_teJ:
            return self.__teJ()
        elif self.method == TE_te:
            return self.__te()
        elif self.method == TE_teMod:
            return self.__teMod()
        else:
            raise ValueError("Undefined decomposition technique.")

    def get_beta(self):
        '''Return the estimated coefficients'''
        return self.__mle()[0]

    def get_residuals(self):
        '''Return the residuals'''
        return self.__mle()[1]

    def get_lambda(self):
        '''Return the lambda'''
        return self.__mle()[2]

    def get_sigma2(self):
        '''Return the sigma2'''
        return self.__mle()[3]

    def get_sigmau2(self):
        '''Return the sigma_u**2'''
        return self.__mle()[4]

    def get_sigmav2(self):
        '''Return the sigma_v**2'''
        return self.__mle()[5]
