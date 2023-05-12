import numpy as np
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

    def __mle(self):

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

        fit = minimize(__loglik, parm, method='BFGS')

        # beta, residuals, lambda, sigma^2
        if self.intercept == False:
            K = len(self.x[0])
        elif self.intercept == True:
            K = len(self.x[0]) + 1
        self.beta = fit.x[0:K]
        self.residuals = self.__resfun(self.beta)
        self.lamda = fit.x[K]
        self.sigma2 = np.sum(self.residuals ** 2)/self.residuals.shape[0]

        # sigma_u^2, sigma_v^2
        self.s2u = self.lamda**2 / (1+self.lamda**2) * self.sigma2
        self.s2v = self.sigma2 / (1+self.lamda**2)

        # calculate standard error, t-value, and p-value
        self.vcov = fit.hess_inv
        self.std_err = np.sqrt(np.diag(self.vcov))
        self.tvalue = fit.x / self.std_err
        self.pvalue = np.around(
            2 * t.sf(abs(self.tvalue), len(self.x) - K), decimals=3)

    def __resfun(self, beta):
        if self.intercept == False:
            return self.y - np.dot(self.x, beta[0:])
        elif self.intercept == True:
            return self.y - beta[0] - np.dot(self.x, beta[1:])

    def __teJ(self):
        '''Efficiencies estimates using the conditional mean approach 
            Jondrow et al. (1982, 235)'''

        self.ustar = - self.sign * self.residuals * \
            self.lamda**2/(1+self.lamda**2)
        self.sstar = self.lamda/(1+self.lamda**2)*sqrt(self.sigma2)
        return np.exp(-self.ustar - self.sstar *
                      (norm.pdf(self.ustar/self.sstar)/norm.cdf(self.ustar/self.sstar)))

    def __te(self):
        '''Efficiencies estimated by minimizing the mean square error; 
            Eq. (7.21) in Bogetoft and Otto (2011, 219) and Battese and Coelli (1988, 392)'''

        self.ustar = - self.sign * self.residuals * \
            self.lamda**2/(1+self.lamda**2)
        self.sstar = self.lamda/(1+self.lamda**2)*sqrt(self.sigma2)
        return norm.cdf(self.ustar/self.sstar - self.sstar) / \
            norm.cdf(self.ustar/self.sstar) * \
            np.exp(self.sstar**2/2 - self.ustar)

    def __teMod(self):
        '''Efficiencies estimates using the conditional mode approach;
            Bogetoft and Otto (2011, 219), Jondrow et al. (1982, 235)'''

        self.ustar = - self.sign * self.residuals * \
            self.lamda**2/(1+self.lamda**2)
        return np.exp(np.minimum(0, -self.ustar))

    def __model_estimation(self):
        '''Model estimation'''

        self.__mle()
        if self.method == TE_teJ:
            self.te = self.__teJ()
        elif self.method == TE_te:
            self.te = self.__te()
        elif self.method == TE_teMod:
            self.te = self.__teMod()
        else:
            raise ValueError("Undefined decomposition technique.")

        return self.beta, self.residuals, self.lamda, self.sigma2, self.s2u, self.s2v, self.std_err, self.tvalue, self.pvalue, self.te

    def get_beta(self):
        '''Return the estimated coefficients'''
        return self.__model_estimation()[0]

    def get_residuals(self):
        '''Return the residuals'''
        return self.__model_estimation()[1]

    def get_lambda(self):
        '''Return the lambda'''
        return self.__model_estimation()[2]

    def get_sigma2(self):
        '''Return the sigma2'''
        return self.__model_estimation()[3]

    def get_sigmau2(self):
        '''Return the sigma_u**2'''
        return self.__model_estimation()[4]

    def get_sigmav2(self):
        '''Return the sigma_v**2'''
        return self.__model_estimation()[5]

    def get_std_err(self):
        '''Return the standard errors'''
        return self.__model_estimation()[6]

    def get_tvalue(self):
        '''Return the standard errors'''
        return self.__model_estimation()[7]

    def get_pvalue(self):
        '''Return the standard errors'''
        return self.__model_estimation()[8]

    def get_technical_efficiency(self):
        '''Return the technical efficiency'''
        return self.__model_estimation()[9]
