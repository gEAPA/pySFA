# A script to test the SFA model
# Example from the book by Coelli et al
# Reference: Bogetoft, P. and L. Otto (2011). Benchmarking with DEA, SFA, and R, Springer.

import numpy as np
import pandas as pd
import sfa

#  read data
df = np.array(pd.read_csv('41Firm.csv', index_col=0))
y = np.log(df[:, 0])
x = np.log(df[:, 1:3])

# Estimate SFA model
beta, residuals, lamda, sigma2, sigmau2, sigmav2 = sfa.sfa(x,y)

print('beta: ', beta)
print('lamda: ', lamda)
print('sigma2: ', sigma2)
print('sigmau2: ', sigmau2)
print('sigmav2: ', sigmav2)
print('residuals: ', residuals)

# Estimate efficiency for each unit
te = sfa.te(residuals, lamda, sigma2)
print(te)

te = sfa.teJ(residuals, lamda, sigma2)
print(te)

te = sfa.teMod(residuals, lamda)
print(te)