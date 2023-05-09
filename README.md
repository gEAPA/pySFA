# Stochastic Frontier Analysis (SFA)

## Installation

The [`pySFA`](https://pypi.org/project/pysfa/) package is now avaiable on PyPI and the latest development version can be installed from the Github repository [`pySFA`](https://github.com/gEAPA/pySFA). Please feel free to download and test it. We welcome any bug reports and feedback.

#### PyPI [![PyPI version](https://img.shields.io/pypi/v/pysfa.svg?maxAge=3600)](https://pypi.org/project/pysfa/)[![PyPI downloads](https://img.shields.io/pypi/dm/pysfa.svg?maxAge=21600)](https://pypistats.org/packages/pysfa)

    pip install pysfa

#### GitHub

    pip install -U git+https://github.com/gEAPA/pySFA


## Authors

- [Sheng Dai](https://daisheng.io), PhD, Turku School of Economics, University of Turku, Finland.
- [Zhiqiang Liao](https://liaozhiqiang.com), Doctoral Researcher, Aalto University School of Business, Finland.


## Demo: Estimating a production function by `pySFA`

```python
import numpy as np
import pandas as pd
from pysfa import SFA
from pysfa.dataset import load_Tim_Coelli_frontier


# import the data from Tim Coelli Frontier 4.1
df = load_Tim_Coelli_frontier(x_select=['labour', 'capital'],
                              y_select=['output'])
y = np.log(df.y)
x = np.log(df.x)

# Estimate SFA model
res = SFA.SFA(y, x, fun=SFA.FUN_PROD, lamda0=1, method=SFA.TE_teJ)

# print estimates
print(res.get_beta())
print(res.get_lambda())
print(res.get_sigma2())
print(res.get_sigmau2())
print(res.get_sigmav2())

# print TE
print(res.get_technical_efficiency())
```

