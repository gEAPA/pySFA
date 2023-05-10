# function
FUN_PROD = "prod"
"""
FUN_PROD: Production function.
"""

FUN_COST = "cost"
"""
FUN_COST: Cost function.
"""

FUN_Categories = {
    FUN_PROD: "Production function",
    FUN_COST: "Cost function"
}

# Log Likelihood functions
LOG_hnormal = "hnormal"
"""
LOG_hnormal: Log-Lik normal/half-normal distribution.
"""

LOG_tnormal = "tnormal"
"""
LOG_hnormal: Log-Lik normal/truncated-normal distribution.
"""

LOG_exp = "exp"
"""
LOG_hnormal: Log-Lik normal/exponential distribution.
"""


# Technical inefficiency
TE_teJ = "teJ"
"""
TE_teJ: Using conditional mean approach.
"""

TE_te = "te"
"""
TE_te: Minimizing the mean square error.
"""

TE_teMod = "teMod"
"""
TE_teMod: Using conditional mode approach.
"""

RED_Categories = {
    TE_teJ: "Conditional mean",
    TE_te: "Mean square error",
    TE_teMod: "Conditional mode"
}
