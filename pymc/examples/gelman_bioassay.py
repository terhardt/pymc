from pymc import *
from numpy import ones, array
import theano.tensor as t

def tinvlogit(x):
    # Inverse-logit transform using Theano tensors
    return t.exp(x) / (1 + t.exp(x))

# Samples for each dose level
n = 5 * ones(4, dtype=int)
# Log-dose
dose = array([-.86, -.3, -.05, .73])

with Model() as model:

    # Logit-linear model parameters
    alpha = Normal('alpha', 0, 0.01)
    beta = Normal('beta', 0, 0.01)

    theta = tinvlogit(alpha + beta * dose)

    # Data likelihood
    deaths = Binomial('deaths', n, theta, observed=[0, 1, 3, 5])

    # Calculate LD50
    LD50 = -alpha/beta