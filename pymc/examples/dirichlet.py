import numpy as np
from pymc import *

model = Model()
n = 10
x = [[5,2,2,0,1], [6,1,2,1,0]]
with model:

    k = 5
    a = constant(np.array([2, 3., 4, 2, 2]))

    p, p_m1 = model.TransformedVar(
        'p', Dirichlet.dist(k, a, shape=k),
        simplextransform)

    m = Multinomial('m', n, p, observed=x)

if __name__ == '__main__':

    with model:
        H = model.d2logpc()

        s = find_MAP()

        step = HamiltonianMC(model.vars, H(s))
        trace = sample(1000, step, s)
