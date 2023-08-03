import numpy as np
import numpy.ma as ma
from time import perf_counter
import pandas as pd


# Relation between matrix multiplication and covariance
X = np.array([[-1,  1, -1],
              [-1,  1,  1],
              [ 1, -1, -1],
              [ 1, -1,  1]])

np.cov(X, rowvar=False, bias=True)

# Efficient for large data
number_variables = 100
number_samples = 1000

tic = perf_counter()
X = np.random.randn(number_samples, number_variables)
C = np.cov(X, rowvar=False)
toc = perf_counter()
print(f"Numpy COV took {toc-tic:0.4f} seconds.")

C_loop = np.zeros_like(C)
C_loop = np.zeros((number_variables, number_variables))
tic = perf_counter()
for a in range(number_variables):
    for b in range(a+1, number_variables):
        C_loop[a, b] = np.cov(X[:, [a, b]], rowvar=False)[0,1]
toc = perf_counter()
print(f"Looping through COV took {toc-tic:0.4f} seconds.")


# Missing value problem
M = np.nan
X = np.array([[-1, +1, -1],
              [-1, +1, +1],
              [+1, -1, -1],
              [+1, -1, +1],
              [ M,  M,  M]])
np.cov(X, rowvar=False, bias=True)
np.cov(X[:-1, :], rowvar=False, bias=True)

X = np.array([[-1, +1, -1],
              [-1, +1, +1],
              [+1, -1, -1],
              [+1, -1, +1]])


X = np.array([[-3, 3, 0],
              [ 0, 0, 0],
              [ 3,-3, 0]])

X = np.array([[-3, 3, 0],
              [ 0, 0, 0],
              [ 3,-3, 0],
              [ M, M, M]])
np.cov(X, rowvar=False, bias=True)
np.cov(X[:-1, :], rowvar=False, bias=True)

X = np.array([[-3, 3, 0],
              [ M, 3, 0],
              [ 0, 0, 0],
              [ 0, M, 0],
              [ 3,-3, 0],
              [ 3,-3, M]])
np.cov(X, rowvar=False, bias=True)
np.cov(X[range(0, 5, 2), ], rowvar=False, bias=True)


# End
# -------------------------------------------------------------------