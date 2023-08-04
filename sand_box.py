import numpy as np
import numpy.ma as ma
from time import perf_counter
import pandas as pd


# Missing value
M = np.nan


# -------------------------------------------------------------------
# Helper function
import math

X = np.array([[-3, 3, 0],
              [ M, 3, 0],
              [ 0, 0, 0],
              [ 0, M, 0],
              [ 3,-3, 0],
              [ 3,-3, M]])
# np.cov(X[range(0, 5, 2), ], rowvar=False, bias=True)
# x = X[:,0]
# y = X[:,1]
# p = [a*b for a, b in zip(x,y)]
# c = sum([v for v in p if not math.isnan(v)])

def cov_loop_unbiased_upper(X):
    number_variables = X.shape[1]
    C = np.zeros((number_variables, number_variables))
    for v1 in range(number_variables):
        for v2 in range(v1+1, number_variables):
            x = X[:,v1]
            y = X[:,v2]

            # remove observations where missing for either x or y
            products = [a*b for a, b in zip(x,y)]
            nx = [xi for xi, pi in zip(x, products) if not math.isnan(pi)]
            ny = [yi for yi, pi in zip(y, products) if not math.isnan(pi)]

            # mean-centered variables
            cx = [x-sum(nx)/len(nx) for x in nx]
            cy = [y-sum(ny)/len(ny) for y in ny]

            # main result
            products = [a*b for a, b in zip(cx,cy)]
            C[v1, v2] = sum(products) / (len(products)-1)
    return(C)

def cov_loop_unbiased(X):
    number_variables = X.shape[1]
    C = np.zeros((number_variables, number_variables))
    for v1 in range(number_variables):
        for v2 in range(number_variables):
            x = X[:,v1]
            y = X[:,v2]

            # remove observations where missing for either x or y
            products = [a*b for a, b in zip(x,y)]
            nx = [xi for xi, pi in zip(x, products) if not math.isnan(pi)]
            ny = [yi for yi, pi in zip(y, products) if not math.isnan(pi)]

            # mean-centered variables
            cx = [x-sum(nx)/len(nx) for x in nx]
            cy = [y-sum(ny)/len(ny) for y in ny]

            # main result
            products = [a*b for a, b in zip(cx,cy)]
            C[v1, v2] = sum(products) / (len(products)-1)
    return(C)

C_me = pd.DataFrame(cov_loop_unbiased(X))
C_pd = pd.DataFrame(X).cov(ddof=1)

# Pandas cov() appears to be doing something different than simply
#   excluding samples on a pairwise basis.
#
#   Perhaps it is doing listwise deletion?


v1 = np.array([-3, M, 0, 0,  3,  3])
v2 = np.array([ 3, 3, 0, M, -3, -3])
p = v1 * v2
p = p[np.isnan(p)==False]
c_pair = p.mean()
assert c_pair == C_me.iat[0,1]
if (c_pair != C_pd.iat[0,1]):
    print("pandas.DataFrame.cov() does not exclude samples pairwise.")

X_listwise_delete = X[range(0, 5, 2), ]
C_listwise = np.cov(X_listwise_delete, rowvar=False, bias=True)
if (C_listwise[0,1] != C_pd.iat[0,1]):
    print("pandas.DataFrame.cov() does not use listwise deletion.")

pd.DataFrame(X).cov(ddof=1, min_periods=1, numeric_only=False).iat[0, 1]
# -8.25

pd.DataFrame(X).cov(ddof=0, min_periods=1, numeric_only=False).iat[0, 1]
# -8.25

np.cov(X_listwise_delete, rowvar=False)[0, 1]
# -9

np.cov(X_listwise_delete, rowvar=False, bias=True)[0, 1]
# -6

X_01 = X[:, 0:2]
keep = np.isnan(X_01.prod(1)) == False
X_pairwise_delete = X_01[keep, :]

np.cov(X_pairwise_delete, rowvar=False)[0, 1]
# -8.25

np.cov(X_pairwise_delete, rowvar=False, bias=True)[0, 1]
# -6.1875

print("pandas.DataFrame.cov() same as pairwise numpy.cov(bias=False) or normalized by (N-1).")

print("pandas.DataFrame.cov(ddof=0) does not seem to work. Forces ddof=1 and normalized by (N-1).")

# Let's further explore
#   np.cov(X_pairwise_delete, rowvar=False)[0, 1]


# -------------------------------------------------------------------
# Four-sample examples

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


# Missing value problems
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


# -------------------------------------------------------------------
# Three-sample examples

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