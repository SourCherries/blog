import numpy as np
import numpy.ma as ma
from time import perf_counter
import pandas as pd


# -------------------------------------------------------------------
# To do
# 1. Text. Shorten end of intro.
# 2. Shorten prologue. sexier.
# 3. Fix indices of last products equation. maybe set to 2 lines each.
# 4. ***In explanation of Cov, maybe highlight COV table with bold and bold corresponding pair of vars in data

# 0. (MAYBE INSERT) matrix ops are fast while loops are slow

# 1. Missing values
# - basic with single nan and explain carry-over
# - case where listwise deletion is ok
# - realistic data (listwise is highly inefficient)

# - pandas does pairwise 
#     - fast?? test now!!!

# - timing of loops

# 2. Masked arrays

# 3. Percentage agreement (Cohen's kappa)
#     Now that we know we can still use matrix multiplication in a way that accounts for missing values in an efficient pairwise fashion,
#     what can we do? As long as we are doing the same calculation over and over for many pairs of variables, we might be able to speed things up with matrix multiplication.
#     Even with realistic data. In other words, realistic data is no longer a limit to harnessing the power of matrix multiplications.

#     In this example, we use a trick so we can use matrix multiplication to speed up 


# -------------------------------------------------------------------
# Constants

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

C_loops = pd.DataFrame(cov_loop_unbiased(X))
C_panda = pd.DataFrame(X).cov(ddof=1)
assert np.allclose(C_loops, C_panda)


Xm = ma.masked_invalid(X)
C_masks = ma.cov(Xm, rowvar=False, bias=False, allow_masked=True).filled(np.nan)
print(C_masks == C_loops)

# C_masks[0,1]
# C_loops.iat[0,1]

if (C_masks[0,1] < C_loops.iat[0,1]):
    print("\n\nC_masks[0,1] < C_loops.iat[0,1]\n\n")

# C_loops.iat[0,1]*4  # sum is 33
# 33/4 = 8.25  # loops and pandas
# 33/d = 8.40
# d = 33/8.40 = 3.9285714285714284
# print(8.4 * np.arange(6))

Xclean = X[[0, 2, 4, 5],0:2]  # pairwise deletion
M = np.tile(Xclean.mean(0).reshape((1,2)), (4, 1))
Xclean = Xclean - M
assert C_loops.iat[0, 1] == Xclean.prod(1).sum()/(4-1)
assert C_panda.iat[0, 1] == Xclean.prod(1).sum()/(4-1)

# Let's try masked arrays but instead of COV use basic operations
#
# COLUMN WISE DELETION!!!!

# DO NOT USE MASKED COV
# MUST USE INDIVIDUAL OPS WHEN USING MASKED ARRAYS
# SO TRY DOING THIS NEXT -- JUST LIKE WITH AGREEMAT!!!

Mm = np.tile(Xm.mean(0).reshape((1,3)), (6,1))
Xcenter = Xm - Mm  # seems correct
S = ma.dot(Xcenter.transpose(), Xcenter)


valid = np.ones_like(Xm)  # valid responses (n x k)
valid[ma.getmaskarray(R)] = 0
N = np.dot(valid.transpose(), valid)  # NO
N[0,1] == 4
# Xclean.prod(1).sum() / 2.95







# Pandas cov() appears to be doing something different than simply
#   excluding samples on a pairwise basis.
#
#   Perhaps it is doing listwise deletion?


v1 = np.array([-3, M, 0, 0,  3,  3])
v2 = np.array([ 3, 3, 0, M, -3, -3])
p = v1 * v2
p = p[np.isnan(p)==False]
c_pair = p.mean()
assert c_pair == C_loops.iat[0,1]
if (c_pair != C_panda.iat[0,1]):
    print("pandas.DataFrame.cov() does not exclude samples pairwise.")

X_listwise_delete = X[range(0, 5, 2), ]
C_listwise = np.cov(X_listwise_delete, rowvar=False, bias=True)
if (C_listwise[0,1] != C_panda.iat[0,1]):
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

C_loops = np.zeros_like(C)
C_loops = np.zeros((number_variables, number_variables))
tic = perf_counter()
for a in range(number_variables):
    for b in range(a+1, number_variables):
        C_loops[a, b] = np.cov(X[:, [a, b]], rowvar=False)[0,1]
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