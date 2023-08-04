import numpy as np
import numpy.ma as ma
import pandas as pd
import math
from time import perf_counter


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
# Helper functions

def cov_loop_unbiased(X):
    """Covariance matrix using only base Python and for-loops.
       Normalized by (N-1) where N is observations.
       Pairwise deletion.

       Variables centered on a pairwise basis.
       
    Parameters
    ----------
    X : array_like
        An n by k array of n observations of k variables.
       
    Returns
    -------
    C : ndarray
        A k x k covariance matrix. 
        Covariance between X[:,i] and X[:,j] is C[i, j] or C[j, i].
    """
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


def cov_loop_unbiased_pre_center(X):
    """Covariance matrix using only base Python and for-loops.
       Normalized by (N-1) where N is observations.
       Pairwise deletion.

       Variables centered on a variable-by-variable basis.
       
    Parameters
    ----------
    X : array_like
        An n by k array of n observations of k variables.
       
    Returns
    -------
    C : ndarray
        A k x k covariance matrix. 
        Covariance between X[:,i] and X[:,j] is C[i, j] or C[j, i].
    """
    number_samples, number_variables = X.shape
    # mean-centered variables
    col_means = []
    for col in X.transpose():
        values = [i for i in col if not math.isnan(i)]
        col_means.append(sum(values) / len(values))
    M = np.tile(np.array(col_means).reshape((1, number_variables)), (number_samples, 1))
    CX = X - M
    C = np.zeros((number_variables, number_variables))
    for v1 in range(number_variables):
        for v2 in range(number_variables):
            x = CX[:,v1]
            y = CX[:,v2]

            # remove observations where missing for either x or y
            products = [a*b for a, b in zip(x,y)]
            nx = [xi for xi, pi in zip(x, products) if not math.isnan(pi)]
            ny = [yi for yi, pi in zip(y, products) if not math.isnan(pi)]

            # main result
            products = [a*b for a, b in zip(nx, ny)]
            C[v1, v2] = sum(products) / (len(products)-1)
    return(C)

def cov_pandas_unbiased(X):
    return(pd.DataFrame(X).cov(ddof=1))

def cov_masked_unbiased(X):
    Xm = ma.masked_invalid(X)
    C_masks = ma.cov(Xm, rowvar=False, bias=False, allow_masked=True)
    return(C_masks.filled(np.nan))


# --------------------------------------------------------------
# What are pandas and numpy.MaskedArray doing when they
#   calculate covariance?
X = np.array([[-3, 3, 0],
              [ M, 3, 0],
              [ 0, 0, 0],
              [ 0, M, 0],
              [ 3,-3, 0],
              [ 3,-3, M]])

C_panda = cov_pandas_unbiased(X)
C_loops = cov_loop_unbiased(X)
if np.allclose(C_panda, C_loops):
    print("pandas.DataFrame.cov() does pairwise centering and deletion.")

C_masks = cov_masked_unbiased(X)
C_loops_pre = cov_loop_unbiased_pre_center(X)
if np.allclose(C_masks, C_loops_pre):
    print("numpy.ma.cov() does variable-by-var#iable centering then pairwise deletion.")


# -------------------------------------------------------------------
# Now finally make function where I use masked arrays but
#   coded so using basic matrix operations rather than COV().

# this is going to be hard because I have a potentially different
#   centering for each pair!!!

Xm = ma.masked_invalid(X)
C_masks = ma.cov(Xm, rowvar=False, bias=False, allow_masked=True)
# bad because mean-centered variable-by-variable

# possible to mean-center pairwise? yes, but I will now have a 3D matrix ...
#
#   (number_variables x number_variables) x number_variables
#   (        covariance matrix          ) x number_variables


# BEFORE I SPEND TIME ON THIS CONFIRM THAT PANDAS FASTER THAN  cov_loop_unbiased()


# -------------------------------------------------------------------
# Timing
number_variables = 100
number_samples = 1000

X = np.random.randn(number_samples, number_variables)

tic = perf_counter()
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