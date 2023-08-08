import numpy as np
import numpy.ma as ma
import pandas as pd
import math
from time import perf_counter
import timeit


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

def cov_loop(X, ddof=1):
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
            C[v1, v2] = sum(products) / (len(products)-ddof)
    return(C)


def cov_loop_pre_center(X, ddof=1):
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
            C[v1, v2] = sum(products) / (len(products)-ddof)
    return(C)

def cov_pandas_unbiased(X):
    """ ddof has no effect """
    return(pd.DataFrame(X).cov(ddof=1))

def cov_masked(X, bias=False):
    Xm = ma.masked_invalid(X)
    C_masks = ma.cov(Xm, rowvar=False, bias=bias, allow_masked=True)
    return(C_masks.filled(np.nan))

# def cov_masked_unbiased_pre_center(X):
X = np.array([[1, 2, 3],
              [M, 2, 0],
              [2, 2, 2],
              [0, M, 4],
              [3, 2, 1],
              [4, 2, M]])



ddof = 0
maskedX = ma.masked_invalid(X)
Ones = ma.ones_like(maskedX)
P = ma.dot(maskedX.transpose(), maskedX)        # sums of products
SumsA = ma.dot(maskedX.transpose(), Ones)       # sums of 1st variable in pair
SumsB = ma.dot(Ones.transpose(), maskedX)       # sums of 2nd variable in pair
Counts = ma.dot(Ones.transpose(), Ones)         # counts valid pairs
MSS = ma.multiply(P, 1/(Counts-ddof))           # mean sum of products
MeansA = ma.multiply(SumsA, 1/(Counts-ddof))    # means of 1st variable
MeansB = ma.multiply(SumsB, 1/(Counts-ddof))    # means of 2nd variable
C_ = MSS - ma.multiply(MeansA, MeansB)           # covariance matrix

C_loops_pre_ = cov_loop_pre_center(X, ddof=0)
C_loops_ = cov_loop(X, ddof=0)

if np.allclose(C_, C_loops_pre_)==False:
    print("\nImplementation of rapid computation does NOT do variable-by-variable centering then pairwise deletion.\n\n")

if np.allclose(C_, C_loops_)==False:
    print("\nImplementation of rapid computation does NOT do pairwise centering and deletion.")
    print("I was expecting it to do exactly this 😔.\n\n")
else:
    print("\nImplementation of rapid computation does do pairwise centering and deletion!")
    print("Exactly what I was expecting! 🚀")
    print("Only thing is I cannot get it do work with unbiased (ddof=1) ... \n\n")

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
C_loops = cov_loop(X, ddof=1)
if np.allclose(C_panda, C_loops):
    print("pandas.DataFrame.cov() does pairwise centering and deletion.")

C_masks = cov_masked(X, bias=False)
C_loops_pre = cov_loop_pre_center(X, ddof=1)
if np.allclose(C_masks, C_loops_pre):
    print("numpy.ma.cov() does variable-by-variable centering then pairwise deletion.")


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


# -------------------------------------------------------------------
# Percentage agreement
# -------------------------------------------------------------------

# example data
X = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
              [0, 0, 0, 0, 1, 1, 1, 1],
              [0, 1, 0, 1, 0, 1, 0, 1],
              [1, 1, 1, 1, 0, 0, 0, 0]]).transpose()

col_labels = ['Item ' + str(i) for i in range(1,5)]
row_labels = ['Person ' + str(i) for i in range(1,9)]
print(pd.DataFrame(X, columns=col_labels, index=row_labels))

# for-loop
def pa_loop(X):
    number_samples, number_variables = X.shape
    percent_agreement = np.zeros((number_variables, number_variables))
    for item_a in range(number_variables):
        for item_b in range(item_a+1, number_variables):
            percent_agreement[item_a, item_b] = (X[:, item_a]==X[:, item_b]).sum()
    percent_agreement /= number_samples
    return(percent_agreement)

# for-loop (large sample)
number_samples_large, number_variables_large = 10000, 200
X = np.random.choice([0, 1], size=(number_samples_large, number_variables_large), replace=True)
tic = perf_counter()
percent_agreement = pa_loop(X)
toc = perf_counter()
seconds_loop = toc-tic
print(f"Computing all pairwise percentage agreements took {seconds_loop:0.4f} seconds.")

# mat mul (explained)
X = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
              [0, 0, 0, 0, 1, 1, 1, 1],
              [0, 1, 0, 1, 0, 1, 0, 1],
              [1, 1, 1, 1, 0, 0, 0, 0]]).transpose()

number_samples, number_variables = X.shape
yesYes = np.dot(X.transpose(), X)   # counts of yes-yes (k x k)
F = np.abs(X-1)                     # [0,1] -> [1,0]
noNo = np.dot(F.transpose(), F)     # counts of no-no   (k x k)
S = yesYes + noNo                   # counts of agreements (k x k)
A = S / number_samples              # percentage agreements (k x k)   
print(pd.DataFrame(A, columns=col_labels, index=col_labels))

# mat mul (timed)
def pa_mat(X):
    number_samples, number_variables = X.shape
    yesYes = np.dot(X.transpose(), X)   # counts of yes-yes (k x k)
    F = np.abs(X-1)                     # [0,1] -> [1,0]
    noNo = np.dot(F.transpose(), F)     # counts of no-no   (k x k)
    S = yesYes + noNo                   # counts of agreements (k x k)
    A = S / number_samples              # percentage agreements (k x k)   
    return(A)

def pa_mat_(X):
    """Much slower. Making copy takes more time than transpose."""
    number_samples, number_variables = X.shape
    XT = np.copy(X.transpose())
    yesYes = np.dot(XT, X)              # counts of yes-yes (k x k)
    F = np.abs(X-1)                     # [0,1] -> [1,0]
    FT = np.abs(XT-1)                   # [0,1] -> [1,0]
    noNo = np.dot(FT, F)                # counts of no-no   (k x k)
    S = yesYes + noNo                   # counts of agreements (k x k)
    A = S / number_samples              # percentage agreements (k x k)   
    return(A)

def pa_mat__(X):
    """In-place Numpy operation. Indeed better than pa_mat()"""
    number_samples, number_variables = X.shape
    yesYes = np.dot(X.transpose(), X)   # counts of yes-yes (k x k)
    np.abs(X-1, out=X)                  # [0,1] -> [1,0]
    noNo = np.dot(X.transpose(), X)     # counts of no-no   (k x k)
    S = yesYes + noNo                   # counts of agreements (k x k)
    A = S / number_samples              # percentage agreements (k x k)   
    return(A)

# mat mul (timed)
def pa_mat_rev(X):
    number_samples, number_variables = X.shape
    yesYes = np.dot(X, X.transpose())   # counts of yes-yes (k x k)
    F = np.abs(X-1)                     # [0,1] -> [1,0]
    noNo = np.dot(F, F.transpose())     # counts of no-no   (k x k)
    S = yesYes + noNo                   # counts of agreements (k x k)
    A = S / number_samples              # percentage agreements (k x k)   
    return(A)

X = np.random.choice([0, 1], size=(number_samples_large, number_variables_large), replace=True)
XT = X.transpose()
tic = perf_counter()
percent_agreement = pa_mat(X)
toc = perf_counter()
seconds_mat = toc-tic
print(f"Computing all pairwise percentage agreements took {seconds_mat:0.4f} seconds.")

print(f"Loops takes {seconds_loop / seconds_mat:0.4f} times longer.")

# number_samples_large = 1000
# number_variables_large = [10, 50, 100, 500, 1000]
# seconds = np.zeros((len(number_variables_large), 4))
# for i, nvl in enumerate(number_variables_large):
#     print(str(i) + "\n")
#     # X = np.random.choice([0, 1], size=(number_samples_large, nvl), replace=True)
#     X = np.random.choice([0., 1.], size=(number_samples_large, nvl), replace=True)
#     tic = perf_counter(); percent_agreement = pa_loop(X); toc = perf_counter()
#     seconds_loop = toc-tic

#     tic = perf_counter(); percent_agreement = pa_mat(X); toc = perf_counter()
#     seconds_mat = toc-tic

#     tic = perf_counter(); percent_agreement = pa_mat__(X); toc = perf_counter()
#     seconds_mat_ = toc-tic

#     tic = perf_counter(); percent_agreement = pa_mat_rev(XT); toc = perf_counter()
#     seconds_mat_rev = toc - tic

#     seconds[i, 0] = seconds_loop
#     seconds[i, 1] = seconds_mat
#     seconds[i, 2] = seconds_mat_
#     seconds[i, 3] = seconds_mat_rev

# kappa (no need to explain)

# import matplotlib.pyplot as plt
# seconds = np.array([[4.52540000e-04, 1.64543002e-04],
#        [5.14091700e-03, 5.63542700e-03],
#        [1.85514320e-02, 1.90126080e-02],
#        [4.45230638e-01, 5.00560685e-01],
#        [1.80888819e+00, 2.05097258e+00],
#        [6.66040610e+01, 9.29598458e+01]])

# plt.loglog(number_variables_large, seconds); plt.show()

# S = pd.DataFrame(seconds, columns=["loop", "vectorized"], index=number_variables_large)
# S.plot(loglog=True)
# plt.show()





X = np.random.choice([0, 1], size=(number_samples_large, 1000), replace=True)

tic = perf_counter()
np.dot(X.transpose(), X)
toc = perf_counter()
print(f"{toc-tic:0.4f} seconds.")

tic = perf_counter()
np.dot(X, X.transpose())
toc = perf_counter()
print(f"{toc-tic:0.4f} seconds.")


# np.dot
tran_first = """
X = np.random.choice([0, 1], size=(1000, 1000), replace=True);
np.dot(X.transpose(), X)
"""
tran_second = """
X = np.random.choice([0, 1], size=(1000, 1000), replace=True);
np.dot(X, X.transpose())
"""
tf = timeit.Timer(tran_first, setup="import numpy as np")
print(tf.timeit(number=10)/10)

ts = timeit.Timer(tran_second, setup="import numpy as np")
print(ts.timeit(number=10)/10)

# np.matmul
tran_first = """
X = np.random.choice([0, 1], size=(1000, 1000), replace=True);
np.matmul(X.transpose(), X)
"""
tran_second = """
X = np.random.choice([0, 1], size=(1000, 1000), replace=True);
np.matmul(X, X.transpose())
"""
tf = timeit.Timer(tran_first, setup="import numpy as np")
print(tf.timeit(number=10)/10)

ts = timeit.Timer(tran_second, setup="import numpy as np")
print(ts.timeit(number=10)/10)


# -------------------------------------------------------------------
# 🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
# Include in blog (order of operations during MM and data type)

# Now I make data X with same number rows and cols.

# np.dot(X.transpose(), X)  # about 1.4 seconds

# np.dot(X, X.transpose())  # about 0.48 seconds

# This is quite dramatic.

# That was with X dims (1000, 1000)

# Now that was with X dtype int64

# If I use X dtype float64, we get the reverse

# np.dot(X.transpose(), X)  # 0.0085 seconds

# np.dot(X, X.transpose())  # 0.018 seconds

# simulations with int64
#   X = np.random.choice([0, 1], size=(number_samples_large, nvl), replace=True)
#   in loop

# >>> pd.DataFrame(seconds, columns=['loop','mat','mat__','mat_rev'])
#        loop       mat     mat__   mat_rev
# 0  0.000170  0.000219  0.000150  1.727916
# 1  0.004799  0.004580  0.004512  1.537690
# 2  0.018997  0.021193  0.019723  1.728106
# 3  0.453382  0.512756  0.509368  1.610502
# 4  1.822136  2.089260  2.087054  1.666854

# simulations with float64
#   X = np.random.choice([0., 1.], size=(number_samples_large, nvl), replace=True)
#   in loop

# >>> pd.DataFrame(seconds, columns=['loop','mat','mat__','mat_rev'])
#        loop       mat     mat__   mat_rev
# 0  0.000243  0.000364  0.000053  1.764093
# 1  0.004907  0.000346  0.000161  1.574247
# 2  0.018095  0.001281  0.000521  1.572380
# 3  0.481681  0.009352  0.006546  1.913391
# 4  1.988601  0.032003  0.026432  1.992859
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pa_loop(X):
    number_samples, number_variables = X.shape
    percent_agreement = np.zeros((number_variables, number_variables))
    for item_a in range(number_variables):
        for item_b in range(item_a+1, number_variables):
            percent_agreement[item_a, item_b] = (X[:, item_a]==X[:, item_b]).sum()
    percent_agreement /= number_samples
    return(percent_agreement)

def pa_loop_t(X):
    number_variables, number_samples = X.shape
    percent_agreement = np.zeros((number_variables, number_variables))
    for item_a in range(number_variables):
        for item_b in range(item_a+1, number_variables):
            percent_agreement[item_a, item_b] = (X[item_a, :]==X[item_b, :]).sum()
    percent_agreement /= number_samples
    return(percent_agreement)

def pa_vect(X):
    number_samples, _ = X.shape
    yesYes = np.dot(X.transpose(), X)   # counts of yes-yes (k x k)
    np.abs(X-1, out=X)                  # [0,1] -> [1,0]
    noNo = np.dot(X.transpose(), X)     # counts of no-no   (k x k)
    S = yesYes + noNo                   # counts of agreements (k x k)
    A = S / number_samples              # percentage agreements (k x k)   
    return(A)

number_samples_large = 1000
number_variables_large = [10, 50, 100, 500, 1000]
seconds = np.zeros((len(number_variables_large), 3))
for i, nvl in enumerate(number_variables_large):
    print(str(i) + "\n")
    X = np.random.choice([0., 1.], size=(number_samples_large, nvl), replace=True)
    tic = perf_counter(); percent_agreement = pa_loop(X); toc = perf_counter()
    seconds_loop = toc-tic
    XT = X.transpose()
    tic = perf_counter(); percent_agreement = pa_loop_t(XT); toc = perf_counter()
    seconds_loop_t = toc-tic
    tic = perf_counter(); percent_agreement = pa_vect(X); toc = perf_counter()
    seconds_vect = toc-tic
    seconds[i, 0] = seconds_loop
    seconds[i, 1] = seconds_loop_t
    seconds[i, 2] = seconds_vect

df = pd.DataFrame(seconds, 
                  columns=['loop','loop_t','vectorized'],
                  index=number_variables_large)

# seconds = np.array([[3.31700000e-04, 2.42221000e-04, 1.59303570e-02],
#        [9.02653600e-03, 9.12693800e-03, 5.07415000e-04],
#        [4.34402050e-02, 4.33210080e-02, 1.39954200e-03],
#        [8.41689109e-01, 8.08106641e-01, 6.60953810e-02],
#        [3.24740389e+00, 3.25047152e+00, 1.21597452e-01]])

#           loop    loop_t  vectorized
# 10    0.000332  0.000242    0.015930
# 50    0.009027  0.009127    0.000507
# 100   0.043440  0.043321    0.001400
# 500   0.841689  0.808107    0.066095
# 1000  3.247404  3.250472    0.121597

# df.plot(loglog=True, xlabel="Number of variables", ylabel="Seconds")
# plt.show()

# speed_up = seconds[:,0] / seconds[:,1]
# plt.semilogx(number_variables_large, speed_up)
# plt.xlabel("Number of variables")
# plt.ylabel("Speed advantage")
# plt.title("Vectorized operations beats loops")
# plt.show()



# To plot in another terminal because ... strange conda inconsistency
# seconds = np.array([[2.43103000e-04, 3.63571000e-04, 5.31710002e-05, 1.76409273e+00],
#        [4.90698300e-03, 3.46419000e-04, 1.61126000e-04, 1.57424746e+00],
#        [1.80947730e-02, 1.28098600e-03, 5.20920000e-04, 1.57238003e+00],
#        [4.81681406e-01, 9.35171500e-03, 6.54643100e-03, 1.91339133e+00],
#        [1.98860055e+00, 3.20029940e-02, 2.64323600e-02, 1.99285874e+00]])
# df = pd.DataFrame(seconds[:,[0,2]], 
#                   columns=['loop','vectorized_inplace'],
#                   index=number_variables_large)
# df.plot(loglog=True, xlabel="Number of variables", ylabel="Seconds")
# plt.show()

# speed_up = seconds[:,0] / seconds[:,2]
# plt.semilogx(number_variables_large, speed_up)
# plt.xlabel("Number of variables")
# plt.ylabel("Speed advantage")
# plt.title("Vectorized operations beats loops")
# plt.show()
n_samples, n_variables = 1000, 100
responses, probs = [0., 1., M], [.4, .4, .2]
X = np.random.choice(responses, 
                     size=(n_samples, n_variables), 
                     replace=True,
                     p=probs)


# def pa_loop_missing(X):
#     number_samples, number_variables = X.shape
#     percent_agreement = np.zeros((number_variables, number_variables))
#     for item_a in range(number_variables):
#         for item_b in range(item_a+1, number_variables):
#             percent_agreement[item_a, item_b] = (X[:, item_a]==X[:, item_b]).sum()
#     percent_agreement /= number_samples
#     return(percent_agreement)

# Need specific X to test (take form unit tests)
V = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
              [0, 0, 0, 0, 1, 1, 1, 1],
              [0, 1, 0, 1, 0, 1, 0, 1],
              [1, 1, 1, 1, 0, 0, 0, 0]]).transpose()
W = np.array([[0, 0, 0, M, 1, 1, 1, M],
              [0, 0, M, 0, 1, 1, M, 1],
              [0, M, 0, 1, 0, M, 0, 1],
              [M, 1, 1, 1, M, 0, 0, 0]]).transpose()
X = np.r_[V, W]
def pa_loop_missing(X):
    _, number_variables = X.shape
    percent_agreement = np.zeros((number_variables, number_variables))
    for item_a in range(number_variables):
        for item_b in range(item_a+1, number_variables):
                x = X[:, item_a]
                y = X[:, item_b]
                # remove observations where missing for either x or y
                products = [a*b for a, b in zip(x,y)]
                nx = [xi for xi, pi in zip(x, products) if not math.isnan(pi)]
                ny = [yi for yi, pi in zip(y, products) if not math.isnan(pi)]
                sumA = sum([va==vb for va, vb in zip(nx, ny)])
                totA = len(nx)
                percent_agreement[item_a, item_b] = [sumA / totA if totA > 0 else math.nan][0]
    return(percent_agreement)



# def pa_vect(X):
#     number_samples, _ = X.shape
#     yesYes = np.dot(X.transpose(), X)   # counts of yes-yes (k x k)
#     np.abs(X-1, out=X)                  # [0,1] -> [1,0]
#     noNo = np.dot(X.transpose(), X)     # counts of no-no   (k x k)
#     S = yesYes + noNo                   # counts of agreements (k x k)
#     A = S / number_samples              # percentage agreements (k x k)   
#     return(A)

def pa_vect_missing(X):
    R = ma.masked_invalid(X)                # (n x k)
    yesYes = ma.dot(R.transpose(), R)       # counts of yes-yes (k x k)
    F = ma.abs(R-1)                         # [0,1] -> [1,0]
    noNo = ma.dot(F.transpose(), F)         # counts of no-no   (k x k)
    S = yesYes + noNo                       # counts of agreements (k x k)
    valid = np.ones_like(R)                 # valid responses (n x k)
    valid[ma.getmaskarray(R)] = 0
    N = np.dot(valid.transpose(), valid)    # valid count (k x k)
    A = ma.multiply(S, N**-1)               # percentage agreement (k x k)
    return(A)



responses, probs = [0., 1., M], [.4, .4, .2]
number_samples_large = 1000
number_variables_large = [10, 50, 100, 500, 1000]

seconds_missing = np.zeros((len(number_variables_large), 2))
for i, nvl in enumerate(number_variables_large):
    print(str(i) + "\n")
    X = np.random.choice(responses, 
                        size=(number_samples_large, nvl), 
                        replace=True,
                        p=probs)
    tic = perf_counter(); percent_agreement = pa_loop_missing(X); toc = perf_counter()
    seconds_loop = toc-tic
    tic = perf_counter(); percent_agreement = pa_vect_missing(X); toc = perf_counter()
    seconds_vect = toc-tic
    seconds_missing[i, 0] = seconds_loop
    seconds_missing[i, 1] = seconds_vect

df_missing = pd.DataFrame(seconds_missing, 
                          columns=['loop','vectorized'],
                          index=number_variables_large)



# PRIORITY
#
# Check that my loopy functions adhere to this order
#
# Explicit loops in row-major order.
# total = 0
# for i in range(rows):
#     for j in range(cols):
#         total += A[i, j]

# End
# -------------------------------------------------------------------


# Problem with matplotlib in my conda env blog

# df.plot(loglog=True, xlabel="Number of variables", ylabel="Seconds")

# qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
# This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

# Available platform plugins are: eglfs, minimal, minimalegl, offscreen, vnc, webgl, xcb.

# Aborted