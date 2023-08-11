import math
import numpy as np
import numpy.ma as ma
from tqdm import tqdm
from time import perf_counter
import pandas as pd
import matplotlib.pyplot as plt


# Missing value
M = np.nan
rng = np.random.default_rng(seed=12345)


# -------------------------------------------------------------------
# Functions to calculate percentage agreement
def pa_loop(X):
    number_samples, number_variables = X.shape
    percent_agreement = np.zeros((number_variables, number_variables))
    for item_a in range(number_variables):
        for item_b in range(item_a+1, number_variables):
            percent_agreement[item_a, item_b] = (X[:, item_a]==X[:, item_b]).sum()
    percent_agreement /= number_samples
    return(percent_agreement)

def pa_vect(X):
    number_samples, number_variables = X.shape # (n x k)
    yesYes = np.dot(X.transpose(), X)          # counts of yes-yes (k x k)
    np.abs(X-1, out=X)                         # [0,1] -> [1,0]
    noNo = np.dot(X.transpose(), X)            # counts of no-no   (k x k)
    S = yesYes + noNo                          # counts of agreements (k x k)
    A = S / number_samples                     # percentage agreements (k x k)   
    return(A)

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


# -------------------------------------------------------------------
# Tests

# No missing values -------------------------------------------------
number_samples = 1000
number_variables = [10, 50, 100, 500, 1000]
seconds = np.zeros((len(number_variables), 2))
iterations = 36

for i, nvl in enumerate(number_variables):
    for _ in range(iterations):
        X = rng.choice([0., 1.], size=(number_samples, nvl), replace=True)
        tic = perf_counter(); percent_agreement = pa_loop(X); toc = perf_counter()
        seconds_loop = toc-tic
        tic = perf_counter(); percent_agreement = pa_vect(X); toc = perf_counter()
        seconds_vect = toc-tic
        seconds[i, 0] += seconds_loop
        seconds[i, 1] += seconds_vect

seconds /= iterations
    
df = pd.DataFrame(seconds, 
                  columns=['loop','matrix-based'],
                  index=number_variables)

# df.plot(loglog=True, xlabel="Number of variables", ylabel="Seconds")
# plt.show()

df.to_csv("tests.csv")

# Missing values ----------------------------------------------------
responses, probs = [0., 1., M], [.4, .4, .2]
number_samples_large = 1000
number_variables_large = [10, 50, 100, 500, 1000]
seconds_missing = np.zeros((len(number_variables_large), 2))
iterations = 2
for i, nvl in enumerate(number_variables_large):
  for _ in tqdm(range(iterations)):
    print(f"\n\n{i}**********************************************\n\n")
    X = rng.choice(responses, 
                   size=(number_samples_large, nvl), 
                   replace=True,
                   p=probs)
    tic = perf_counter()
    percent_agreement = pa_loop_missing(X)
    toc = perf_counter()
    seconds_loop = toc-tic
    tic = perf_counter()
    percent_agreement = pa_vect_missing(X)
    toc = perf_counter()
    seconds_vect = toc-tic
    seconds_missing[i, 0] += seconds_loop
    seconds_missing[i, 1] += seconds_vect

seconds_missing /= iterations

df_missing = pd.DataFrame(seconds_missing, 
                          columns=['loop','vectorized'],
                          index=number_variables_large)

# df_missing.plot(loglog=True, xlabel="Number of variables", ylabel="Seconds")
# plt.show()

df_missing.to_csv("tests_missing.csv")


import matplotlib.ticker as mticker


speedup = seconds_missing[:,0]/seconds_missing[:,1]
power = np.floor(np.log10(speedup.max()))
lower = np.floor(speedup.max() / 10**power) * 10**power

fig, ax = plt.subplots(figsize=(7, 2.7), layout="constrained")  # Create a figure containing a single axes.
ax.plot(number_variables_large, 
        speedup, 
        color='blue', 
        linewidth=3, 
        linestyle='-',
        marker='o')
ax.set_xscale("log")
ax.set_xlabel("Number of variables")
ax.set_ylabel("Speed advantage")
ax.set_title("Vectorization up to ")
ax.set_title(f"Vectorized code more than {lower:.0f}" + r"$\bf{X}$ faster than loops")
ylim = ax.get_ylim()
ax.set_ylim((1,ylim[1]))
# fixing yticks with matplotlib.ticker "FixedLocator"
ticks_loc = ax.get_yticks().tolist()
ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
ytickLabels = [str(int(n)) + r"$\bf{X}$" for n in ticks_loc]
ax.set_yticklabels(ytickLabels)
# yticks = ax.get_yticks()
# ytickLabels = [str(int(n)) + r"$\bf{X}$" for n in yticks]
# ax.set_yticklabels(ytickLabels)
plt.show()

fig.savefig("tests-missing.svg")








import seaborn as sns
df_missing['speedup'] = df_missing.loop / df_missing.vectorized
df_missing['num_vars'] = number_variables_large

sns.set_theme(style="ticks", font_scale=1.25)
sns.relplot(
    data=df_missing, kind="line",
    x="num_vars", y="speedup",
    markers=True
)
plt.show()


# End
# -------------------------------------------------------------------


# X = np.random.choice(responses, size=(1000, 1000), replace=True, p=probs)
# tic = perf_counter(); percent_agreement = pa_loop_missing(X); toc = perf_counter()
# seconds_loop = toc-tic
# tic = perf_counter(); percent_agreement = pa_vect_missing(X); toc = perf_counter()
# seconds_vect = toc-tic
# print(f"\n\n{seconds_loop} seconds for loop\n{seconds_vect} seconds for vect\n\n")