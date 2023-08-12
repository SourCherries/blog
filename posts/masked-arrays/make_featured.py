import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker


df = pd.read_csv("tests_missing.csv", index_col=0)

df['speedup'] = df.loop / df.vectorized
df['num_vars'] = df.index

speedup_max = df['speedup'].max()

power = np.floor(np.log10(speedup_max))
lower = np.floor(speedup_max / 10**power) * 10**power

# 
# fig, ax = plt.subplots(figsize=(7, 2.7), layout="constrained") # good aspect ratio
dpi = 96
pixels_h, pixels_v = 1280, 640

# with plt.style.context('../featured.538style'):
with plt.style.context(['fivethirtyeight','../featured.538style']):
    fig, ax = plt.subplots(figsize=(pixels_h/dpi, pixels_v/dpi), dpi=dpi)
    # ax.plot(df['num_vars'], 
    #         df['speedup'], 
    #         color='blue', 
    #         linewidth=3, 
    #         linestyle='-',
    #         marker='o')
    ax.plot(df['num_vars'], 
            df['speedup'])
    ax.set_xscale("log")
    ax.set_xlabel("Number of variables")
    ax.set_ylabel("Speed advantage")
    ax.set_title("Vectorization up to ")
    ax.set_title(f"MM more than {lower:.0f}" + r"$\bf{X}$ faster than loops")
    ylim = ax.get_ylim()
    ax.set_ylim((1,ylim[1]))
    ticks_loc = ax.get_yticks().tolist()
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ytickLabels = [str(int(n)) + r"$\bf{X}$" for n in ticks_loc]
    ax.set_yticklabels(ytickLabels)
    # ax.text(x=20, y=3500, s="Masked-matrix multiplication")
    # plt.style.use('fivethirtyeight')  # fivethirtyeight, seaborn-poster
    # plt.style.use('../featured.538style')  # fivethirtyeight, seaborn-poster
    plt.subplots_adjust(left=0.20, right=0.80, top=0.85, bottom=0.20)
    # plt.show()
    fig.savefig("featured.png", dpi=dpi, pad_inches=0.1*pixels_h/dpi)
# featured.png: PNG image data, 700 x 270, 8-bit/color RGBA, non-interlaced
#
# Want 1280 x 640 with 40 pt border
# End
# -------------------------------------------------------------------