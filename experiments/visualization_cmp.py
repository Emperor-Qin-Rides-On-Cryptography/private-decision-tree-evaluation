import os

import matplotlib.pyplot as plt
import pandas as pd
import yaml
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle
from commons import *
# PROJECT_ROOT='/home/r5akhava/private-decision-tree-evaluation/'

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- Custom visual style to make output visually distinct from the original ---
# A tasteful, clearly different color cycle, softer background and grid, bolder lines
from cycler import cycler
custom_colors = ['#2F4F4F', '#E69F00', '#56B4E9', '#009E73', '#D55E00', '#CC79A7']
plt.rcParams.update({
    'axes.prop_cycle': cycler('color', custom_colors),
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#fbfbfb',
    'axes.edgecolor': '#333333',
    'grid.color': '#d0d0d0',
    'grid.linestyle': '--',
    'grid.linewidth': 1.2,
    'lines.linewidth': 2.0,
    'lines.markersize': 7,
    'legend.frameon': True,
    'legend.framealpha': 0.95,
})
# ----------------------------------------------------------------------------


# Helper: annotate points but sample when too many
def annotate_points(ax, x, y, max_annotations=25, fmt=""):
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    if n == 0:
        return
    if n <= max_annotations:
        idxs = list(range(n))
    else:
        # divide into segments and pick the median index in each
        idxs = []
        for i in range(max_annotations):
            start = int(i * n / max_annotations)
            end = int((i + 1) * n / max_annotations)
            if end <= start:
                idx = min(start, n - 1)
            else:
                idx = start + (end - start) // 2
            idxs.append(min(n - 1, max(0, idx)))
        # unique and sorted
        idxs = sorted(set(idxs))

    for i in idxs:
        ax.annotate(fmt.format(y[i]), xy=(x[i], y[i]), xytext=(0, 6), textcoords='offset points',
                    ha='center', fontsize=8, color='#333333')


def plot_line_with_markers(ax, x, y, label=None, annotate=False, inset_ax=None, max_annotations=10, **kw):
    # default styles (marker 'D' is a diamond)
    style = dict(linestyle='-', marker='o', markersize=4, markeredgewidth=0.9)
    style.update(kw)
    line = ax.plot(x, y, label=label, **style)
    # also plot markers in inset if provided
    if inset_ax is not None:
        inset_ax.plot(x, y, marker=style.get('marker', 'D'), markersize=max(5, style.get('markersize', 6)-2),
                      linestyle=style.get('linestyle', '-'), color=line[0].get_color())
    if annotate:
        annotate_points(ax, x, y, max_annotations=max_annotations)
    return line


data = pd.read_csv(
        os.path.join(PROJECT_ROOT,f"experiments/results-{version_tag}/cmp/cmp_bench.csv"),
        # correct,bitlength,comparison,log_deg,hamming_weight,code_length,num_cmps,total_time,amortized_time
        names=["correct","bitlength","comparison","log_deg","hamming_weight","code_length","num_cmps","total_time","amortized_time"],
    )
data['amortized_time'] =data['amortized_time']/1000


fig, ax = plt.subplots()
plt.xlabel("比特精度")
plt.ylabel("毫秒")
plt.title("比较操作的平均运行时间")
ax.set_facecolor('#fcfcfc')
# stronger, more visible grid
ax.grid(True, which='major', linestyle='--', linewidth=1.2, color='#d0d0d0')

# Adjust figure size and main plot position: lower the top so we can place the inset above
# the main axes without covering the plotted curves.
fig.subplots_adjust(left=0.1, bottom=0.1, right=1, top=0.75)

# Create separate Axes object for the magnified part and place it above the main plot,
# centered horizontally so it does not overlap the curves. Coordinates are in figure
# fraction: [left, bottom, width, height].
# Placing bottom just above the main axes top (0.78) keeps it out of the plotting area.
axins = fig.add_axes([0.415, 0.44 ,0.3, 0.3])  # left, bottom, width, height (top-center)
# Style the inset so it looks like an inset panel and sits above the main axes
axins.set_facecolor('white')
axins.patch.set_alpha(1.0)
axins.set_zorder(5)

# Folklore
data1=data[data['comparison']==1]
data1_avg = data1.groupby(['bitlength'], as_index=False).mean()
data1_std = data1.groupby(['bitlength'], as_index=False).std()
plot_line_with_markers(ax, data1_avg['bitlength'], data1_avg['amortized_time'], label="Folklore", inset_ax=axins)
ax.fill_between(
    data1_avg['bitlength'],
    data1_avg['amortized_time']-data1_std['amortized_time'],
    data1_avg['amortized_time']+data1_std['amortized_time'],
    alpha=0.12
)
if axins is not None:
    axins.fill_between(
        data1_avg['bitlength'],
        data1_avg['amortized_time']-data1_std['amortized_time'],
        data1_avg['amortized_time']+data1_std['amortized_time'],
        alpha=0.08
    )

# Range Cover
for hw, x_low, x_high in [
        (2,None, None),
        (4,None, 22),
        (8,None, 34),
        (16,None, None),
        (32,None, None),
    ]:
    data1=data[(data['comparison']==0) & (data['hamming_weight']==hw)]
    if x_low is not None:
        data1 = data1[data1['bitlength']>=x_low]
    if x_high is not None:
        data1 = data1[data1['bitlength']<=x_high]
    
    data1_avg = data1.groupby(['bitlength'], as_index=False).mean()
    data1_std = data1.groupby(['bitlength'], as_index=False).std()
    plot_line_with_markers(ax, data1_avg['bitlength'], data1_avg['amortized_time'], label=f'本作品方案 (h={hw})', inset_ax=axins)
    ax.fill_between(
        data1_avg['bitlength'],
        data1_avg['amortized_time']-data1_std['amortized_time'],
        data1_avg['amortized_time']+data1_std['amortized_time'],
        alpha=0.12
    )
    if axins is not None:
        axins.fill_between(
            data1_avg['bitlength'],
            data1_avg['amortized_time']-data1_std['amortized_time'],
            data1_avg['amortized_time']+data1_std['amortized_time'],
            alpha=0.08
        )

# XCMP
# data1=data[data['comparison']==2]
# data1_avg = data1.groupby(['bitlength'], as_index=False).mean()
# data1_std = data1.groupby(['bitlength'], as_index=False).std()
# ax.plot(data1_avg['bitlength'], data1_avg['amortized_time'], label="XXCMP")
# ax.fill_between(
#     data1_avg['bitlength'],
#     data1_avg['amortized_time']-data1_std['amortized_time'],
#     data1_avg['amortized_time']+data1_std['amortized_time'],
#     alpha=0.1
# )
# axins.plot(data1_avg['bitlength'], data1_avg['amortized_time'])
# axins.fill_between(
#     data1_avg['bitlength'],
#     data1_avg['amortized_time']-data1_std['amortized_time'],
#     data1_avg['amortized_time']+data1_std['amortized_time'],
#     alpha=0.1
# )

# SortingHats (base on their paper)
plot_line_with_markers(ax, [11], [0.4], label="SortingHats", annotate=False, inset_ax=axins, markersize=10)
if axins is not None:
    axins.plot(11, 0.4, "D", markersize=6)


# # Ilia et al.
# avg_times = []
#
# # Open and read the file
# filename='results-v9/ilia/case1.txt'
# with open(filename, 'r') as file:
#     lines = file.readlines()
#
#     # Iterate over the lines
#     for line in lines:
#         # If the line contains 'Avg. time per integer'
#         if "Avg. time per integer" in line:
#             # Extract the value and append to the list
#             value = float(line.split(':')[1].split('ms')[0].strip())
#             avg_times.append(value)
#
# # Calculate and return the average
# ilia = sum(avg_times) / len(avg_times)
# ax.plot(64, ilia, "o", label="Iliashenko et al.")
# axins.plot(64, ilia, "o")

# put the legend top left with a subtle frame and slightly larger font
ax.legend(loc='upper left', frameon=True, edgecolor='#666666', fontsize=10)

# axins.set_yticks([0,2,4,6,8])
# axins.set_yticklabels([0,2,4,6,8])


# Set the limits for the magnifying glass
x1, x2, y1, y2 = 4, 37, -0.5, 5
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

# Add a title to the inset plot
# axins.set_title("Zoomed In")

# Add a box (rectangle) around the magnified part of the main plot
rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='grey', facecolor='none', linestyle="--")
ax.add_patch(rect)

# Create lines connecting the magnified part of the main plot to the magnified part's box
xyA = (x2, y1)
xyB = (x2, y2)
coordsA = "data"
coordsB = "data"
con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA=coordsA, coordsB=coordsB,
                      axesA=axins, axesB=ax, arrowstyle="-", linestyle="--", color="grey", linewidth=1)
axins.add_artist(con)

xyA = (x1, y1)
xyB = (x1, y2)
con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA=coordsA, coordsB=coordsB,
                      axesA=axins, axesB=ax, arrowstyle="-", linestyle="--", color="grey", linewidth=1)
axins.add_artist(con)

# Display the plot
# plt.show()
plt.savefig(f"results-{version_tag}/figures/cmp_compare_all.pdf", bbox_inches='tight')