#%%

import matplotlib.pyplot as plt
import numpy as np
from math import pi 
from matplotlib import rc
from matplotlib.patches import Patch
import matplotlib
import matplotlib.transforms as mtrans

plt.rcParams['text.usetex'] = True
rc('font', family='sans-serif')#, sans-serif='Times')
plt.rcParams.update({
    "text.usetex": True,
#     "font.family": "serif",
#     "font.sans-serif": ["Times"]})
    "font.family": "sans-serif",
    'text.latex.preamble': r"""\usepackage{bm}\usepackage{amsmath}\usepackage{amssymb}""",
    "font.sans-serif": ["Helvetica"]})
# matplotlib.rcParams['text.latex.preamble'].join([r'\usepackage{amsmath}',
#                                                 r'\usepackage{amssymb}',
#                                                 r"\usepackage{bm}"])

plt.rcParams.update({'font.size': 13})

c100_b2000 = {
    'onproto': [34.40],
    'onproto_jigsaw': [36.56]
}
c100_b500 = {
    'onproto': [21.74],
    'onproto_jigsaw': [24.20]
}
miniimg_b2000 = {
    'onproto': [19.30],
    'onproto_jigsaw': [20.42]
}
miniimg_b8000 = {
    'onproto': [20.48], # todo change
    'onproto_jigsaw': [23.26] # todo change
}
colors = ['#0079b2','#29a035','#9268bb', '#9c9c9c'] # b g v gray
filcolors = [c+'66' for c in colors] # opacity 0.6    
hatches = ['','///']

fig, ax = plt.subplots(figsize=(5,4.5))
# ax.set_title('OnProto') 
ax.set_ylabel(r'Final Average Accuracy (\%)', fontsize=15)
tcifar = r'\textbf{CIFAR-100}'
timg = r'\textbf{\textit{mini}ImageNet}'

bar_width = 0.35
# opacity = 0.4
index = np.arange(4)
filename = 'onproto.pdf'
m = 0
for d, dset in enumerate((c100_b500, c100_b2000, miniimg_b2000, miniimg_b8000)):
    m = max(m, max(dset['onproto']), max(dset['onproto_jigsaw']))
    # ax.bar(index[d], dset['onproto'][0], bar_width, alpha=1, color=colors[-1])

    ax.bar(index[d], dset['onproto'][0], bar_width, color=filcolors[-1], edgecolor=colors[-1], hatch=hatches[0])
    ax.bar(index[d] + bar_width + 0.01, dset['onproto_jigsaw'][0], bar_width, color=filcolors[0], edgecolor=colors[0], hatch=hatches[1])
    # bar with filled color white
    # ax.bar(index[d], dset['onproto'][1], bar_width, alpha=opacity, color='white', edgecolor=colors[-1], linewidth=1)
# m=round(m) + 4
m=42

plt.ylim(0, m)
plt.xlim(-bar_width, index[-1] + 2*bar_width)
# plt.axvspan(-bar_width, 1.675, color='red', alpha=0.1)
# plt.axvspan(1.675, index[-1] + 2*bar_width, color='blue', alpha=0.1)
plt.axvline(x=1.675, color='black', linestyle='--')
l = 6.5
plt.text(0.5, m-l, tcifar, fontsize=15, ha='center')
plt.text(2.65, m-l, timg, fontsize=15, ha='center')

ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(['500', '2000', '2000', '8000'])
ax.set_xlabel(r'Buffer size', fontsize=15)

# ax.set_yticks(np.linspace(0, m, 6).astype(int))
ax.set_yticks(np.arange(0, m+1, 10))

ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')

# ax.set_ylim(0, 30)

ax.xaxis.get_major_formatter()._usetex = False
ax.yaxis.get_major_formatter()._usetex = False

legend_elements = [Patch(facecolor=filcolors[0], edgecolor=colors[0], label='OnPro + CLER', hatch=hatches[1]), 
                     Patch(facecolor=filcolors[-1], edgecolor=colors[-1], label='OnPro', hatch=hatches[0])]
ax.legend(handles=legend_elements, loc='upper center', fancybox=False, 
            framealpha=1, edgecolor='black', fontsize=12,
            bbox_to_anchor=(0.5, 1.015), ncols=2)

# dashed grid 
plt.tight_layout()
plt.savefig('onproto_ptx.pdf', bbox_inches='tight')
plt.show()
# %%
