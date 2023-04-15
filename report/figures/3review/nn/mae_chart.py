import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

y_data = np.array([0.8, 2.5, 3.2, 3.5, 5.9])
x_data = np.array([1, 2, 3, 4, 5])

plt.style.use(['seaborn-v0_8-white'])
plt.rcParams.update({
    'axes.linewidth': 0.5,
    'lines.linewidth': 0.7,
    'font.family': 'serif',
    'font.size': 12,
    'pgf.texsystem': 'xelatex',
    # 'pgf.preamble':  r'\usepackage{stix2}\usepackage[scale=0.88]{inter}',
    # 'pgf.preamble': r'\usepackage{unicode-math,fontspec}\setmathfont{STIX Two Math}\setsansfont{Inter}[Scale=0.88]',
    'pgf.preamble': r'\usepackage[default]{fontsetup}',
    # 'pgf.preamble': r'\usepackage{unicode-math,fontspec}\setmathfont{STIX Two Math}\setmainfont{STIX Two Text}\setsansfont{TeX Gyre Heros}[LetterSpace=-0.2]',
    'pgf.rcfonts' : False
})


fig, ax =  plt.subplots(1,1, figsize=(5,3)) # figsize=(5,x)


ax.plot([5,5], [5,5.9], color='red')
ax.scatter(x_data,y_data, label='Prediction Data', zorder=2)
ax.plot(y_data,y_data, label='Target Data')
ax.set_xlabel('Input data')
ax.set_ylabel('Output data')
ax.legend(loc=4, frameon=True)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.annotate(
    text='Distance between target\nand prediction data',
    xy=(5,5.5),
    xytext=(2.7,5.5),
    verticalalignment='center',
    horizontalalignment='center',
    arrowprops={
        'arrowstyle': 'wedge',
        'facecolor': 'black'
    },
)
fig.tight_layout()
fig.savefig('mae_chart.pdf', backend='pgf')
# fig.savefig('figures/3review/nn/mae_chart.pdf', backend='pgf')
# plt.show()