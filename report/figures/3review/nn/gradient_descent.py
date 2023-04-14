import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def loss_fn(x):
    return x**2

x_cont = np.arange(-2,1,0.01)
x_points = np.array([2,1.5,1,0.6,0.3,0.1,0])*(-1)



plt.style.use(['seaborn-v0_8-darkgrid', 'seaborn-v0_8-dark-palette'])
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'pgf.texsystem': 'xelatex',
    # 'pgf.preamble':  r'\usepackage{stix2}\usepackage[scale=0.88]{inter}',
    # 'pgf.preamble': r'\usepackage{unicode-math,fontspec}\setmathfont{STIX Two Math}\setsansfont{Inter}[Scale=0.88]',
    'pgf.preamble': r'\usepackage[default]{fontsetup}',
    # 'pgf.preamble': r'\usepackage{unicode-math,fontspec}\setmathfont{STIX Two Math}\setmainfont{STIX Two Text}\setsansfont{TeX Gyre Heros}[Scale=0.92, LetterSpace=-0.2]',
    'pgf.rcfonts' : False
})


fig, ax =  plt.subplots(1,1, figsize=(5,3)) # figsize=(5,x)

ax.scatter(x_points, loss_fn(x_points), label='Learning Steps', color='darkred', zorder=2)
ax.plot(x_cont, loss_fn(x_cont), label='Loss Function', zorder=1)
ax.set_xlabel('Weight')
ax.set_ylabel('Loss Function')
ax.legend(loc=1, frameon=True)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.annotate(
    text='Initial value',
    xy=(-2,4),
    xytext=(-1,3.8),
    verticalalignment='center',
    horizontalalignment='center',
    arrowprops={
        'arrowstyle': 'wedge',
        'facecolor': 'black'
    },
    

)
ax.annotate(
    'Minimum',
    (-0,0),
    verticalalignment='center',
    horizontalalignment='center',
    arrowprops={
        'arrowstyle': 'wedge',
        'facecolor': 'black'
    },
    xytext=(0.25,1.5),

)
fig.tight_layout()
fig.savefig('figures/3review/nn/gradient_descent.pdf', backend='pgf')
# plt.show()