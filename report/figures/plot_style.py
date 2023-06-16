import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-3,3,0.1)

def f(x):
    return x**3

def g(x):
    return np.sin(x)

plt.style.use(['seaborn-v0_8-darkgrid', 'seaborn-v0_8-colorblind'])
# from cycler import cycler
plt.rcParams.update({
    'axes.prop_cycle': plt.cycler(linestyle=['solid','dashed','dotted', 'dashdot']),
    # 'axes.facecolor': '#e6e6e6',
    'axes.grid': True,
    # 'axes.linewidth': 0.5, # Use with seaborn-v0_8-white and dracula
    # 'grid.alpha': 0.1, # Use with dracula
    'lines.linewidth': 1,
    'grid.linewidth': 0.5,
    'font.family': 'serif',
    'font.size': 12,
    'pgf.texsystem': 'xelatex',
    # 'pgf.preamble':  r'\usepackage{stix2}\usepackage[scale=0.88]{inter}',
    # 'pgf.preamble': r'\usepackage{unicode-math,fontspec}\setmathfont{STIX Two Math}\setsansfont{Inter}[Scale=0.88]',
    'pgf.preamble': r'\usepackage[default]{fontsetup}',
    # 'pgf.preamble': r'\usepackage{unicode-math,fontspec}\setmathfont{STIX Two Math}\setmainfont{STIX Two Text}\setsansfont{TeX Gyre Heros}[LetterSpace=-0.2]',
    'pgf.rcfonts' : False
})

fig, ax =  plt.subplots(2,2, figsize=(5,5)) # figsize=(5,x)

ax[0,0].plot(x,f(x), label='$f(x) = x^3$')
ax[0,0].plot(x,-f(x), label='$f(x) = -x^3$')
ax[0,0].plot(x,2*f(x), label='$f(x) = 2x^3$')
ax[0,0].plot(x,0.5*f(x), label='$f(x) = \\frac{1}{2}x^3$')
ax[0,0].legend(fontsize=6, loc='lower center')

ax[0,1].plot(x,g(x))
ax[0,1].plot(x,-g(x))
ax[0,1].plot(x,2*g(x))

ax[1,0].plot(x,f(x), label='$f(x) = x^3$')
ax[1,0].plot(x,-f(x), label='$f(x) = -x^3$')
ax[1,0].plot(x,2*f(x), label='$f(x) = 2x^3$')
ax[1,0].legend()

ax[1,1].plot(x,g(x))
ax[1,1].plot(x,-g(x))
ax[1,1].plot(x,2*g(x))

for i,j in zip(range(1),range(1)):

    ax[i,j].set_title('Funções')
    ax[i,j].set_xlabel('Tempo (s)')
    ax[i,j].set_ylabel('Trajetória (m)')


fig.tight_layout(pad=1)
# fig.savefig('figures/plot_style.pgf', backend='pgf')
fig.savefig('plot_style.pdf', backend='pgf')

plt.show()
