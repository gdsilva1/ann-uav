import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-3,3,0.1)

def f(x):
    return x**3

def g(x):
    return np.sin(x)

# Use 'seaborn-v0_8-darkgrid' for results/real data 
# Use 'seaborn-v0_8-white' for literature review/theory explanations

plt.style.use(['seaborn-v0_8-darkgrid'])
# plt.style.use(['dracula'])
plt.rcParams.update({
    'axes.grid': True,
    # 'axes.linewidth': 0.5, # Use with seaborn-v0_8-white and dracula
    # 'grid.alpha': 0.1, # Use with dracula
    'lines.linewidth': 0.7,
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

fig, ((ax1,ax2), (ax3,ax4)) =  plt.subplots(2,2, figsize=(5,5)) # figsize=(5,x)

ax1.plot(x,f(x), label='$f(x) = x^3$')
ax1.plot(x,-f(x), label='$f(x) = -x^3$')
ax1.plot(x,2*f(x), label='$f(x) = 2x^3$')
ax1.set_title('Funções Polinomiais', fontsize=12)
# Measure unit is upright
ax1.set_xlabel('Tempo (s)')
ax1.set_ylabel('Trajetória (m)')
ax1.legend(fontsize=10)


ax2.set_xlabel('Tempo (s)')
ax2.set_ylabel('Trajetória (m)')
ax2.plot(x,g(x))
ax2.plot(x,-g(x))
ax2.plot(x,2*g(x))
ax2.set_title('Funções Trigonométricas', fontsize=12)

ax3.plot(x,f(x), label='$f(x) = x^3$')
ax3.plot(x,-f(x), label='$f(x) = -x^3$')
ax3.plot(x,2*f(x), label='$f(x) = 2x^3$')
ax3.set_title('Funções Polinomiais', fontsize=12)
ax3.set_xlabel('Tempo (s)')
ax3.set_ylabel('Trajetória (m)')
ax3.legend()


ax4.set_xlabel('Tempo (s)')
ax4.set_ylabel('Trajetória (m)')
ax4.plot(x,g(x))
ax4.plot(x,-g(x))
ax4.plot(x,2*g(x))
ax4.set_title('Funções Trigonométricas', fontsize=12)


fig.tight_layout(pad=1)
fig.savefig('plot_style.pdf', backend='pgf')
