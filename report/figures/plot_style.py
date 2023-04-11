import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-3,3,0.1)

def f(x):
    return x**3

def g(x):
    return np.sin(x)

plt.style.use(['seaborn-v0_8-darkgrid', 'seaborn-v0_8-dark-palette'])
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'pgf.texsystem': 'xelatex',
    # 'pgf.preamble':  r'\usepackage{stix2}\usepackage[scale=0.88]{inter}',
    # 'pgf.preamble': r'\usepackage{unicode-math,fontspec}\setmathfont{STIX Two Math}\setsansfont{Inter}[Scale=0.88]',
    # 'pgf.preamble': r'\usepackage[default]{fontsetup}',
    'pgf.preamble': r'\usepackage{unicode-math,fontspec}\setmathfont{STIX Two Math}\setmainfont{STIX Two Text}\setsansfont{TeX Gyre Heros}',
    'pgf.rcfonts' : False
})

fig, ((ax1,ax2), (ax3,ax4)) =  plt.subplots(2,2, figsize=(5,5)) # figsize=(5,x)

ax1.plot(x,f(x), label='$f(x) = x^3$')
ax1.plot(x,-f(x), label='$f(x) = -x^3$')
ax1.plot(x,2*f(x), label='$f(x) = 2x^3$')
ax1.set_title('Funções Polinomiais')
# Measure unit is upright
ax1.set_xlabel('Tempo (s)')
ax1.set_ylabel('Trajetória (m)')
ax1.legend(fontsize='small')


ax2.set_xlabel('Tempo (s)')
ax2.set_ylabel('Trajetória (m)')
ax2.plot(x,g(x))
ax2.plot(x,-g(x))
ax2.plot(x,2*g(x))
ax2.set_title('Funções Trigonométricas')

ax3.plot(x,f(x), label='$f(x) = x^3$')
ax3.plot(x,-f(x), label='$f(x) = -x^3$')
ax3.plot(x,2*f(x), label='$f(x) = 2x^3$')
ax3.set_title('Funções Polinomiais')
ax3.set_xlabel('Tempo (s)')
ax3.set_ylabel('Trajetória (m)')
ax3.legend()


ax4.set_xlabel('Tempo (s)')
ax4.set_ylabel('Trajetória (m)')
ax4.plot(x,g(x))
ax4.plot(x,-g(x))
ax4.plot(x,2*g(x))
ax4.set_title('Funções Trigonométricas')


fig.tight_layout(pad=1)
fig.savefig('plot_style.pdf', backend='pgf')
