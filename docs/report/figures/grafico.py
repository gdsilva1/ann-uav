from matplotlib import pyplot as plt
from numpy import linspace, sin, cos

x = linspace(-4,4,10000)

plt.style.use(['bmh'])

plt.rcParams.update({
    'axes.formatter.use_locale': True,
    'text.usetex': True,
    'font.size': 12,
    'pgf.texsystem': 'lualatex',
    'pgf.preamble': r'\usepackage{fontspec,unicode-math}\setmainfont{STIX Two Text} \setmathfont{STIX Two Math}'
})

cm=1/2.54
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14.1*cm, 7*cm))

ax1.plot(x,sin(x), label=r'$f(x) = \sin(x)$')
ax1.plot(x,cos(x), label=r'$f(x) = \cos(x)$')
ax1.legend()

ax2.plot(x,sin(x))
ax2.plot(x,cos(x))

fig.tight_layout()
plt.savefig('graf.pdf', backend='pgf')