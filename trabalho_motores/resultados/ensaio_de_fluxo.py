import pandas as pd
import numpy as np
import locale
import seaborn as sns
import matplotlib.pyplot as plt
import sympy as sp

locale.setlocale(locale.LC_ALL, 'pt_BR')
plt.style.use('duarte')


def area_cortina(d_valvula, levante, n=3):
    return n * np.pi * d_valvula * levante

def area_garganta(d_valvula, d_haste, n=3):
    return n * np.pi/4 * (d_valvula**2 - d_haste**2)

def rho(P, R, T):
    return P / (R*T)

def velocidade_isentropica(P0, rho0, deltaP, gamma=1.4):
    rhoS = rho0 * (1-deltaP/P0)**(1/gamma)

    f1 = P0/rhoS
    f2 = (2*gamma) / (gamma - 1)
    f3 = 1 - (1-(deltaP/P0))**((gamma-1)/gamma)
    

    return (f1*f2*f3)**0.5

def area_efetiva(mex, p1, p2, R, T, gamma=1.4):
    n = mex * np.sqrt(gamma*R*T)
    
    df1 = gamma * p1 * (p2/p1)**(1/gamma)
    df2 = np.sqrt((2/(gamma-1)) * (1-(p2/p1)**(1-1/gamma)))

    return n/(df1*df2)

def cd(vazao_real, vel_isentropica, area):
    Qr = vazao_real
    Qt = vel_isentropica * area
    return Qr/Qt

def plot_fluxo(levante, fluxo_ava, fluxo_eve, fluxo_eva, unidade_fluxo='m/s', unidade_levante='m'):
    MARKER = '.'
    CM = 1/2.54
    plt.cla()
    plt.clf()
    fig, ax = plt.subplots(figsize=(16*CM, 8*CM))

    ax.plot(levante * 1e3, 
            fluxo_ava * 2119, 
            label='AVA',
            marker=MARKER)
    ax.plot(levante * 1e3, 
            fluxo_eve * 2119, 
            label='EVE',
            marker=MARKER)
    ax.plot(levante * 1e3, 
            fluxo_eva * 2119, 
            label='EVA',
            marker=MARKER)
    ax.legend(loc=4)
    ax.set_xlabel(f'Levante da válvula ({unidade_levante})')
    ax.set_ylabel(f'Fluxo ({unidade_fluxo})')

    fig.tight_layout()
    fig.savefig('../relatorio/figuras/levante_fluxo.pgf',
                backend='pgf')
    fig.savefig('../relatorio/figuras/levante_fluxo.pdf',
                backend='pgf')
    

def plot_cd(levante, 
            cd_garganta_ava, 
            cd_garganta_eve, 
            cd_garganta_eva, 
            cd_amin_ava,
            cd_amin_eve,
            cd_amin_eva):
    MARKER = '.'
    CM = 1/2.54
    plt.cla()
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16*CM, 8*CM))

    ax1.plot(levante * 1e3, 
            cd_garganta_ava, 
            label='AVA', 
            marker=MARKER)
    ax1.plot(levante * 1e3, 
            cd_garganta_eve, 
            label='EVE', 
            marker=MARKER)
    ax1.plot(levante * 1e3, 
            cd_garganta_eva, 
            label='EVA', 
            marker=MARKER)
    ax1.set_xlabel('Levante da válvula (mm)')
    ax1.set_ylabel('Coeficiente de Descarga, $C_d$')
    ax1.set_title('Área de garganta')

    ax2.plot(levante * 1e3, 
            cd_amin_ava, 
            label='AVA', 
            marker=MARKER)
    ax2.plot(levante * 1e3, 
            cd_amin_eve, 
            label='EVE', 
            marker=MARKER)
    ax2.plot(levante * 1e3, 
            cd_amin_eva, 
            label='EVA', 
            marker=MARKER)
    ax2.set_xlabel('Levante da válvula (mm)')
    ax2.set_ylabel('Coeficiente de Descarga, $C_d$')
    ax2.set_title('Área mínima')

    h, l = ax1.get_legend_handles_labels()
    fig.legend(h, l, ncols=3, loc='lower center', bbox_to_anchor=(0.5,-0.1))
    fig.tight_layout()
    fig.savefig('../relatorio/figuras/levante_cd_new.pgf',
                backend='pgf', bbox_inches='tight')
    fig.savefig('../relatorio/figuras/levante_cd_new.pdf',
                backend='pgf', bbox_inches='tight')


ge = pd.read_excel(io='dados.xlsx',
                   sheet_name='geometria',
                   header=[0,1])

ef = pd.read_excel('ensaio_fluxo.xlsx',
                   header=[0,1])

MARKER='.'
CM = 1/2.54

D_VAL_E = ge.describe().loc['mean'].escape.d_valvula * 1e-3    # m
D_VAL_A = ge.describe().loc['mean'].admissao.d_valvula * 1e-3  # m``
H_VAL_E = ge.describe().loc['mean'].escape.h_valvula * 1e-3    # m
H_VAL_A = ge.describe().loc['mean'].admissao.h_valvula * 1e-3  # m
D_GARG_E = ge.describe().loc['mean'].escape.d_int * 1e-3       # m
D_GARG_A = ge.describe().loc['mean'].admissao.d_int * 1e-3     # m
P_ENSAIO = 25 * 249                                            # Pa
P0 = 96.3 * 1e3                                                # Pa
T0 = 25.8 + 273.15                                             # K
AG_ADM = area_garganta(D_GARG_A, H_VAL_A)                       # m2
AG_ESC = area_garganta(D_GARG_E, H_VAL_E)                       # m2
RHO_AR = rho(P = P0,        # Pa
             T = T0,        # K
             R = 287        # J/kg*K (Tab. A1 - ÇENGEL)
             )                                                 # kg/m3
VTH = velocidade_isentropica(P0=P0,
                             rho0=RHO_AR,
                             deltaP=P_ENSAIO)                  # m/s

fluxo_ava = ef.ava.fluxo / 2119                                # m3/s
fluxo_eve = ef.eve.fluxo / 2119                                # m3/s
fluxo_eva = ef.eva.fluxo / 2119                                # m3/s
levante = np.arange(1,11,1) * 1e-3                             # m
ac_adm = area_cortina(D_VAL_A, levante)                        # m2
ac_esc = area_cortina(D_VAL_E, levante)                        # m2

# Calculo da area efetiva
a_efetiva_ava = area_efetiva(mex=fluxo_ava,
                             p1=P0,
                             p2=P0-P_ENSAIO,
                             R=287,
                             T=T0)                             # m2
a_efetiva_eve = area_efetiva(mex=fluxo_eve,
                             p1=P0+P_ENSAIO,
                             p2=P0,
                             R=287,
                             T=T0)                             # m2
a_efetiva_eva = area_efetiva(mex=fluxo_eva,
                             p1=P0+P_ENSAIO,
                             p2=P0,
                             R=287,
                             T=T0)                             # m2

# Calculo do coeficiente de descarga considerando a área de garganta
# cd_garganta_ava = cd(vazao_real=fluxo_ava,
#                      vel_isentropica=VTH,
#                      area=AG_ADM)

# cd_garganta_eve = cd(vazao_real=fluxo_eve,
#                      vel_isentropica=VTH,
#                      area=AG_ADM)

# cd_garganta_eva = cd(vazao_real=fluxo_eva,
#                      vel_isentropica=VTH,
#                      area=AG_ADM)

# Calculo do coeficiente de descarga considerando a área mínima
amin_adm = np.array([ac if ac < AG_ADM else AG_ADM for ac in ac_adm])
amin_esc = np.array([ac if ac < AG_ESC else AG_ESC for ac in ac_esc])

# cd_amin_ava = cd(vazao_real=fluxo_ava,
#                 vel_isentropica=VTH,
#                 area=amin_adm)

# cd_amin_eve = cd(vazao_real=fluxo_eve,
#                 vel_isentropica=VTH,
#                 area=amin_esc)

# cd_amin_eva = cd(vazao_real=fluxo_eva,
#                 vel_isentropica=VTH,
#                 area=amin_adm)

# Calculo do coeficiente de descarga considerando area de garganta (new)
cd_garganta_ava_new = a_efetiva_ava/AG_ADM
cd_garganta_eve_new = a_efetiva_eve/AG_ESC
cd_garganta_eva_new = a_efetiva_eva/AG_ADM


# Calculo do coeficiente de descarga considerando area minima (new)
cd_amin_ava_new = a_efetiva_ava/amin_adm
cd_amin_eve_new = a_efetiva_eve/amin_esc
cd_amin_eva_new = a_efetiva_eva/amin_adm

# Plot dos gráficos
# plot_fluxo(levante=levante,
#            fluxo_ava=fluxo_ava,
#            fluxo_eve=fluxo_eve,
#            fluxo_eva=fluxo_eva,
#            unidade_fluxo='cfm',
#            unidade_levante='mm')

plot_cd(levante=levante,
        cd_amin_ava=cd_amin_ava_new,
        cd_amin_eve=cd_amin_eve_new,
        cd_amin_eva=cd_amin_eva_new,
        cd_garganta_ava=cd_garganta_ava_new,
        cd_garganta_eve=cd_garganta_eve_new,
        cd_garganta_eva=cd_garganta_eva_new)

# plt.show()





