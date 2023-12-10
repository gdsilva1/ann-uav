import pandas as pd
import numpy as np
import locale
import seaborn as sns
import matplotlib.pyplot as plt
import sympy as sp
from ensaio_de_fluxo import velocidade_isentropica, rho
from uncertainties import unumpy, ufloat
from ensaio_de_fluxo import area_garganta

locale.setlocale(locale.LC_ALL, 'pt_BR')
plt.style.use('duarte')

def carga_lateral(potencia, cilindrada, rotacao, diametro, RL, alpha=np.pi/2, return_bmep=False):
    P = (2 * potencia) / (cilindrada * rotacao)

    n = np.pi * P * diametro * RL * np.sin(alpha)
    d = 4*np.sqrt(1 - RL**2*(np.sin(alpha))**2)

    if return_bmep:
        return n/d, P
    else:
        return n/d
    
def velocidade_instantanea(r, l, omega, t):
    f1 = r * omega * unumpy.sin(omega * t)
    f2 = 1 + (r*unumpy.cos(omega * t)) / (l**2-r**2*(unumpy.sin(omega * t))**2)**0.5

    return f1*f2

def mach(vel, vel_som=343):
    return vel / vel_som

def mach_fergunson(b, Up, Af, ci=343):
    z = (np.pi * b**2 * Up) / (Af * ci)
    return z
    
def plot_pressao_torque_potencia_carga_lateral(rotacao,
                                               erro_rotacao,
                                               pressao,
                                               erro_pressao,
                                               torque,
                                               erro_torque, 
                                               potencia, 
                                               erro_potencia,
                                               carga_lateral,
                                               erro_carga_lateral):
    
    CM = 1/2.54
    MARKER='.'
    CAPSIZE = 3
    CAPTHICK = 0.5
    ELINEW = 0.5
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(16*CM, 16*CM))
    
    # PRESSAO MEDIA X ROTACAO
    ax1.errorbar(x=rotacao * 60,
                 xerr=erro_rotacao * 60,
                 y=pressao * 1e-6,
                 yerr=erro_pressao * 1e-6,
                 capsize=CAPSIZE,
                 capthick=CAPTHICK,
                 elinewidth=ELINEW,
                 marker=MARKER)
    ax1.set_xlabel('Rotação (RPM)')
    ax1.set_ylabel('BMEP (MPa)')
    ax1.minorticks_off()

    # TORQUE NOMINAL X ROTACAO
    ax2.errorbar(x=rotacao * 60,
                 xerr=erro_rotacao * 60,
                 y=torque,
                 yerr=erro_torque,
                 capsize=CAPSIZE,
                 capthick=CAPTHICK,
                 elinewidth=ELINEW,
                 marker=MARKER)
    ax2.set_xlabel('Rotação (RPM)')
    ax2.set_ylabel('Torque (N$\cdot$m)')
    ax2.minorticks_off()

    # POTENCIA X ROTACAO
    ax3.errorbar(x=rotacao * 60,
                 xerr=erro_rotacao * 60,
                 y=potencia * 1e-3,
                 yerr=erro_potencia * 1e-3,
                 capsize=CAPSIZE,
                 capthick=CAPTHICK,
                 elinewidth=ELINEW,
                 marker=MARKER)
    ax3.set_xlabel('Rotação (RPM)')
    ax3.set_ylabel('Potência (kW)')
    ax3.minorticks_off()

    # CARGA LATERAL X ROTACAO
    ax4.errorbar(x=rotacao * 60,
                 xerr=erro_rotacao * 60,
                 y=carga_lateral * 1e-3,
                 yerr=erro_carga_lateral * 1e-3,
                 capsize=CAPSIZE,
                 capthick=CAPTHICK,
                 elinewidth=ELINEW,
                 marker=MARKER)
    ax4.set_xlabel('Rotação (RPM)')
    ax4.set_ylabel('Carga Lateral (kN)')
    ax4.minorticks_off()

    fig.tight_layout()
    fig.savefig('../relatorio/figuras/PTPotF_rot.pgf',
                backend='pgf')
    fig.savefig('../relatorio/figuras/PTPotF_rot.pdf',
                backend='pgf')
    
def plot_velocidades(rotacao,
                     erro_rotacao,
                     rotacao_original,
                     erro_rotacao_original,
                     t,
                     velocidade_media,
                     erro_velocidade_media,
                     velocidade_instantanea,
                     erro_velocidade_instantanea):
    CM = 1/2.54
    MARKER='.'
    CAPSIZE = 3
    CAPTHICK = 0.5
    ELINEW = 0.5
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16*CM, 8*CM))

    # VELOCIDADE MEDIA X ROTACAO
    ax1.errorbar(x=rotacao * 60,
                 xerr=erro_rotacao * 60,
                 y=velocidade_media,
                 yerr=erro_velocidade_media,
                 capsize=CAPSIZE,
                 capthick=CAPTHICK,
                 elinewidth=ELINEW,
                 marker=MARKER)
    ax1.set_xlabel('Rotação (RPM)')
    ax1.set_ylabel('Velocidade Média (m/s)')

    # VELOCIDADE INSTANTANEA X ROTACAO
    ax2.scatter(x=rotacao_original * 60,
                y=velocidade_instantanea,                marker=MARKER)
    ax2.set_xlabel('Rotação (RPM)')
    ax2.set_ylabel('Velocidade Instantânea (m/s)')

    fig.tight_layout()
    fig.savefig('../relatorio/figuras/velocidades_rot.pgf',
                backend='pgf')
    fig.savefig('../relatorio/figuras/velocidades_rot.pdf',
                backend='pgf')
    

def pressao_boost(rotacao,
                  erro_rotacao,
                  pressao_boost,
                  erro_pressao_boost,
                  bmep,
                  erro_bmep):
    CM = 1/2.54
    MARKER = '.'
    CAPSIZE = 3
    CAPTHICK = 0.5
    ELINEW = 0.5
    fig, ax = plt.subplots(figsize=(16*CM, 8*CM))

    ax.errorbar(x=rotacao * 60,
                xerr=erro_rotacao * 60,
                y=pressao_boost * 1e-3,
                yerr=erro_pressao_boost * 1e-3,
                marker=MARKER,
                capsize=CAPSIZE,
                capthick=CAPTHICK,
                elinewidth=ELINEW,
                color='black')
    ax.set_xlabel('Rotação (RPM)')
    ax.set_ylabel('Pressão de Boost (kPa)')

    ax1 = ax.twinx()
    ax1.errorbar(x=rotacao * 60,
                 xerr=erro_rotacao * 60,
                 y=bmep * 1e-6,
                 yerr=erro_bmep * 1e-6,
                 capsize=CAPSIZE,
                 capthick=CAPTHICK,
                 elinewidth=ELINEW,
                 marker=MARKER)
    ax1.set_xlabel('Rotação (RPM)')
    ax1.set_ylabel('BMEP (MPa)', color='#0093dd')
    ax1.tick_params(axis='y', labelcolor='#0093dd')

    fig.tight_layout()
    fig.savefig('../relatorio/figuras/pressao_boost_rotacao.pgf',
                backend='pgf')
    fig.savefig('../relatorio/figuras/pressao_boost_rotacao.pdf',
                backend='pgf')



ed = pd.read_excel(io='ensaio_din.xlsx',
                   skiprows=[x for x in range(5,12)],
                   header=[3,4])
ed.columns = ['_'.join(col).strip() for col in ed.columns.values]
ed = ed[~ed['NOME_PASSO_Unnamed: 5_level_1'].str.contains(' ESTAB')]
ed['NOME_PASSO_Unnamed: 5_level_1'] = ed['NOME_PASSO_Unnamed: 5_level_1'].astype(int)  
ed.drop(columns=['DATA_DD/MM/AA',
                 'HORA_HH:MM:SS',
                 'HORIM_TOTAL_HH:MM',
                 'HORIM_AUTO_HH:MM',
                 'NÚMERO_PASSO_Unnamed: 4_level_1',
                 'OBSERVAÇÃO_Unnamed: 6_level_1',
                 'CICLO_Unnamed: 7_level_1',
                 'FC_NBR ISO 1585',
                 'RES_4_Unnamed: 27_level_1',
                 'RES_5_Unnamed: 28_level_1',
                 'RES_6_Unnamed: 29_level_1',
                 'RES_7_Unnamed: 30_level_1',
                 'REGISTROS_Unnamed: 32_level_1'],
        inplace=True)

ed.rename(columns={'NOME_PASSO_Unnamed: 5_level_1': 'ROTAÇÃO_NOMINAL'}, inplace=True)
ed_original = ed
ed = ed.groupby('ROTAÇÃO_NOMINAL').agg([np.mean, np.std]).reset_index()

ed_original

CILINDRADA = 6.8 * 1e-3                                # m3
D_CILINDRO = 106 * 1e-3                                # m
CURSO = (2 * CILINDRADA) / (3 * np.pi * D_CILINDRO**2) # m
R_MANIVELA = CURSO/2                                   # m
L_BIELA = 203 * 1e-3                                   # m 
RL = R_MANIVELA / L_BIELA
TORQUE_MAXIMO = 934                                    # N.m
POTENCIA_MAXIMA = 187 * 1e3                            # W
ROTACAO_TORQUE_MAXIMO = 1400 / 60                      # RPS
ROTACAO_POTENCIA_MAXIMA = 2400 / 60                    # RPS
V_SOM = 343                                            # m/s
P_ATM = 96654.2                                        # Pa

torque = unumpy.uarray(ed['TORQUE_Nm','mean'], ed['TORQUE_Nm','std']) # N.m
rotacao_real = unumpy.uarray(ed['ROTAÇÃO_rpm','mean'], ed['ROTAÇÃO_rpm','std']) / 60 # RPS
rotacao_original = ed_original['ROTAÇÃO_rpm'] / 60 # RPS
rotacao_intervalada = ed.ROTAÇÃO_NOMINAL / 60                       # RPS
potencia = unumpy.uarray(ed['POT_EFET_kW','mean'], ed['POT_EFET_kW','std']) * 1e3 # W
p_col_adm = unumpy.uarray(ed['P_COL_ADM_mbar','mean'], ed['P_COL_ADM_mbar','std']) * 1e2 # Pa
temp_col_adm = unumpy.uarray(ed['T_COL_ADM_ºC', 'mean'], ed['T_COL_ADM_ºC', 'std']) + 273.15 # K
temp_amb = unumpy.uarray(ed['T_AMBIENTE_ºC', 'mean'], ed['T_AMBIENTE_ºC', 'std']) + 273.15 # K
tamanho_tempo = len(rotacao_intervalada)
t = np.linspace(0,tamanho_tempo, tamanho_tempo)                     # s

# Cálculo das velocidades (média e instantânea)
vel_med = 2 * rotacao_real * CURSO
vel_inst = velocidade_instantanea(r=R_MANIVELA,
                                  l=L_BIELA,
                                  omega=rotacao_original * 2*np.pi,
                                  t=np.linspace(0,
                                                len(rotacao_original),
                                                len(rotacao_original)))

vel_inst_torque_max = velocidade_instantanea(r=R_MANIVELA,
                                            l=L_BIELA,
                                            omega=ROTACAO_TORQUE_MAXIMO,
                                            t=t)

vel_inst_potencia_max = velocidade_instantanea(r=R_MANIVELA,
                                              l=L_BIELA,
                                              omega=ROTACAO_POTENCIA_MAXIMA,
                                              t=t)

vel_max_torque_max = max(vel_inst_torque_max)
vel_max_potencia_max = max(vel_inst_potencia_max)


# Cálculo da carga lateral
cl, bmep = carga_lateral(rotacao=rotacao_intervalada,
                         cilindrada=CILINDRADA,
                         diametro=D_CILINDRO,
                         potencia=potencia,
                         RL=RL,
                         return_bmep=True)

# Calculo do indice de Mach (torque maximo)
RHO_AR_TORQUE_MAXIMO = rho(P=P_ATM,
                           T=temp_amb[torque.argmax()],
                           R=287)

vth_torque_maximo = velocidade_isentropica(
    P0=P_ATM,
    deltaP=abs(p_col_adm[torque.argmax()]-P_ATM),
    rho0=RHO_AR_TORQUE_MAXIMO
)

# Calculo do indice de Mach (potencia maxima)
RHO_AR_POTENCIA_MAXIMA = rho(P=P_ATM,
                             T=temp_amb[potencia.argmax()],
                             R=287)

vth_potencia_maxima = velocidade_isentropica(
    P0=P_ATM,
    deltaP=abs(p_col_adm[potencia.argmax()]-P_ATM),
    rho0=RHO_AR_POTENCIA_MAXIMA
)


mach_potencia_max = mach(vth_potencia_maxima, V_SOM)
mach_torque_max = mach(vth_torque_maximo, V_SOM)

ag = area_garganta(d_haste=7 * 1e-3,
                   d_valvula=37 * 1e-3,
                   n=6)


mach_potencia_max_f = mach_fergunson(b=D_CILINDRO,
                                     Up=vel_med[potencia.argmax()],
                                     Af=ag,
                                     ci=V_SOM)


mach_torque_max_f = mach_fergunson(b=D_CILINDRO,
                                   Up=vel_med[torque.argmax()],
                                   Af=ag,
                                   ci=V_SOM)



plot_pressao_torque_potencia_carga_lateral(
    rotacao=unumpy.nominal_values(rotacao_real),
    erro_rotacao=unumpy.std_devs(rotacao_real),
    carga_lateral=unumpy.nominal_values(cl),
    erro_carga_lateral=unumpy.std_devs(cl),
    potencia=unumpy.nominal_values(potencia),
    erro_potencia=unumpy.std_devs(potencia),
    torque=unumpy.nominal_values(torque),
    erro_torque=unumpy.std_devs(torque),
    pressao=unumpy.nominal_values(bmep),
    erro_pressao=unumpy.std_devs(bmep)
)


plot_velocidades(
    rotacao=unumpy.nominal_values(rotacao_real),
    erro_rotacao=unumpy.std_devs(rotacao_real),
    rotacao_original=rotacao_original,
    erro_rotacao_original=None,
    t=t,
    velocidade_media=unumpy.nominal_values(vel_med),
    erro_velocidade_media=unumpy.std_devs(vel_med),
    velocidade_instantanea=unumpy.nominal_values(vel_inst),
    erro_velocidade_instantanea=unumpy.std_devs(vel_inst)
)

pressao_boost(rotacao=unumpy.nominal_values(rotacao_real),
              erro_rotacao=unumpy.std_devs(rotacao_real),
              pressao_boost=unumpy.nominal_values(p_col_adm),
              erro_pressao_boost=unumpy.std_devs(p_col_adm),
              bmep=unumpy.nominal_values(bmep),
              erro_bmep=unumpy.std_devs(bmep))

print(f'Velocidade máxima (torque máximo): {vel_max_torque_max:.2f} m/s')
print(f'Velocidade máxima (potência máxima): {vel_max_potencia_max:.2f} m/s')
print(f'Mach (torque máximo): {mach_torque_max:.2f}')
print(f'Mach (potência máxima): {mach_potencia_max:.1f}')
print(f'Mach F (torque máximo): {mach_torque_max_f:.4f}')
print(f'Mach F (potência máxima): {mach_potencia_max_f:.5f}')
