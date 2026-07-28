# Continental Geotherm 2D transient validation using enthalpy formulation
# author: Mohammad Afzal Shadab
# date: 02/12/2021

import sys
sys.path.insert(1, '../Solver')

# ignore warning 
import warnings
warnings.filterwarnings("ignore")

# clear all 
from IPython import get_ipython
get_ipython().magic('reset -sf') #for clearing everything
get_ipython().run_line_magic('matplotlib', 'qt') #for plotting in separate window

# import python libraries
import numpy as np
import scipy.sparse as sp
import scipy.special as sp_special
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib.ticker import LinearLocator, FormatStrFormatter
plt.rcParams.update({'font.size': 22})


# import personal libraries
import build_gridfun2D
import build_opsfun2D_optimized as build_ops
from comp_algebraicmean_optimized import comp_algebraic_mean
from comp_harmonicmean import comp_harmonicmean
from comp_fluxfunade import comp_flux_ade
from mobilityfun import mobility
from build_bndfun_optimized import build_bnd
from solve_lbvpfun_optimized import solve_lbvp
from comp_fluxfun import comp_flux
from flux_upwindfun2D_optimized import flux_upwind  #instead of from adv_opfun2D import adv_opfun
from frac_flow_derivativefun import frac_flow_derivative
from time import perf_counter
from eval_phase_behavior import eval_phase_behaviorCwH, enthalpyfromT,eval_phase_behaviorCwH_diagram

tprog_start = perf_counter()

#simulation parameters
N_H = 5000
N_C = 5000
simulation_name = f'plotting_H_CvsT_phiw_dTdH'

#problem parameters
k0 = 2.1       # thermal conductivity of ice [W/(m K)]
T_cutoff = 100+273.16    #cutoff temperature for the contour plots [K]
T_min =   100+273.16 # average annual surface temperature [K]
T_max =  -100+273.16 # average annual surface temperature [K]

rho_w = 1000    # density of water [kg/m^3]
cp_w  = 4186    # specific heat of water at constant pressure [J/(kg K)]
k_w   = 0.606   # coefficient of thermal conductivity of water [W / (m K)]
rho_g = 1.225   # density of gas [kg/m^3]
cp_g  = 1003.5  # specific heat of gas at constant pressure [J/(kg K)]
rho_i = 917     # average density of ice cap [kg/m^3]
cp_i  = 2106.1  # specific heat of ice at constant pressure [J/(kg K)]
k_i   = 2.25    # coefficient of thermal conductivity of ice [W / (m K)]
kk    = k_i/(rho_i*cp_i) # thermal diffusivity of ice
Tm    = 273.16  # melting point temperature of ice [K]
L_fusion= 333.55e3# latent heat of fusion of water [J / kg]



H_max = enthalpyfromT(np.array([[T_max]]),Tm,rho_i,rho_w,rho_g,cp_i,cp_w,cp_g,np.array([[0.0]]),np.array([[0.0]]),L_fusion)[0][0] #phi_w is not useful
H_min = enthalpyfromT(np.array([[T_min]]),Tm,rho_i,rho_w,rho_g,cp_i,cp_w,cp_g,np.array([[0.0]]),np.array([[0.0]]),L_fusion)[0][0] #phi_w is not useful

H = np.linspace(H_min,H_max,N_H)
C = np.linspace(0,rho_w*1.0,N_C)

Cmesh,Hmesh = np.meshgrid(C,H)

H_col  = np.reshape(np.transpose(Hmesh), (N_H*N_C,-1))    #building the single H vector
C_col  = np.reshape(np.transpose(Cmesh), (N_H*N_C,-1))    #building the single C vector

phi_w,phi_i,T,dTdH = eval_phase_behaviorCwH_diagram(H_col,Tm,rho_i,rho_w,rho_g,cp_i,cp_w,cp_g,C_col,L_fusion)

phi_w[T > T_cutoff] = np.nan

phi_i[np.absolute(T) > T_cutoff] = np.nan
dTdH[np.absolute(T) > T_cutoff] = np.nan
T[np.absolute(T) > T_cutoff] = np.nan
T[T<173] = np.nan
phi_w[T<173] = np.nan
phi_i[T<173] = np.nan
dTdH[T<173] = np.nan


norm = colors.Normalize(vmin=0, vmax=1)
fig = plt.figure(figsize=(10,8), dpi=100)
plt.contour(Cmesh,Hmesh,np.transpose(phi_w.reshape(N_H,N_C)), levels=10, linewidths=0.5, colors='k')
ax1= plt.contourf(Cmesh,Hmesh,np.transpose(phi_w.reshape(N_H,N_C)),100,cmap=mpl.cm.coolwarm,linewidths=2.0)
clb = plt.colorbar(cmap=mpl.cm.coolwarm, norm=norm)
plt.ylim([np.min(H)/2,3.35e8])
clb.set_label(r'$\phi_w$', labelpad=-40, y=1.06, rotation=0)
plt.ylabel(r'Enthalpy, $H \quad[J/m^3]$')
plt.xlabel(r'Composition, $C \quad[kg/m^3]$')
plt.tight_layout()
plt.savefig(f'HCvsphi_w.pdf',bbox_inches='tight', dpi = 600)
plt.savefig(f'HCvsphi_w.png',bbox_inches='tight', dpi = 600)


fig = plt.figure(figsize=(10,8), dpi=100)
plt.contour(Cmesh,Hmesh,np.transpose((1-phi_i-phi_w).reshape(N_H,N_C)), levels=10, linewidths=0.5, colors='k')
ax1= plt.contourf(Cmesh,Hmesh,np.transpose((1-phi_i-phi_w).reshape(N_H,N_C)),100,cmap=mpl.cm.coolwarm,linewidths=2.0)
clb = plt.colorbar(cmap=mpl.cm.coolwarm, norm=norm)
plt.ylim([np.min(H)/2,3.35e8])
clb.set_label(r'$\phi_g$', labelpad=-40, y=1.06, rotation=0)
plt.ylabel(r'Enthalpy, $H \quad[J/m^3]$')
plt.xlabel(r'Composition, $C \quad[kg/m^3]$')
plt.tight_layout()
plt.savefig(f'HCvsphi_g.pdf',bbox_inches='tight', dpi = 600)
plt.savefig(f'HCvsphi_g.png',bbox_inches='tight', dpi = 600)

fig = plt.figure(figsize=(10,8), dpi=100)
plt.contour(Cmesh,Hmesh,np.transpose(phi_i.reshape(N_H,N_C)), levels=10, linewidths=0.5, colors='k')
ax1= plt.contourf(Cmesh,Hmesh,np.transpose(phi_i.reshape(N_H,N_C)),100,cmap=mpl.cm.coolwarm,linewidths=2.0)
clb = plt.colorbar(cmap=mpl.cm.coolwarm, norm=norm)
plt.ylim([np.min(H)/2,3.35e8])
clb.set_label(r'$\phi_i$', labelpad=-40, y=1.06, rotation=0)
plt.ylabel(r'Enthalpy, $H \quad[J/m^3]$')
plt.xlabel(r'Composition, $C \quad[kg/m^3]$')
plt.tight_layout()
plt.savefig(f'HCvsphi_i.pdf',bbox_inches='tight', dpi = 600)
plt.savefig(f'HCvsphi_i.png',bbox_inches='tight', dpi = 600)


norm = colors.Normalize(vmin=0, vmax=1)
fig = plt.figure(figsize=(10,8), dpi=100)
plt.contour(Cmesh,Hmesh,np.transpose(phi_w.reshape(N_H,N_C)), levels=10, linewidths=0.5, colors='k')
ax1= plt.contourf(Cmesh,Hmesh,np.transpose(phi_w.reshape(N_H,N_C)),100,cmap='Blues',linewidths=2.0)
clb = plt.colorbar(cmap='Blues', norm=norm)
plt.ylim([np.min(H)/2,3.35e8])
clb.set_label(r'$\phi_w$', labelpad=-40, y=1.06, rotation=0)
plt.ylabel(r'Enthalpy, $H \quad[J/m^3]$')
plt.xlabel(r'Composition, $C \quad[kg/m^3]$')
plt.tight_layout()
plt.savefig(f'HCvsphi_w_new.pdf',bbox_inches='tight', dpi = 600)
plt.savefig(f'HCvsphi_w_new.png',bbox_inches='tight', dpi = 600)


fig = plt.figure(figsize=(10,8), dpi=100)
plt.contour(Cmesh,Hmesh,np.transpose((1-phi_i-phi_w).reshape(N_H,N_C)), levels=10, linewidths=0.5, colors='k')
ax1= plt.contourf(Cmesh,Hmesh,np.transpose((1-phi_i-phi_w).reshape(N_H,N_C)),100,cmap='Greens',linewidths=2.0)
clb = plt.colorbar(cmap='Blues', norm=norm)
plt.ylim([np.min(H)/2,3.35e8])
clb.set_label(r'$\phi_g$', labelpad=-40, y=1.06, rotation=0)
plt.ylabel(r'Enthalpy, $H \quad[J/m^3]$')
plt.xlabel(r'Composition, $C \quad[kg/m^3]$')
plt.tight_layout()
plt.savefig(f'HCvsphi_g_new.pdf',bbox_inches='tight', dpi = 600)
plt.savefig(f'HCvsphi_g_new.png',bbox_inches='tight', dpi = 600)

fig = plt.figure(figsize=(10,8), dpi=100)
plt.contour(Cmesh,Hmesh,np.transpose(phi_i.reshape(N_H,N_C)), levels=10, linewidths=0.5, colors='k')
ax1= plt.contourf(Cmesh,Hmesh,np.transpose(phi_i.reshape(N_H,N_C)),100,cmap='Greys',linewidths=2.0)
clb = plt.colorbar(cmap='Blues', norm=norm)
plt.ylim([np.min(H)/2,3.35e8])
clb.set_label(r'$\phi_i$', labelpad=-40, y=1.06, rotation=0)
plt.ylabel(r'Enthalpy, $H \quad[J/m^3]$')
plt.xlabel(r'Composition, $C \quad[kg/m^3]$')
plt.tight_layout()
plt.savefig(f'HCvsphi_i_new.pdf',bbox_inches='tight', dpi = 600)
plt.savefig(f'HCvsphi_i_new.png',bbox_inches='tight', dpi = 600)

fig = plt.figure(figsize=(10,8), dpi=100)
norm = colors.Normalize(vmin=-100, vmax=100)
plt.contour(Cmesh,Hmesh,np.transpose(T.reshape(N_H,N_C))-Tm, levels=10, linewidths=0.5, colors='k')
ax1= plt.contourf(Cmesh,Hmesh,np.transpose(T.reshape(N_H,N_C))-Tm,100,cmap=mpl.cm.coolwarm ,linewidths=2.0)
clb =plt.colorbar(cmap=mpl.cm.coolwarm, norm=norm)
plt.ylim([np.min(H)/2,3.35e8])
clb.set_label(r'$T [^oC]$', labelpad=-40, y=1.06, rotation=0)
plt.ylabel(r'Enthalpy, $H \quad[J/m^3]$')
plt.xlabel(r'Composition, $C \quad[kg/m^3]$')
plt.tight_layout()
plt.savefig(f'HCvsT.pdf',bbox_inches='tight', dpi = 600)
plt.savefig(f'HCvsT.png',bbox_inches='tight', dpi = 600)

fig = plt.figure(figsize=(10,8), dpi=100)
plt.contour(Cmesh,Hmesh,np.transpose(dTdH.reshape(N_H,N_C)), levels=10, linewidths=0.5, colors='k')
ax1= plt.contourf(Cmesh,Hmesh,np.transpose(dTdH.reshape(N_H,N_C)),100,cmap=mpl.cm.coolwarm,linewidths=2.0)
clb = plt.colorbar(cmap=mpl.cm.coolwarm, norm=norm)
plt.ylim([np.min(H),3.35e8])
clb.set_label(r'$dT/dH $', labelpad=-40, y=1.06, rotation=0)
plt.ylabel(r'Enthalpy, $H \quad[J/m^3]$')
plt.xlabel(r'Composition, $C \quad[kg/m^3]$')
plt.tight_layout()
plt.savefig(f'HCvsdTdH.pdf',bbox_inches='tight', dpi = 600)

'''
#Using the volume fraction of the gas
H = np.linspace(H_min,H_max,N_H)
phi_g = np.linspace(0,1.0,N_C)

phi_gmesh,Hmesh = np.meshgrid(C,H)

H_col  = np.reshape(np.transpose(Hmesh), (N_H*N_C,-1))    #building the single H vector
phi_g_col  = np.reshape(np.transpose(phi_gmesh), (N_H*N_C,-1))    #building the single C vector

phi_w,phi_i,T,dTdH = eval_phase_behavior(H_col,Tm,rho_i,rho_w,rho_g,cp_i,cp_w,cp_g,phi_g_col*rho_g,L_fusion)

phi_w[np.absolute(T) > 1000] = np.nan
phi_i[np.absolute(T) > 1000] = np.nan
T[np.absolute(T) > 1000] = np.nan
dTdH[np.absolute(T) > 1000] = np.nan
T[T<0] = np.nan
phi_w[T<0] = np.nan
phi_i[T<0] = np.nan
dTdH[T<0] = np.nan
phi_w[phi_g_col > 1.0] = np.nan
phi_i[phi_g_col > 1.0] = np.nan
T[phi_g_col > 1.0] = np.nan
dTdH[phi_g_col > 1.0] = np.nan


norm = colors.Normalize(vmin=0, vmax=1)
fig = plt.figure(figsize=(15,7.5), dpi=300)
plt.contour(phi_gmesh,Hmesh,np.transpose(phi_w.reshape(N_H,N_C)), levels=14, linewidths=0.5, colors='k')
ax1= plt.contourf(phi_gmesh,Hmesh,np.transpose(phi_w.reshape(N_H,N_C)),100,cmap=mpl.cm.coolwarm,linewidths=2.0)
clb = plt.colorbar(cmap=mpl.cm.coolwarm, norm=norm)
plt.ylim([np.min(H),np.max(H)])
plt.title(r'$\phi_w$')
plt.ylabel(r'$Enthalpy \quad [J/m^3]$')
plt.xlabel(r'$\phi_g$')
plt.xlim([0,1])
plt.savefig(f'Hphigvsphi_w.pdf',bbox_inches='tight')

fig = plt.figure(figsize=(15,7.5), dpi=100)
plt.contour(phi_gmesh,Hmesh,np.transpose(phi_i.reshape(N_H,N_C)), levels=14, linewidths=0.5, colors='k')
ax1= plt.contourf(phi_gmesh,Hmesh,np.transpose(phi_i.reshape(N_H,N_C)),100,cmap=mpl.cm.coolwarm,linewidths=2.0)
clb = plt.colorbar(cmap=mpl.cm.coolwarm, norm=norm)
plt.ylim([np.min(H),np.max(H)])
plt.title(r'$\phi_i$')
plt.ylabel(r'$Enthalpy \quad [J/m^3]$')
plt.xlabel(r'$\phi_g$')
plt.xlim([0,1])
plt.savefig(f'Hphigvsphi_i.pdf',bbox_inches='tight')

fig = plt.figure(figsize=(15,7.5) , dpi=100)
plt.contour(phi_gmesh,Hmesh,np.transpose(T.reshape(N_H,N_C)), levels=14, linewidths=0.5, colors='k')
ax1= plt.contourf(phi_gmesh,Hmesh,np.transpose(T.reshape(N_H,N_C)),100,cmap=mpl.cm.coolwarm ,linewidths=2.0)
plt.colorbar(cmap=mpl.cm.coolwarm, norm=norm)
plt.ylim([np.min(H),np.max(H)])
plt.title(r'$T [K]$')
plt.ylabel(r'$Enthalpy \quad [J/m^3]$')
plt.xlabel(r'$\phi_g$')
plt.xlim([0,1])
plt.savefig(f'HphigvsT.pdf',bbox_inches='tight')

fig = plt.figure(figsize=(15,7.5) , dpi=100)
plt.contour(phi_gmesh,Hmesh,np.transpose(dTdH.reshape(N_H,N_C)), levels=14, linewidths=0.5, colors='k')
ax1= plt.contourf(phi_gmesh,Hmesh,np.transpose(dTdH.reshape(N_H,N_C)),100,cmap=mpl.cm.coolwarm,linewidths=2.0)
plt.colorbar(cmap=mpl.cm.coolwarm, norm=norm)
plt.ylim([np.min(H),np.max(H)])
plt.title(r'$dT/dH $')
plt.ylabel(r'$Enthalpy \quad [J/m^3]$')
plt.xlabel(r'$\phi_g$')
plt.xlim([0,1])
plt.savefig(f'HphigvsdTdH.pdf',bbox_inches='tight')


'''