#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:02:39 2022

@author: Mohammad Afzal Shadab
Colliander et al, GRL, 2022 data from 08/09/2016
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'font.family': 'Serif'}) 
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import sys
sys.path.insert(1, '../Solver')

#Parameters
LWC_top = 0.03 #From the data liquid water content
krw0    = 1.0  #Endpoint relative permeability, Brooks-Corey model
phi     = 0.72  #Average porosity in top 1m
Delta_rho = 999.9-1.2754 #Difference in water - gas density, kg/m^3, IUPAC
g       = 9.808 #Acceleration due to gravity, m/s^2
day2s   = 24*60*60 #Conversion from day to seconds
m       = 3     #Absolute permeability power law coefficient, Meyer and Hewitt 2017, k=k0 * porosity^m
mu      = 1e-3  #Viscosity of water, Pa.s


#Importing results from Colliander et. al, 2022
shock     = np.genfromtxt('../Solver/Colliander_data/Shock.csv', delimiter=',')
rare_front= np.genfromtxt('../Solver/Colliander_data/Rarefaction_front.csv', delimiter=',')
rare_end  = np.genfromtxt('../Solver/Colliander_data/Rarefaction_end.csv', delimiter=',')

#Analysis
##Shock
m, b = np. polyfit(shock[:-6,0], shock[:-6,1], 1)
t_arr= np.linspace(0,12,100) 
Shock_speed = m.copy() #Shocks speed, m/day
K_l     = LWC_top*Shock_speed #Meltwater_flux, m/day
k0      = 5.6e-11   #Absolute permeability, Meyer and Hewitt 2017, k=k0 * porosity^m
n_shock = np.log((K_l/day2s)*mu/(k0*krw0*Delta_rho*g*phi))/np.log((LWC_top)/phi)

##Rarefaction
m_rare1, b_rare1 = np. polyfit(rare_front[2:5,0], rare_front[2:5,1], 1)
Rarefaction_speed_front = m_rare1.copy()
m_rare2, b_rare2 = np. polyfit(rare_end[:3,0], rare_end[:3,1], 1)
Rarefaction_speed_back = m_rare2.copy()


def LWC_calc(LWC_res):
    res = (LWC_res * Rarefaction_speed_back/day2s)*mu/(n_shock * k0*krw0*Delta_rho*g*phi**m*(LWC_res/phi)**n_shock) - 1
    return res

LWC_res = scipy.optimize.fsolve(LWC_calc,0.0001)


Rarefaction_speed_front1 = (n_shock * k0*krw0*Delta_rho*g*phi**m*(LWC_top/phi)**n_shock)/((LWC_top / day2s)*mu)

print(f' For {phi} porosity, \n  \
      The infiltration rate is {K_l} m/day. \n \
      The evaluated speed of shock is {Shock_speed} m/day. \n \
      The evaluated Brooks-Corey coefficient n is {n_shock}. \n \
      The modeled speed of the fastest rarefaction characteristic is be {Rarefaction_speed_front1} m/day. \n \
      The estimated speed of the slowest rarefaction characteristic may be {Rarefaction_speed_back} m/day \n \
      The estimated residual LWC may be {LWC_res[0]}.')

#Plotting
plt.figure(figsize=(10,10),dpi=100)
plt.plot(shock[:-4,0],shock[:-4,1],'ro',label='Shock, Colliander et al. (2022)',markersize=10)
plt.plot(rare_end[0:5,0],rare_end[0:5,1],'k^',label='Rarefaction end, Colliander et al. (2022)',markersize=10)
plt.xlabel(r'time [days]')
plt.ylabel(r'depth [m]')
plt.title(f"Assumed porosity is {phi}")
plt.plot(t_arr,m*t_arr+b,'k--',label=r"Shock (fit), $\Lambda_\mathcal{S}$= %0.3f m/day, n= %0.3f" %(Shock_speed,n_shock),markersize=10)
plt.plot(t_arr,Rarefaction_speed_front1 *t_arr + b_rare2*Rarefaction_speed_front1/m_rare2,'b-.',label=r'Rarefaction front (model), $\lambda_2$=%0.3f m/day'%Rarefaction_speed_front1,markersize=10)
plt.plot(t_arr,m_rare2*t_arr+b_rare2,'k-.',label=r'Rarefaction back (fit), res. LWC=%0.4f'%LWC_res,markersize=10)
plt.legend(loc='lower right')
plt.ylim([2,0])
plt.tight_layout()
plt.savefig(f'Colliander_analysis{phi}.pdf',bbox_inches='tight', dpi = 600)


def spun_up_profile_May2016_Colliander():
    data     = np.genfromtxt('../Solver/Colliander_data/density_profile_greenland_DYE2_python.csv', delimiter=',')
    depth    = data[:,0]  #Depth in m
    depth[0] = 0.1
    temp     = data[:,1]  #Temperature in deg C
    rho_firn = data[:,2]  #Density of firn including ice kg/m^3
    rho_bulk = data[:,3]  #Density including ice kg/m^3
    ice      = data[:,4]  #Ice in m
    porosity = 1-rho_bulk/917
    
    #plotting
    fig, (ax1, ax2) = plt.subplots(1,2, sharey=True,figsize=(12,10))
    plt.subplot(1,2,1)
    plt.plot(porosity,depth,'kX',label=r'Colliander$^+$ (2022)',markersize=10)
    
    fit_porosity = interp1d(depth,porosity, kind='cubic', fill_value='extrapolate')
    print(fit_porosity)
    fit_depth    = np.linspace(depth[0],depth[-1],1000)   
    plt.plot(fit_porosity(fit_depth),fit_depth,'k--',label='Fit',markersize=10)
           
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'depth [m]')
    plt.ylim([20,0])
    plt.legend(loc='right')
    plt.tight_layout()

    plt.subplot(1,2,2)
    plt.plot(temp,depth,'ro',label=r'Colliander$^+$ (2022)',markersize=10)
    fit_temp     = interp1d(depth[0:],temp[0:], kind='cubic', fill_value='extrapolate')
    fit_depth    = np.linspace(depth[0],depth[-1],1000)   
    plt.plot(fit_temp(fit_depth),fit_depth,'k--',label='Fit',markersize=10)
    plt.xlabel(r'Temperature $[\circ C]$')
    plt.ylabel(r'depth [m]')
    plt.ylim([20,0])
    plt.legend(loc='right')
    plt.tight_layout()
    plt.savefig(f'Colliander_analysis_porosity_Temp.pdf',bbox_inches='tight', dpi = 600)
    return fit_temp,fit_porosity

# Function to calculate the power-law with constants a and b


def spun_up_profile_May2016_Colliander_powerlaw():
    data     = np.genfromtxt('../Solver/Colliander_data/density_profile_greenland_DYE2_python.csv', delimiter=',')
    depth    = data[:,0]  #Depth in m
    depth[0] = 0.1
    temp     = data[:,1]  #Temperature in deg C
    rho_firn = data[:,2]  #Density of firn including ice kg/m^3
    rho_bulk = data[:,3]  #Density including ice kg/m^3
    ice      = data[:,4]  #Ice in m
    porosity = 1-rho_bulk/917
    
    phi0 = porosity[-1]
    phi1 = porosity[0] - porosity[-1]
    H    = depth[-1]
    
    def power_law(x,m):
        return phi0 + phi1*np.power(1-x/H, m)
    
    # Fit the dummy power-law data
    pars, cov = curve_fit(f=power_law, xdata=depth, ydata=porosity, p0=[1], bounds=(0,13))
    m = pars[0]
    #plotting
    fig, (ax1) = plt.subplots(1,1, sharey=True,figsize=(7,10))
    plt.subplot(1,1,1)
    
    fit_depth    = np.linspace(depth[0],depth[-1],1000)   
    plt.plot(power_law(fit_depth,m),fit_depth,'k-',label='Fit',markersize=10)
           
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'depth [m]')
    plt.ylim([20,0])
    plt.title(r'$\phi(y) = %.2f+%.2f*\left(\frac{y}{%.0f} \right)^{%.2f}$'%(phi0,phi1,H,m),pad=0.1)
    plt.plot(power_law(0,m)*np.ones_like(fit_depth),fit_depth,'r--',label=r'$\phi_{top}$')
    plt.plot(power_law(20,m)*np.ones_like(fit_depth),fit_depth,'b--',label=r'$\phi_{bottom}$')
    plt.plot(porosity,depth,'rX',label=r'Colliander$^+$ (2022)',markersize=10)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'Colliander_analysis_porosity_power_law.pdf',bbox_inches='tight', dpi = 600)
    return phi0,phi1,m

def Qnet_May2Sept2016_Samira():
    data     = np.genfromtxt('../Solver/Colliander_data/Q_net.csv', delimiter=',')
    day      = data[:,0]  #Time in days
    day      = data[:,0]-data[0,0]
    Qnet     = data[:,1]  #Flux in W/m^2
    Qnet_func= interp1d(day,Qnet, kind='linear', fill_value='extrapolate')
    
    
    #Plotting
    plt.figure(figsize=(10,10),dpi=100)
    plt.plot(day,Qnet,'ro',label='Data',markersize=10)
    fit_time = np.linspace(0,day[-1],1000)
    plt.plot(fit_time,Qnet_func(fit_time),'k-',label='Fit',markersize=10)
    plt.xlabel(r'time [days]')
    plt.ylabel(r'Q$_{net}$ [W/m$^2$]')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'Samira_flux.pdf',bbox_inches='tight', dpi = 600)
    return day,Qnet,Qnet_func



'''
lightgrey = [220/255,220/255,220/255]
#For beautiful plots
data     = np.genfromtxt('../Solver/Colliander_data/density_profile_greenland_DYE2_python.csv', delimiter=',')
depth    = data[:,0]  #Depth in m
depth[0] = 0.1
temp     = data[:,1]  #Temperature in deg C
rho_firn = data[:,2]  #Density of firn including ice kg/m^3
rho_bulk = data[:,3]  #Density including ice kg/m^3
ice      = data[:,4]  #Ice in m
porosity = 1-rho_bulk/917

phi0 = porosity[-1]
phi1 = porosity[0] - porosity[-1]
H    = depth[-1]

def power_law(x,m):
    return phi0 + phi1*np.power(1-x/H, m)

# Fit the dummy power-law data
pars, cov = curve_fit(f=power_law, xdata=depth, ydata=porosity, p0=[1], bounds=(0,13))
m = pars[0]
#plotting
fig, (ax1) = plt.subplots(1,1, sharey=True,figsize=(7,10))
plt.subplot(1,1,1)

fit_depth    = np.linspace(depth[0],depth[-1],1000)   
plt.plot(power_law(fit_depth,m),10-fit_depth,'k-',label='Fit',markersize=10,color=lightgrey)
       
plt.xlabel(r'$\phi$')
plt.ylabel(r'h [m]')
plt.ylim([0,10])
plt.plot(power_law(0,m)*np.ones_like(fit_depth),10-fit_depth,'r--',label=r'$\phi_{top}$',color=lightgrey)
plt.plot(power_law(20,m)*np.ones_like(fit_depth),10-fit_depth,'b--',label=r'$\phi_{bottom}$')
plt.tight_layout()
plt.savefig(f'Colliander_analysis_porosity_power_law_bot.pdf',bbox_inches='tight', dpi = 600)


#plotting
fig, (ax1) = plt.subplots(1,1, sharey=True,figsize=(7,10))
plt.subplot(1,1,1)

fit_depth    = np.linspace(depth[0],depth[-1],1000)   
plt.plot(power_law(fit_depth,m),10-fit_depth,'k-',label='Fit',markersize=10,color=lightgrey)
       
plt.xlabel(r'$\phi$')
plt.ylabel(r'h [m]')
plt.ylim([0,10])
plt.plot(power_law(20,m)*np.ones_like(fit_depth),10-fit_depth,'b--',label=r'$\phi_{bottom}$',color=lightgrey)
plt.plot(power_law(0,m)*np.ones_like(fit_depth),10-fit_depth,'r--',label=r'$\phi_{top}$')
plt.tight_layout()
plt.savefig(f'Colliander_analysis_porosity_power_law_top.pdf',bbox_inches='tight', dpi = 600)

#plotting
fig, (ax1) = plt.subplots(1,1, sharey=True,figsize=(7,10))
plt.subplot(1,1,1)

fit_depth    = np.linspace(depth[0],depth[-1],1000)   
       
plt.xlabel(r'$\phi$')
plt.ylabel(r'h [m]')
plt.ylim([0,10])
plt.plot(power_law(0,m)*np.ones_like(fit_depth),10-fit_depth,'r--',label=r'$\phi_{top}$',color=lightgrey)
plt.plot(power_law(20,m)*np.ones_like(fit_depth),10-fit_depth,'b--',label=r'$\phi_{bottom}$',color=lightgrey)
plt.plot(power_law(fit_depth,m),10-fit_depth,'k-',label='Fit',markersize=10)
plt.tight_layout()
plt.savefig(f'Colliander_analysis_porosity_power_law_power_law.pdf',bbox_inches='tight', dpi = 600)


'''
