######################################################################
#Figure 4 - Dye 2, 2016
#Mohammad Afzal Shadab
#Date modified: 05/03/2022
######################################################################

######################################################################
#import libraries
######################################################################

import sys
sys.path.insert(1, '../Solver')

# import personal libraries and class
from classes import *    

from two_components_aux import *
from solve_lbvpfun_SPD import solve_lbvp_SPD
from complex_domain import find_faces, find_bnd_cells, find_all_faces, find_all_x_faces,find_all_y_faces
from comp_fluxfun import comp_flux
from scipy.integrate import solve_ivp
from comp_sat_unsat_bnd_flux_fun import comp_sat_unsat_bnd_flux, find_top_bot_cells, find_left_right_cells
from comp_face_coords_fun import comp_face_coords
from find_plate_dofs import find_plate_dofs,multiple_plate_dofs
from comp_mean_matrix import comp_mean_matrix
from eval_phase_behavior import eval_phase_behaviorCwH
from random_correlated_fields import GeneratePermField,GeneratePermField2D,GeneratePermField2D_papers, GeneratePermField2D_new

#Samira Site A actual init temp but RetMIP porosity
from RetMIP_data_analysis import RetMIP_fit_porosity, Samira_data_actual_fit_temp, RetMIP_acc_rate
fit_temp     = Samira_data_actual_fit_temp
fit_porosity = RetMIP_fit_porosity
from colliander_data_analysis import spun_up_profile_May2016_Colliander, Qnet_May2Sept2016_Samira
day,Qnet,Qnet_func             = Qnet_May2Sept2016_Samira() #Samira

from scipy import integrate

#for 2D
from build_gridfun2D import build_grid
from spin_up_firn import spin_up_firn

#Non-dimensional permeability: Harmonic mean
def f_Cm(phi,m):
    fC = np.zeros_like(phi)        
    fC = phi**m / phi_L**m          #Power law porosity
    return fC

#Rel perm of water: Upwinded
def f_Cn(C,phi,n):
    fC = np.zeros_like(phi)
    fC = ((C/phi-s_wr)/(1-s_gr-s_wr))**n    #Power law rel perm
    fC[C<=0]  = 0.0      
    return fC

def find_highest_in_arr1_for_unique_in_arr2(arr1, arr2):
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    
    # Step 1: Identify unique values in arr2
    unique_arr2 = np.unique(arr2)
    
    # Step 2: For each unique value in arr2, find the corresponding values in arr1
    result = []
    for val in unique_arr2:
        # Find indices in arr2 where the value is found
        indices = np.where(arr2 == val)[0]
        # Get corresponding values in arr1 and take the maximum
        max_value = np.max(arr1[indices])
        result.append(max_value)
    
    return np.array(result)


def find_nearest_numbers(arr1, arr2):
    """
    Finds the nearest numbers in arr1 for each value in arr2.

    Args:
        arr1 (numpy.ndarray): The array to search within.
        arr2 (numpy.ndarray): The array containing the values to find the nearest numbers for.

    Returns:
        numpy.ndarray: An array of the nearest numbers from arr1 for each value in arr2.
    """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    
    indices = np.argmin(np.abs(arr1[:, np.newaxis] - arr2), axis=0)
    return arr1[indices]

def find_indices(array1, array2):
  """
  Finds the indices of elements in array1 that are also present in array2.

  Args:
    array1: The first array (NumPy array).
    array2: The second array (NumPy array).

  Returns:
    A NumPy array containing the indices of elements in array1 that are in array2.
    Returns an empty array if no elements are found.
  """
  indices = np.where(np.isin(array1, array2))[0]
  return indices

def remove_lower_from_consecutive(arr):
    """
    Finds consecutive numbers in a NumPy array and removes all but the highest value in each sequence.

    Args:
        arr (np.ndarray): The input NumPy array.

    Returns:
        np.ndarray: A new array with lower values removed from consecutive sequences.
    """
    if arr.size == 0:
        return arr
    
    diffs = np.diff(arr)
    split_indices = np.where(diffs != 1)[0] + 1
    consecutive_groups = np.split(arr, split_indices)
    
    result = []
    for group in consecutive_groups:
        result.append(np.max(group))
        
    return np.array(result)


######################################################################
#parameters
######################################################################
##simulation
simulation_name = f'samimi_without_acc_new_0deg_slope'
diffusion = 'yes'
corr_rnd_field = 'yes' #yes or no
normalized = False    #normalize the porosity field 
simulation_name = f'samimi_without_acc_new_0deg_slope'+f'n_ormalized{normalized}'
CFL    = 0.1#0.5     #CFL number
tilt_angle = 0#60   #angle of the slope

rho_i_closeoff = 917#830 #closeoff porosity [kg/m3] 830 for later

enable_outflow = True #enable outflow

##hydrology
m      = 3 #Cozeny-Karman coefficient for numerator K = K0 (1-phi_i)^m
n      = 2 #Corey-Brooks coefficient krw = krw0 * sw^n
s_wr   = 0.0   #Residual water saturation
s_gr   = 0.0   #Residual gas saturation
k_w0   = 1.0  #relative permeability threshold: wetting phase
rho_w  = 1000.0  #density of non-wetting phase
mu_w   = 1e-3 #dynamic viscosity: wetting phase    
grav   = 9.81    #acceleration due to gravity
k0     = 5.6e-11#absolute permeability m^2 in pore space Meyer and Hewitt 2017

[C_L,phi_L] = [0.03,fit_porosity(0.0)] #water volume fraction, porosity at the surface
npp    = 4   #number of positive days [days]

##thermodynamics
rho_w = 1000    # density of water [kg/m^3]
cp_w  = 4186    # specific heat of water at constant pressure [J/(kg K)]
k_w   = 0.606   # coefficient of thermal conductivity of water [W / (m K)]
rho_nw = 1.225  # density of gas [kg/m^3]
phi_nw_init = 0.5    # volumetric ratio of gas, decreasing exponentially
cp_nw  = 1003.5  # specific heat of gas at constant pressure [J/(kg K)]
rho_i = 917     # average density of ice cap [kg/m^3]
cp_i  = 2106.1  # specific heat of ice at constant pressure [J/(kg K)]
k_i   = 2.25    # coefficient of thermal conductivity of ice [W / (m K)]
kk    = k_i/(rho_i*cp_i) # thermal diffusivity of ice
Tm    = 273.16  # melting point temperature of ice [K]
L_fusion= 333.55e3# latent heat of fusion of water [J / kg]

phi_i_threshold = rho_i_closeoff/rho_i #cutoff porosity

#domain details
z0 = 3.75 #characteristic height (m)

fc = k0*k_w0*rho_w*grav/mu_w*phi_L**3 #Infiltration capacity (m/s)

sat_threshold = 1-1e-2 #threshold for saturated region formation

#injection
Param.xleft_inj= 0e3;  Param.xright_inj= 2e3

#temporal
tf     = 120*day2s #0.5
tmax = tf#0.07#0.0621#2 #5.7#6.98  #time scaling with respect to fc
#t_interest = [0,0.25,0.5399999999999999 + 0.005,0.6,0.7116505061468059,1.0] #swr,sgr=0.05
t_interest = np.linspace(0,tmax,int(tf/day2s*24)+1)   #swr,sgr=0

Nt   = 1000
dt = tmax / (Nt)

#Non-dimensional permeability: Harmonic mean
def f_Cm(phi,m):
    fC = np.zeros_like(phi)
    fC = fc* phi**m / phi_L**m           #Power law porosity
    return fC

#Rel perm of water: Upwinded
def f_Cn(C,phi,n):
    fC = np.zeros_like(phi)
    fC = ((C/phi-s_wr)/(1-s_gr-s_wr))**n    #Power law rel perm
    fC[C<=0]  = 0.0      
    return fC

######################################################################
#Defining grid and operators
######################################################################
Grid.xmin =  0*z0; Grid.xmax =2e3; Grid.Nx =50; #50   #Horizontal direction
Grid.ymin =  0*z0; Grid.ymax =5;  Grid.Ny = 200#200  #Vertically downward direction
Grid = build_grid(Grid)   #building grid
[D,G,I] = build_ops(Grid) #building divergence, gradient and identity operators
D  = -np.transpose(G)
Avg     = comp_mean_matrix(Grid)  #building mean operator

[Xc,Yc] = np.meshgrid(Grid.xc,Grid.yc)                 #building the (x,y) matrix
Xc_col  = np.reshape(np.transpose(Xc), (Grid.N,-1))    #building the single X vector
Yc_col  = np.reshape(np.transpose(Yc), (Grid.N,-1))    #building the single Y vector

s_wfunc    = lambda phi_w,phi_nw: phi_w  / (phi_w + phi_nw)
s_nwfunc   = lambda phi_w,phi_nw: phi_nw / (phi_w + phi_nw)
T_annual_func_sigmoid = lambda Tbot, Ttop, Yc_col, Y0: Tbot + (Ttop - Tbot)/Y0*(Yc_col) 


#Initial conditions
phi_nw  = fit_porosity(Yc_col) #volume fraction of gas 


if corr_rnd_field == 'yes':
    ############################################
    
    #Permeability field
    corr_length = 1; K_mean = 0 ; amplitude = 0.05;s = 25061977 #in m
    factor_x = 4e3; factor_y = 1;
    
    [K,Xc,Yc] = GeneratePermField2D_new(Grid,factor_x*corr_length,factor_y*corr_length,amplitude,K_mean,'exp',s)
    
    K = 10**(-K)
    
    if normalized==True:
        K = np.reshape(np.transpose(K), (Grid.N,-1))/np.max(K)
    else:
        K = np.reshape(np.transpose(K), (Grid.N,-1))
    
    phi_normalized = K**(1/m) #Normalize porosity with respect to maximum value in correlated random field
    
    #Plot porosity
    phi_array = np.transpose(phi_normalized.reshape(Grid.Nx,Grid.Ny))

    vmin = 0.90
    vmax = 1.10
    
    # clip to avoid empty colorbar regions
    phi_array_plot = np.clip(phi_array, vmin, vmax)
    
    fig = plt.figure(figsize=(20, 6), dpi=200)
    
    levels = np.linspace(vmin, vmax, 100)
    
    plot = plt.contourf(
        Xc, Yc, phi_array_plot,
        cmap="Greys_r",
        levels=levels,
        vmin=vmin,
        vmax=vmax
    )
    
    plt.xlabel(r'$x$ [m]')
    plt.ylabel(r'$z$ [m]')
    plt.xlim([Grid.xmin, Grid.xmax])
    plt.ylim([3, Grid.ymin])
    
    clb = plt.colorbar(plot)
    clb.set_ticks([0.9, 0.95, 1.0, 1.05, 1.1])
    clb.set_label(r'$\phi_{corr}(x,z)$', labelpad=1, y=1.1, rotation=0)
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    plt.savefig(f'../Figures/correlated_fields_{Grid.Nx}by{Grid.Ny}_Marcs-logK_fcorr_x{factor_x}m_amp{amplitude}_norm.pdf',bbox_inches='tight', dpi = 200)
    plt.savefig(f'../Figures/correlated_fields_{Grid.Nx}by{Grid.Ny}_Marcs-logK_fcorr_x{factor_x}m_amp{amplitude}_norm.png',bbox_inches='tight', dpi = 600)
        

    
    phi_nw  = phi_normalized*fit_porosity(Yc_col) #volume fraction of gas  ###########
    phi_nw_arr = (phi_nw.reshape(Grid.Nx,Grid.Ny)) #np.flipud(phi_nw.reshape(Grid.Nx,Grid.Ny))
    phi_nw  = np.reshape(phi_nw_arr, (Grid.N,-1))
    
    
    #Plot porosity
    phi_array = np.transpose(phi_nw.reshape(Grid.Nx,Grid.Ny))

    fig = plt.figure(figsize=(20,6),dpi=200)
    plot = [plt.contourf(Xc, Yc, phi_array,cmap="coolwarm",levels=100)]
    plt.xlabel(r'$x$ [m]')
    plt.ylabel(r'$z$ [m]')
    plt.xlim([Grid.xmin, Grid.xmax])
    plt.ylim([3,Grid.ymin])
    #plt.axis('scaled')
    #mm = plt.cm.ScalarMappable(cmap=cm.coolwarm)
    #mm.set_array(phi)
    clb = plt.colorbar(cmap='coolwarm')
    clb.set_label(r'$\phi $', labelpad=1, y=1.075, rotation=0)
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    plt.savefig(f'../Figures/correlated_fields_{Grid.Nx}by{Grid.Ny}_Marcs-logK_corr{corr_length}.pdf',bbox_inches='tight', dpi = 200)
    
    
    simulation_name = simulation_name+f'corr{corr_rnd_field}_lx{factor_x}m_ly{factor_y}m_amp{amplitude}'
    
    ###################################################################################


phi_w   = np.zeros_like(phi_nw)  #volume fraction of water phase #No water
C       = rho_w * phi_w + (1 - phi_w - phi_nw) * rho_i
T       = fit_temp(Yc_col)+Tm
T[Yc_col>3.7] = -15.7 + Tm
H       = enthalpyfromT(fit_temp(Yc_col)+Tm,Tm,rho_i,rho_w,0,cp_i,cp_w,0,phi_w,phi_nw,L_fusion)

phi_w,phi_i,T,dTdH = eval_phase_behaviorCwH(H,Tm,rho_i,rho_w,rho_nw,cp_i,cp_w,cp_nw,C,L_fusion)
s_w_init= s_wfunc(phi_w,phi_nw)
phi = (phi_w+ phi_nw)*np.ones((Grid.N,1))#np.exp(-(1-Yc_col/(grid.ymax-grid.ymin)))#phi*np.ones((grid.N,1)) #porosity in each cell
s_w = s_w_init.copy()#s_wp *np.ones((grid.N,1))
fs_theta = 0.0*np.ones((Grid.N,1))                     #RHS of heat equation

######################################################################
#initializing arrays
######################################################################
s_w_sol = np.copy(s_w) #for water saturation
C_sol   = np.copy(C)   #for composition
H_sol   = np.copy(H)   #for enthalpy
T_sol   = np.copy(T)   #for Temperature
phi_w_sol =np.copy(phi_w) #for wetting phase volume fraction
phi_i_sol=np.copy(phi_i) #for non wetting phase volume fraction
q_w_new_sol = np.zeros((Grid.Nf,1)) #placeholder for wetting face Darcy flux
q_nw_new_sol= np.zeros((Grid.Nf,1)) #placeholder for non-wetting face Darcy flux
phi_w_sol = phi_w.copy()


#injection
dof_inj   = Grid.dof_ymin[  np.intersect1d(np.argwhere(Grid.xc>= Param.xleft_inj),np.argwhere(Grid.xc <= Param.xright_inj))]
dof_f_inj = Grid.dof_f_ymin[np.intersect1d(np.argwhere(Grid.xc>= Param.xleft_inj),np.argwhere(Grid.xc <= Param.xright_inj))]

######################################################################
#Boundary conditions
######################################################################
#boundary condition for saturation equation
BC.dof_dir   = np.array([])
BC.dof_f_dir = np.array([])
BC.dof_neu   = np.array([])
BC.dof_f_neu = np.array([])
BC.qb = np.array([])
BC.C_g    = np.array([])
[B,N,fn]  = build_bnd(BC, Grid, I)
# Enthalpy equation (total)

dof_fixedH = np.setdiff1d(Grid.dof_ymin,dof_inj)
dof_f_fixedH = np.setdiff1d(Grid.dof_f_ymin,dof_f_inj)
#Param.H.dof_dir = np.hstack([dof_fixedH,Grid.dof_ymax,Grid.dof_xmin[1:-1],Grid.dof_xmax[1:-1]])
#Param.H.dof_f_dir = np.hstack([dof_f_fixedH ,Grid.dof_f_ymax,Grid.dof_f_xmin[1:-1],Grid.dof_f_xmax[1:-1]])
#Param.H.g  = np.hstack([H[dof_fixedH-1,0],H[Grid.dof_ymax-1,0],H[Grid.dof_xmin[1:-1]-1,0],H[Grid.dof_xmax[1:-1]-1,0]])#np.hstack([0*np.ones_like(Grid.dof_ymin),LWC_top*rho_w*L_fusion*np.ones_like(Grid.dof_ymin)])


Param.H.dof_dir = np.array([])
Param.H.dof_f_dir = np.array([])
Param.H.g  = np.array([])#np.hstack([0*np.ones_like(Grid.dof_ymin),LWC_top*rho_w*L_fusion*np.ones_like(Grid.dof_ymin)])

Param.H.dof_neu = dof_inj
Param.H.dof_f_neu = dof_f_inj
Param.H.qb = np.zeros((len(dof_inj),1))
[H_B,H_N,H_fn] = build_bnd(Param.H,Grid,I)

dof_ytop = Grid.dof_f_ymin

t    =[0.0]
time = 0
v = np.ones((Grid.Nf,1))

G_original = G.copy() 
D_original = D.copy()  

i = 0

length_added_final = 0

######################################################################
#Time loop starts
######################################################################
while time<tmax:
    surf_loc = 0
    Param.H.qb= Qnet_func(time/day2s)*np.ones((len(dof_inj),1)) #1000*np.ones((len(dof_inj),1)) #
    [H_B,H_N,H_fn] = build_bnd(Param.H,Grid,I)

    #BC.C_g    = C[dof_inj-1] 
    #Param.H.qb= Qnet_func(time/day2s)*np.ones((Grid.Nx,1))

    #Param.H.qb= Qnet_func(time/day2s)*np.transpose([np.concatenate([np.flip(np.linspace(1,1.1,int(Grid.Nx/2))),np.linspace(1,1.1,int(Grid.Nx/2))])])
    G = G_original.copy()
    D = D_original.copy()
    
    # #####################
    # #Ice layer affecting fluid and heat advection
    # #####################
    # if np.any(phi_i>=phi_i_threshold):
    #     dof_ice_layer  = Grid.dof[phi_i[:,0] >= phi_i_threshold]  #
    #     dof_f_bnd_ice_layer = find_faces(dof_ice_layer,D,Grid) 
    #     #Step 1: Modify the gradient to import natural BC at ice layer boundary
    #     DD  = np.transpose(zero_rows(np.transpose(D),dof_f_bnd_ice_layer-1))
    #     GG  = zero_rows(G,dof_f_plate_bnd-1)
    #####################        
    
    ################################################################################################
    non_porous_vol_frac = 1e-2
    if np.any(phi_i<=non_porous_vol_frac):
        dof_complete_melt  = Grid.dof[phi_i[:,0] < non_porous_vol_frac]
        dof_partial_melt   = np.setdiff1d(Grid.dof,dof_complete_melt) #saturated cells 
        dof_f_bnd_melt     = find_faces(dof_complete_melt,D,Grid) 
        dof_f_bnd_melt     = dof_f_bnd_melt[dof_f_bnd_melt>Grid.Nfx]  #location of the actual surface
        ytop,ybot          = find_top_bot_cells(dof_f_bnd_melt,D,Grid)
        #T[dof_complete_melt-1]  = Tm##############trial        
        #phi_w[dof_complete_melt-1]  = 0##############trial        

        '''
        ####Old
        Param.H.dof_neu = ytop
        Param.H.dof_f_neu = dof_f_bnd_melt
        Param.H.qb = Qnet_func(time/day2s)*np.ones((np.shape(Param.H.dof_neu)[0],1))#np.ones((Grid.Nx,1)) ####changed this to cells with melting
        
        Param.H.dof_neu = ytop
        Param.H.dof_f_neu = dof_f_bnd_melt
        Param.H.qb = Qnet_func(time/day2s)*np.ones((Grid.Nx,1)) ####changed this to cells with melting        
        '''
        ####New  
        ytop_new      = Grid.dof_ymin.copy()
        ytop_face_new = Grid.dof_f_ymin.copy()
        
        ytop = ytop[dof_f_bnd_melt>Grid.Nfx] #location of cell above the surface
        
        replace_bnd_face = np.floor((dof_f_bnd_melt-Grid.Nfx-1)/(Grid.Ny+1))*(Grid.Ny+1)+Grid.Nfx+1 #boundary faces to be replaced in BCs
        replace_bnd_cell = np.floor((ytop-1)/Grid.Ny)*Grid.Ny+1             #boundary cells to be replaced in BCs
        

        ytop_face_new[find_indices(ytop_face_new,np.unique(replace_bnd_face))] = find_highest_in_arr1_for_unique_in_arr2(dof_f_bnd_melt,replace_bnd_face)                                            
        ytop_new[find_indices(ytop_new,np.unique(replace_bnd_cell))] = find_highest_in_arr1_for_unique_in_arr2(ytop,replace_bnd_cell)
               
        #concatenate ytop and dof_f_ymin
        
        Param.H.dof_neu  = ytop_new
        Param.H.dof_f_neu= ytop_face_new
        #Param.H.qb = Qnet_func(time/day2s)*np.ones((Grid.Nx,1)) ####changed this to cells with melting               
        ######

        
        [H_B,H_N,H_fn] = build_bnd(Param.H,Grid,I)
        
        G = zero_rows(G,dof_f_bnd_melt-1)
        D = -np.transpose(G)
        D  = zero_rows(D,dof_complete_melt-1) #
    
        ############################################################
        #location of the surface
        surf_loc = Yc_col[ytop_new-1] - Grid.dy/2
    ################################################################################################     

    phi_w_old = phi_w.copy() 
    C_old     = C.copy()  
    H_old     = H.copy()      
    flux      = (comp_harmonicmean(Avg,f_Cm(phi,m))*(flux_upwind(v, Grid) @ f_Cn(phi_w_old,phi,n)))*np.cos(tilt_angle*np.pi/180)
    flux_vert = flux.copy()
    flux_vert[Grid.dof_f<=Grid.Nfx,0] = flux_vert[Grid.dof_f<=Grid.Nfx,0]*np.tan(tilt_angle*np.pi/180)  #making gravity based flux in x direction
    
    if i<=0:
        speed = np.min(f_Cm(1-phi_i,m)*f_Cn(np.array([C_L]),np.array([phi_L]),n)[0]/C_L)
    else:
        speed = f_Cm(phi,m)*f_Cn(phi_w,np.array(phi),n)/(phi_w)  
        speed[np.isnan(speed)] = 0
    
    if np.any(speed[speed>0]):
        dt1   = CFL*Grid.dy/np.max(speed[speed>0]) #Calculating the time step from the filling of volume
    else:
        dt1   = 1e16 #Calculating the time step from the filling of volume

    #flux_vert_mat = sp.dia_matrix((flux_vert[:,0],  np.array([0])), shape=(Grid.Nf, Grid.Nf))
    res = D@flux_vert  #since the gradients are only zero and 1    
    res_vert = res.copy()
    ######
    #Taking out the domain to cut off single phase region
    #dof_act  = Grid.dof[C_old[:,0] / (phi[:,0]*(1-s_gr)) < 0.99999]
    dof_act  = Grid.dof[phi_w_old[:,0] / (phi[:,0]*(1-s_gr)) < sat_threshold]
    
    if np.any(phi_i>=phi_i_threshold):
        dof_ice_layer  = Grid.dof[phi_i[:,0] >= phi_i_threshold]  #find cells in ice layer
        dof_act  = np.unique(np.concatenate([dof_act,dof_ice_layer])) #ignore the cells in the ice layer
    
    dof_inact= np.setdiff1d(Grid.dof,dof_act) #saturated cells
    if len(dof_act)< Grid.N: #when atleast one layer is present
        #############################################
        dof_f_saturated = find_faces(dof_inact,D,Grid)       
        #Step 1: Modify the gradient to import natural BC at the crater
        #G_small = zero_rows(G,dof_f_saturated-1)
        if enable_outflow == True:
            Param.dof_neu   = np.hstack([Grid.dof_xmin, Grid.dof_xmax, Grid.dof_ymin, Grid.dof_ymax])
            dof_act = np.concatenate([dof_act,np.intersect1d(dof_inact,Param.dof_neu)]) #setting outflow
            
        #Step 2 Eliminate inactive cells by putting them to constraint matrix
        Param.P.dof_dir = (dof_act)           
        Param.P.dof_f_dir= np.array([])
        Param.P.g       =  -Yc_col[dof_act-1]*np.cos(tilt_angle*np.pi/180) \
                           -Xc_col[dof_act-1]*np.sin(tilt_angle*np.pi/180)  
        Param.P.dof_neu = np.array([])
        Param.P.dof_f_neu = np.array([])
        Param.P.qb = np.array([])

        [B_P,N_P,fn_P] = build_bnd(Param.P,Grid,I)
        Kd  = comp_harmonicmean(Avg,f_Cm(phi,m)) * (Avg @ f_Cn(phi_w_old,phi,n))
        
        
        Kd  = sp.dia_matrix((Kd[:,0],  np.array([0])), shape=(Grid.Nf, Grid.Nf))
        L = - D @ Kd @ G
        u = solve_lbvp(L,fn_P,B_P,Param.P.g,N_P)   # Non dimensional water potential

        q_w = - Kd @ G @ u
        
        if enable_outflow == True:
            ####### New boundary condition for outflow
           #=============================================================================
            Param.dof_neu   = np.hstack([Grid.dof_xmin, Grid.dof_xmax, Grid.dof_ymin, Grid.dof_ymax])
            Param.dof_f_neu = np.hstack([Grid.dof_f_xmin, Grid.dof_f_xmax, Grid.dof_f_ymin, Grid.dof_f_ymax])        
            Param.dof_dir   = np.array([]) 
            Param.dof_f_dir = np.array([]) 
            fs = np.zeros((Grid.N,1))
            q_w = comp_flux(D, Kd, G, u, fs, Grid, Param)
            #=============================================================================
            
        
        
        #upwinding boundary y-directional flux
        if tilt_angle != 90:
            #finding boundary faces
            dof_ysat_faces = dof_f_saturated[dof_f_saturated>=Grid.Nfx]
            
            #removing boundary faces
            dof_ysat_faces = np.setdiff1d(dof_ysat_faces,np.append(Grid.dof_f_ymin,Grid.dof_f_ymax))
            
            ytop,ybot               = find_top_bot_cells(dof_ysat_faces,D,Grid)
            q_w[dof_ysat_faces-1,0] = comp_sat_unsat_bnd_flux(q_w[dof_ysat_faces-1],flux_vert[dof_ysat_faces-1],ytop,ybot,phi_w,phi,sat_threshold)
            
        #upwinding boundary x-directional flux   ####new line
        if tilt_angle != 0:
            #finding boundary faces
            dof_xsat_faces = dof_f_saturated[dof_f_saturated<Grid.Nfx]
            
            #removing boundary faces
            dof_xsat_faces = np.setdiff1d(dof_xsat_faces,np.append(Grid.dof_f_xmin,Grid.dof_f_xmax))
            
            xleft,xright            = find_left_right_cells(dof_xsat_faces,D,Grid)
            q_w[dof_xsat_faces-1,0] = comp_sat_unsat_bnd_flux(q_w[dof_xsat_faces-1],flux_vert[dof_xsat_faces-1],xright,xleft,phi_w,phi,sat_threshold)
             
        #find all saturated faces
        dof_sat_faces = find_all_faces(dof_inact,D,Grid)  
        
        flux_vert[dof_sat_faces-1] = q_w[dof_sat_faces-1]

        res = D @ flux_vert
        
    #####################
    #Ice layer affecting fluid and heat advection
    #####################
    if np.any(phi_i>=phi_i_threshold):
        dof_ice_layer  = Grid.dof[phi_i[:,0] >= phi_i_threshold]  #
        dof_f_bnd_ice_layer = find_faces(dof_ice_layer,D,Grid) 
        #Step 1: Modify the gradient to import natural BC at ice layer boundary
        DD  = np.transpose(zero_rows(np.transpose(D),dof_f_bnd_ice_layer-1))
        res = DD @ flux_vert
    #####################            
        
    #dt2   = np.abs((phi*sat_threshold - phi*s_gr - phi_w_old)/(res)) #Calculating the time step from the filling of volume
    #dt2  =  CFL*np.min(dt2[dt2>1e-4*z0/fc])

    ###################################
    dt2   = np.abs((phi*sat_threshold - phi*s_gr - phi_w_old)/(res)) #Calculating the time step from the filling of volume
    dt2   = np.abs(dt2[~np.isinf(dt2)]); dt2   = np.abs(dt2[~np.isnan(dt2)])
    if ~np.any(dt2):
    #dt2  =  CFL*np.min(dt2[dt2>1e-6*z0/fc])
        dt2  =  1e16
    else:
        #dt2   = np.min(dt2[dt2>0])
        dt2   = np.min(dt2[dt2>1e-3])
    ###################################
    #Time step of diffusion
    dt3  = np.min(CFL*(Grid.dy**2)/(2*(phi_w*k_w + phi_i*k_i)/(rho_i*phi_i*cp_i+rho_w*phi_w*cp_w))) 
    if np.isnan(dt3): dt3=1e10
    
    # if time/day2s >70 and time/day2s < 75:
    #     CFL = 0.001
    # else:
    #     CFL = 0.05#0.5
        
    dt = np.min([dt1,dt2,dt3])

    #if i<10: 
    #    dt = tmax/(Nt*10)
    if time+dt >= t_interest[np.max(np.argwhere(time+dt >= t_interest))] and time < t_interest[np.max(np.argwhere(time+dt >= t_interest))]:
        dt = t_interest[np.max(np.argwhere(time+dt >= t_interest))] - time   #To have the results at a specific time

    #Explicit Enthalpy update
    phi_w,phi_i,T,dTdH = eval_phase_behaviorCwH(H,Tm,rho_i,rho_w,0,cp_i,cp_w,0,C,L_fusion)

    #Enthalpy
    #calculating K_bar using K_bar = phi_w*k_w + phi_i*k_i
    #K_bar = phi_w=*k_w + phi_i*k_i
    K_bar = 2.22362*(rho_i/rho_w*(phi_i))**1.885 + phi_w*k_w
    K_bar_edge = sp.dia_matrix((np.array(np.transpose(comp_harmonicmean(Avg,K_bar))),  np.array([0])), shape=(Grid.Nf, Grid.Nf))     #average permeability at each interface
    

 
    ##Update
    #Volume fraction
    #RHS = C_old - dt*D@flux  #since the gradients are only zero and 1    
    RHS = C_old - dt*res*rho_w + dt* fn #since the gradients are only zero and 1  

    #saturation equation boundary condition
    BC.dof_dir   = np.array([])
    BC.dof_f_dir = np.array([])
    BC.dof_neu   = np.array([])
    BC.dof_f_neu = np.array([])
    BC.qb        = np.array([])
    BC.C_g       = np.array([])
    [B,N,fn]     = build_bnd(BC, Grid, I)

    #####################
    #Ice layer affecting fluid and heat advection
    #####################
    if np.any(phi_i>=phi_i_threshold):
        BC.dof_dir   = dof_ice_layer
        BC.dof_f_dir = 0*dof_ice_layer  #placeholder
        BC.dof_neu   = np.array([])
        BC.dof_f_neu = np.array([])
        BC.qb = np.array([])
        BC.C_g    = C_old[dof_ice_layer-1]
        [B,N,fn]  = build_bnd(BC, Grid, I)
    
    C   = solve_lbvp(I,RHS,B,BC.C_g,N)
    
    
    # #####################
    # #Ice layer affecting fluid and heat advection
    # #####################
    # if np.any(phi_i>=phi_i_threshold):
    #     C[dof_ice_layer-1]  = C_old[dof_ice_layer-1]

    h_i,h_w,h_nw = eval_h(Tm,T,rho_i,rho_w,0,cp_i,cp_w,0,L_fusion)

    #####################
    #Ice layer affecting fluid and heat advection
    #####################
    if np.any(phi_i>=phi_i_threshold):    #changing the divergence in the heat advection term since water cannot move past ice layer 
        if diffusion == 'yes':
            RHS    =  H - dt * (DD @ (rho_w * flux_vert * comp_harmonicmean(Avg,h_w))  - D @ K_bar_edge @ G @ T   -(fs_theta+H_fn)) # -kappa_edge @ G @ H 
        else:
            RHS    =  H - dt * (DD @ (rho_w * flux_vert * comp_harmonicmean(Avg,h_w)) -(fs_theta+H_fn)) # -kappa_edge @ G @ H 
    else:
        if diffusion == 'yes':
            RHS    =  H - dt * (D @ (rho_w * flux_vert * comp_harmonicmean(Avg,h_w)  -K_bar_edge @ G @ T  ) -(fs_theta+H_fn)) # -kappa_edge @ G @ H 
        else:
            RHS    =  H - dt * (D @ (rho_w * flux_vert * comp_harmonicmean(Avg,h_w)) -(fs_theta+H_fn)) # -kappa_edge @ G @ H 


    H      =  solve_lbvp(I,RHS,H_B,Param.H.g,H_N)


    phi_w,phi_i,T,dTdH = eval_phase_behaviorCwH(H,Tm,rho_i,rho_w,rho_nw,cp_i,cp_w,cp_nw,C,L_fusion)
    phi    = 1-phi_i
    phi_nw = 1 -phi_i -phi_w
    time = time + dt   
    
    if np.any(np.isnan(C)) or np.any(np.isnan(H)):
        print('Nan in C and H')
        break
        
    
    #adding accumulation (new very localized with no cell change)
    if (RetMIP_acc_rate(time/day2s) > 0) and np.any((surf_loc > 0)):  #if accumulation > 0 and surface is below top
        #length_added = RetMIP_acc_rate(time/day2s) * dt * rho_w /315 #'meter' ice equivalent of solid ice accumulation
        #length_added = RetMIP_acc_rate(time/day2s) * dt * rho_w /315 #'meter' ice equivalent of solid ice accumulation
        length_added = integrate.qmc_quad(RetMIP_acc_rate, (time-dt)/day2s, (time)/day2s)[0] * day2s * rho_w /315 #'meter' ice equivalent of solid ice accumulation #dt is involved
        
        
        if length_added_final < Grid.dy:  #insufficient to fill one cell
            length_added_final = length_added_final + length_added 
                
        else: #sufficient to fill one cell atleast
            if  length_added_final == 0:   length_added_final =  length_added 

            cells_added = int(np.floor(length_added_final/Grid.dy))  #numbers of cells to be added
            Grid_dof_ymin = Grid.dof_ymin[ytop_new>Grid.dof_ymin]    #only consider the numerical grid where surface is lower
            ytop_new = ytop_new[ytop_new>Grid.dof_ymin] #only when surface is lower than numerical grid ########Trial
            dof_fresh = np.array([])
            for cells_new in range(1,cells_added+1):  
                potential_dof = np.int64(ytop_new - cells_new)  #use ytop
                ###################################surface dof is less than numerical           #numerical grid is above surface
                final_dof     = np.concatenate([Grid_dof_ymin[Grid_dof_ymin > potential_dof],potential_dof[Grid_dof_ymin <= potential_dof]]) #if accumulation goes above numerical grid, stick to the numerical grid 
                dof_fresh     = np.int64(np.concatenate([dof_fresh, final_dof])) #cells of freshly fallen snow
                print(dof_fresh)
            dof_fresh = np.unique(dof_fresh)
            #break
            
            '''
            surf_loc = surf_loc - length_added_final     #new surface location [m]        
            dof_fresh = dof_complete_melt[np.ravel(Yc_col[dof_complete_melt-1] > surf_loc)] #cells of freshly fallen snow
            '''
            
            if np.any(dof_fresh): print(dof_fresh)
            H[dof_fresh-1] = 0.0   #adding temperate ice on top
            C[dof_fresh-1] = 315   #adding temperate, freshly fallen snow on top [kg/m3]
            phi_i[dof_fresh-1]   = 315/rho_i
            phi_w[dof_fresh-1]   = 0 
            phi_nw[dof_fresh-1]  = 1 - 315/rho_i
            print(time,'\t',length_added_final)
            length_added_final = 0  #re-setting the length to zero for next cycle
    ###################################################################################
    
    # #adding accumulation
    # if (RetMIP_acc_rate(time/day2s) > 0) and np.any((surf_loc > 0)):  #if accumulation > 0 and surface is below top
    #     #length_added = RetMIP_acc_rate(time/day2s) * dt * rho_w /315 #'meter' ice equivalent of solid ice accumulation
    #     #length_added = RetMIP_acc_rate(time/day2s) * dt * rho_w /315 #'meter' ice equivalent of solid ice accumulation
    #     length_added = integrate.qmc_quad(RetMIP_acc_rate, (time-dt)/day2s, (time)/day2s)[0] * day2s * rho_w /315 #'meter' ice equivalent of solid ice accumulation #dt is involved
        
    #     if length_added_final < Grid.dy:  #insufficient to fill one cell
    #         length_added_final = length_added_final + length_added 
                
    #     else: #sufficient to fill one cell atleast
    #         if  length_added_final == 0:   length_added_final =  length_added 
    
    #         cells_added = int(np.floor(length_added_final/Grid.dy))  #numbers of cells to be added
    #         dof_fresh = np.array([])
    #         for cells_new in range(1,cells_added+1):  
    #             potential_dof = np.int64(ytop_new - cells_new)
    #             ###################################surface dof is less than numerical           #numerical grid is above surface
    #             final_dof     = np.concatenate([Grid.dof_ymin[Grid.dof_ymin > potential_dof],potential_dof[Grid.dof_ymin <= potential_dof]]) #if accumulation goes above numerical grid, stick to the numerical grid 
    #             dof_fresh     = np.int64(np.concatenate([dof_fresh, final_dof])) #cells of freshly fallen snow
    #             #print(dof_fresh)
    #         dof_fresh = np.unique(dof_fresh)
    #         '''
    #         surf_loc = surf_loc - length_added_final     #new surface location [m]        
    #         dof_fresh = dof_complete_melt[np.ravel(Yc_col[dof_complete_melt-1] > surf_loc)] #cells of freshly fallen snow
    #         '''
            
    #         if np.any(dof_fresh): print(dof_fresh)
    #         H[dof_fresh-1] = 0.0   #adding temperate ice on top
    #         C[dof_fresh-1] = 315   #adding temperate, freshly fallen snow on top [kg/m3]
    #         phi_i[dof_fresh-1]   = 315/rho_i
    #         phi_w[dof_fresh-1]   = 0 
    #         phi_nw[dof_fresh-1]  = 1 - 315/rho_i
    #         print(time,'\t',length_added_final)
    #         length_added_final = 0  #re-setting the length to zero for next cycle
    # ###################################################################################


    


    if np.isin(time,t_interest):
        t.append(time) 
        C_sol = np.concatenate((C_sol,C), axis=1) 
        s_w_sol = np.concatenate((s_w_sol,s_w), axis=1) 
        H_sol   = np.concatenate((H_sol,H), axis=1) 
        T_sol   = np.concatenate((T_sol,T), axis=1) 
        phi_w_sol    = np.concatenate((phi_w_sol,phi_w), axis=1) 
        phi_i_sol   = np.concatenate((phi_i_sol,phi_i), axis=1) 
        q_w_new_sol  = np.concatenate((q_w_new_sol,flux_vert), axis=1) 
        
        if len(dof_act)< Grid.N:
            print(i,time/day2s,'Saturated cells',Grid.N-len(dof_act))      
        else:    
            print(i,time/day2s)
    i = i+1
    

t = np.array(t)
tday = t/day2s

######################################################################
#Saving the data
######################################################################
np.savez(f'{simulation_name}_C{C_L}_{Grid.Nx}by{Grid.Ny}_t{tmax}.npz', t=t,q_w_new_sol=q_w_new_sol,H_sol=H_sol,T_sol=T_sol,s_w_sol=s_w_sol,phi_w_sol =phi_w_sol,phi_i_sol =phi_i_sol,phi=phi,Xc=Xc,Yc=Yc,Xc_col=Xc_col,Yc_col=Yc_col,Grid_Nx=Grid.Nx,Grid_Ny=Grid.Ny,Grid_xc=Grid.xc,Grid_yc=Grid.yc,Grid_xf=Grid.xf,Grid_yf=Grid.yf)


np.savez(f'{simulation_name}_T_surface.npz', tday=tday,T_surf=T_sol[0,:])

'''
######################################################################
#for loading data
######################################################################
data = np.load('samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx4000.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz')
t=data['t']
phi_w_sol =data['phi_w_sol']
phi_i_sol =data['phi_i_sol']
H_sol =data['H_sol']
T_sol =data['T_sol']
s_w_sol =data['s_w_sol']

phi=data['phi']
Xc=data['Xc']
Yc=data['Yc']
Xc_col=data['Xc_col']
Yc_col=data['Yc_col']
Grid.Nx=data['Grid_Nx']
Grid.Ny=data['Grid_Ny']
Grid.xc=data['Grid_xc']
Grid.yc=data['Grid_yc']
Grid.xf=data['Grid_xf']
Grid.yf=data['Grid_yf']
[dummy,endstop] = np.shape(phi_w_sol)
#C[:,0] = phi_w_sol[:,-1]
phi_sol = 1- phi_i_sol
'''

######################################################################
#plotting
######################################################################
light_red  = [1.0,0.5,0.5]
light_blue = [0.5,0.5,1.0]
light_black= [0.5,0.5,0.5]

######################################################################
#combined 1D plot with upGPR with actual dates and Temperatures + LWC combined
from datetime import datetime 
from datetime import timedelta
import pandas as pd 
phi_sol = 1 - phi_i_sol
df = pd.read_excel('./Samira-data/TDRA_MaytoSept2016.xlsx',header=(1))


#red_array  = np.array([[255,0,0], [255,43,43], [255,85,85], [255,128,128] , [255,170,170], [255,213,213]])/255;
#black_array= np.array([[0,0,0], [43,43,43], [85,85,85], [128,128,128] , [170,170,170], [213,213,213]])/255; 
#blue_array = np.array([[0,0,255], [43,43,255], [85,85,255], [128,128,255] , [170,170,255], [213,213,255]])/255;

red_array  = np.array([[255,0,0], [255,106.5,106.5] , [255,106.5,106.5] , [255,106.5,106.5] , [255,106.5,106.5], [255,213,213]])/255;
black_array= np.array([[0,0,0], [106.5,106.5,106.5], [106.5,106.5,106.5], [106.5,106.5,106.5] , [106.5,106.5,106.5], [213,213,213]])/255; 
blue_array = np.array([[0,0,255], [106.5,106.5,255], [106.5,106.5,255], [106.5,106.5,255] , [106.5,106.5,255], [213,213,255]])/255;

data  = pd.read_csv('/Users/ms6985/Desktop/Research_new/meltwater-percolation/Infiltration/JPL/Samira-data/Achilig_upGPR_data.csv')
dates = data['date'].apply(lambda x: datetime.strptime(x, "'%d-%b-%Y %H:%M:%S.%f'")) 
samira_heilig_time_days =np.array([(dates[i] - dates[850]).total_seconds() for i in range(850,4200)])/day2s   #time in days May 24 through September 18, 2016
Samira_dates = df['Time']
Samira_dates_index = np.where((Samira_dates>=dates[850]) & (Samira_dates<=dates[len(dates)-1]))

# format the string in the given format : day/month/year 
# hours/minutes/seconds-micro seconds
#format_data = "'%d-%b-%Y %H:%M:%S.%f'"
#for i in data['date']:
#    print(datetime.strptime(i, format_data))

from matplotlib import rcParams
rcParams.update({'font.size': 22})
Grid.ymax = 3
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1, sharex=True,figsize=(15,15),dpi=50)
#Neww plot sidebysidesnapshot
phi_w_sol_backup = phi_w_sol.copy()
tday = t/day2s
depth_array=np.kron(np.ones(len(tday)),np.transpose([Grid.yc]))


######
#Remove 99% melted ice
phi_w_sol_backup[phi_sol>1-non_porous_vol_frac] = np.nan
phi_w_sol[phi_sol>1-non_porous_vol_frac] = np.nan
phi_i_sol[phi_sol>1- non_porous_vol_frac] = np.nan
T_sol[phi_sol>1-non_porous_vol_frac] = np.nan
T_sol[np.isnan(phi_i_sol)] = np.nan
phi_w_sol_backup[np.isnan(phi_i_sol)] = np.nan
phi_w_sol[np.isnan(phi_i_sol)] = np.nan

#####

t_array=np.kron(tday,np.ones((Grid.Ny,1)))
#Make actual time array
import pandas as pd
t_dates = ([dates[850] + timedelta(seconds = i) for i in t])
t_dates = np.array([[np.datetime64(i) for i in t_dates]])
t_dates = np.repeat(t_dates, Grid.Ny, axis=0)


phi_w_array = phi_w_sol_backup[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
phi_i_array = phi_i_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
T_array =         T_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
T_array[T_array>Tm] = Tm
T_array[T_array<Tm-22] = Tm-22

plt.subplot(5,1,1)
plt.plot(([dates[850] + timedelta(days = i) for i in day]),Qnet,'r-',label='Data',linewidth=2)
fit_time = np.linspace(tday[0],tday[-1],1000)
plt.ylim([1.1*np.min(Qnet_func(fit_time)),1.1*np.max(Qnet_func(fit_time))])
plt.ylabel(r'Q$_{net}$ [W/m$^2$]',color='red')
mm = plt.cm.ScalarMappable(cmap=cm.Greys)
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
plt.yticks([0,25,50,75])
import matplotlib.dates as mdates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
ax1.xaxis.grid(True, which='major', color='k', linestyle='--')
ax1.spines['left'].set_color('red')
ax1.tick_params(axis='y', colors='red')

ax1new = ax1.twinx()

color = 'tab:blue'
ax1new.spines['right'].set_color('blue')
ax1new.set_ylabel(r'$a$ [mm.w.e./day]', color='blue')  # we already handled the x-label with ax1
ax1new.plot(([dates[850] + timedelta(days = i) for i in day]), RetMIP_acc_rate(day)*1e3*day2s , color='blue')
ax1new.tick_params(axis='y', colors='blue')
plt.yticks([0,10,20])

plt.subplot(5,1,2)
plot = [plt.contourf(t_dates, depth_array, (1-phi_i_array),cmap="Greys",levels=100,ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Greys)
mm.set_array(np.linspace(1-np.max(phi_i_array[~np.isnan(phi_i_array)]),1-np.min(phi_i_array[~np.isnan(phi_i_array)]),10000))
#mm.set_array(1-phi_i_sol)
plt.ylabel(r'z [m]')
#plt.xlim([tday[0],tday[-1]])
plt.ylim([Grid.ymax,Grid.ymin])
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$\phi$', labelpad=-100, y=0.5, rotation=90)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
ax2.xaxis.grid(True, which='major', color='k', linestyle='--')

plt.subplot(5,1,4)
plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.1'])[Samira_dates_index[0]],'-',label='0.3 m',color = black_array[0,:])
#plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.2'])[Samira_dates_index[0]],'-',label='0.6 m',color = black_array[1,:])
plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.3'])[Samira_dates_index[0]],'-',label='0.9 m',color = black_array[2,:])
#plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.4'])[Samira_dates_index[0]],'-',label='1.4 m',color = black_array[3,:])
#plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.5'])[Samira_dates_index[0]],'-',label='1.8 m',color = black_array[4,:])
plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.6'])[Samira_dates_index[0]],'-',label='2.1 m',color = black_array[5,:])

Samira_siteA_depth_array = np.array([0.3, 0.6, 0.9, 1.4, 1.8, 2.1])
Samira_siteA_depth_array_iter = np.array([0.3, 0.9, 2.1])
alpha_array = np.array([1.0, 0.75,0.6,0.4,0.2,0.1])
for i in Samira_siteA_depth_array_iter:   
    print(i)
    index = np.where(Yc_col<i)[0][-1]
    plt.plot(t_dates[0,:],(T_sol[index,:]+T_sol[index+1,:])/2-273.16,'r-',color=red_array[np.argwhere(i==Samira_siteA_depth_array)][0,0])
#plt.legend(fontsize='medium', ncol=2)
plt.ylabel(r'T[$^\circ$C]')
ax4.xaxis.grid(True, which='major', color='k', linestyle='--')

phi_w_array = phi_w_sol_backup[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
phi_w_array[phi_w_array>0.06] = 0.06

mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(np.linspace(np.min(phi_w_array[~np.isnan(phi_w_array)]),np.max(phi_w_array[~np.isnan(phi_w_array)]),10000))
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$LWC$', labelpad=-100, y=0.5, rotation=90)


plt.subplot(5,1,3)
T_array[T_array >= Tm]  = np.nan
new1 = ax3.contourf(t_dates, depth_array, T_array-Tm,cmap="Reds",levels=100,ls=None)
mm = plt.cm.ScalarMappable(cmap=cm.Reds)
mm.set_array(np.linspace(np.min(T_array[~np.isnan(T_array)]-Tm),np.max(T_array[~np.isnan(T_array)]-Tm),10000))
#mm.set_array(T_sol-Tm)
plt.ylabel(r'z [m]')
plt.ylim([Grid.ymax,Grid.ymin])
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'T[$^\circ$C]', labelpad=-100, y=0.5, rotation=90)
#plt.clim(0.000000, 1.0000000)
ax3.xaxis.grid(True, which='major', color='k', linestyle='--')


T_array = T_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
T_array[T_array>Tm] = Tm
T_array[T_array<Tm-22] = Tm-22
#phi_w_array[phi_w_array>0.06] = 0.06
phi_w_array[T_array < Tm] = np.nan
new2 = ax3.contourf(t_dates, depth_array, phi_w_array,cmap="Blues",levels=100,ls=None)

plt.plot(dates[850:4200],data['height'][850] - data['height'][850:4200],'k-')

plt.subplot(5,1,5)

phi_w_array = phi_w_sol_backup[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
#phi_w_array[phi_w_array>0.06] = 0.06

data_LWC  = pd.read_csv('/Users/afzal-admin/Documents/Research/meltwater-percolation/Infiltration/JPL/Samira-data/samimi_LWC.csv')

data_LWC[np.isnan(data_LWC)] = -10

plt.plot(([dates[850] + timedelta(days = i-3.5) for i in data_LWC['0.3m']]),data_LWC['0.3m.1']-data_LWC['0.3m.1'][0],'ko',label='0.3 m',  color = black_array[0,:], markersize=4)
#plt.plot(([dates[850] + timedelta(days = i-3.5) for i in data_LWC['0.6m']]),data_LWC['0.6m.1']-data_LWC['0.6m.1'][0],'ks',label='0.6 m',color = black_array[1,:], markersize=3)
plt.plot(([dates[850] + timedelta(days = i-3.5) for i in data_LWC['0.9m']]),data_LWC['0.9m.1']-data_LWC['0.9m.1'][0],'kd',label='0.9 m',color = black_array[2,:], markersize=4)
#plt.plot(([dates[850] + timedelta(days = i-3.5) for i in data_LWC['1.4m']]),data_LWC['1.4m.1']-data_LWC['1.4m.1'][0],'k^',label='1.4 m',color = black_array[3,:], markersize=3)
#plt.plot(([dates[850] + timedelta(days = i-3.5) for i in data_LWC['1.8m']]),data_LWC['1.8m.1']-data_LWC['1.8m.1'][0],'k>',label='1.8 m',color = black_array[4,:], markersize=3)
plt.plot(([dates[850] + timedelta(days = i-3.5) for i in data_LWC['2.1m']]),data_LWC['2.1m.1']-data_LWC['2.1m.1'][0],'k<',label='2.1 m',color = black_array[5,:], markersize=4)

plt.ylabel(r'LWC')

Samira_siteA_depth_array = np.array([0.3, 0.6, 0.9, 1.4, 1.8, 2.1])
alpha_array = np.array([1.0, 0.75,0.6,0.4,0.2,0.1])
for i in [0,2,5]:   
    index = np.where(Yc_col<Samira_siteA_depth_array[i])[0][-1]
    print(index)
    print(i)
    plt.plot(t_dates[0,:],(phi_w_sol[index,:]+phi_w_sol[index+1,:])/2,'-',color = blue_array[i,:])

plt.ylim(0,0.0275)
plt.xlim()
#plt.legend( ncol=2)
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
ax5.xaxis.grid(True, which='major', color='k', linestyle='--')
plt.xlabel(r'Calendar date')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

#yyyymmdd format
chosen_dates = [20160529,20160623,20160718,20160812,20160906]
chosen_dates = [datetime.strptime(str(int(dateee)),'%Y%m%d') for dateee in chosen_dates]

plt.xticks(chosen_dates,rotation = 0) # Rotates X-Axis Ticks by 45-degrees

chosen_dates = [20160524,20160918]
chosen_dates = [datetime.strptime(str(int(dateee)),'%Y%m%d') for dateee in chosen_dates]

plt.xlim(chosen_dates)

plt.tight_layout(w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_rhow{rho_w}_LWC_super_combined_withQAcc_actual_dateswithT_TplusLWC.pdf',bbox_inches='tight', dpi = 50)


'''
from matplotlib.patches import Rectangle
phi_w_sol_backup = phi_w_sol.copy()
######################################################################
#Making video
######################################################################
blue = np.array([135, 206, 235])/255
#A fun combined  phi, LWC, T
from datetime import datetime 
from datetime import timedelta
import pandas as pd 
tday = t/day2s
ymax_loc =5
fps = 100000 # frame per sec
#frn = endstop # frame number of the animation

######
#Remove 99% melted ice
phi_w_sol_backup[phi_sol>1-non_porous_vol_frac] = np.nan
phi_w_sol[phi_sol>1-non_porous_vol_frac] = np.nan
phi_i_sol[phi_sol>1- non_porous_vol_frac] = np.nan
T_sol[phi_sol>1-non_porous_vol_frac] = np.nan
T_sol[np.isnan(phi_i_sol)] = np.nan
phi_w_sol_backup[np.isnan(phi_i_sol)] = np.nan
phi_w_sol[np.isnan(phi_i_sol)] = np.nan

#####

[N,frn] = np.shape(T_sol) # frame number of the animation from the saved file
[Xy,Yf] = np.meshgrid(Grid.xf,Grid.yf)
# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize=(15,7.5) , dpi=100)
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)


ax3.set_xlim([-30,0])#np.max(T_sol)-273.16])
ax3.set_ylim([ymax_loc,Grid.ymin])
ax3.set_xlabel(r'T[$^\circ$C]')


ax1.set_xlim([0,1])
ax1.set_ylim([ymax_loc,Grid.ymin])
ax1.set_xlabel(r'$\phi$')
ax1.set_ylabel(r'Depth $[m]$')


#ax2.set_xlim([np.min(phi_w_sol/(1-phi_i_sol)),np.max(phi_w_sol/(1-phi_i_sol))])
ax2.set_xlim([0,0.1])
ax2.set_ylim([ymax_loc,Grid.ymin])
ax2.set_xlabel(r'$LWC$')

manager = plt.get_current_fig_manager()
manager.window.showMaximized()

x = Yc[:,1]
y = np.zeros_like(x)
z = np.zeros_like(x)
a = np.zeros_like(Yf[:,1])
face = Yf[:,1]

line1, = ax1.plot(y, x, lw=2, color= 'k',linestyle='-')
line2, = ax2.plot(z, x, lw=2, color= 'k')
line3, = ax3.plot(y, x, lw=2, color= 'k',linestyle='--')
line4, = ax3.plot(z, x, lw=2, color= 'k')
line5, = ax1.plot(y, x, lw=2, color= 'k',linestyle='-',alpha=0.5)


ax2.set_title("Time: %0.2f days" %t[0],loc = 'center', fontsize=18)
plt.tight_layout()
line   = [line1, line2, line3, line4, line5]
 
data  = pd.read_csv('/Users/ms6985/Desktop/Research_new/meltwater-percolation/Infiltration/JPL/Samira-data/Achilig_upGPR_data.csv')
dates = data['date'].apply(lambda x: datetime.strptime(x, "'%d-%b-%Y %H:%M:%S.%f'")) 
t_dates = []
t_dates = ([dates[850] + timedelta(seconds = i) for i in t])
t_dates = np.array([[np.datetime64(i) for i in t_dates]])

def animate(i,u_sol,Sw_sol,phi_w_sol,phi_i_sol,Yc,Yf,time,t_dates):

# initialization function: plot the background of each frame
def init():
    line[0].set_data([], [])
    line[1].set_data([], [])
    line[2].set_data([], [])
    line[3].set_data([], [])
    line[4].set_data([], [])
    return line,

# dist = 0

# patch1 = ax1.add_patch(Rectangle((0, 0), 1,dist,
#                                  edgecolor = blue,
#                                  facecolor = blue,
#                                  fill=True,
#                                  lw=3))
# patch2 = ax2.add_patch(Rectangle((0, 0), 1,dist,
#                                  edgecolor = blue,
#                                  facecolor = blue,
#                                  fill=True,
#                                  lw=3))
# patch3 = ax3.add_patch(Rectangle((-30, 0), 30,dist,
#                                  edgecolor = blue,
#                                  facecolor = blue,
#                                  fill=True,
#                                  lw=3))

# animation function.  This is called sequentially
def animate(i,u_sol,Sw_sol,phi_w_sol,phi_i_sol,Yc,Yf,time,t_dates):
    #patch1.remove()
    #patch2.remove()
    #patch3.remove()
     # ax1.cla()
     # ax2.cla()
     # ax3.cla()    
    x = Yc[:,1] 
    y = np.transpose(u_sol[:,i].reshape(Grid.Nx,Grid.Ny))[:,1]-273.16    
    z = np.transpose(phi_w_sol[:,i].reshape(Grid.Nx,Grid.Ny))[:,1] 
    zz= 1-np.transpose(phi_i_sol[:,i].reshape(Grid.Nx,Grid.Ny))[:,1] 
    zz0= 1-np.transpose(phi_i_sol[:,0].reshape(Grid.Nx,Grid.Ny))[:,1]                           
    Tmelt = 0.0*Yc[:,1]
    line[0].set_data( zz, x)
    line[1].set_data(z, x)
    line[2].set_data(Tmelt, x)
    line[3].set_data(y, x)
    line[4].set_data( zz0, x)
    if np.any(np.isnan(phi_w_sol[:,i])):
        
        #ax1.axhline(y=np.max(Yc_col[np.isnan(phi_w_sol[:,i])]), c=blue,linestyle='-',lw=3)
        #ax2.axhline(y=np.max(Yc_col[np.isnan(phi_w_sol[:,i])]), c=blue,linestyle='-',lw=3)
        #ax3.axhline(y=np.max(Yc_col[np.isnan(phi_w_sol[:,i])]), c=blue,linestyle='-',lw=3)
        #add rectangle to plot
        dist = np.max(Yc_col[np.isnan(phi_w_sol[:,i])])
        
        
        ax1.add_patch(Rectangle((0, .25), 1, .50,
                              edgecolor=(1, 1, 1, 1),
                              facecolor=(1, 1, 1, 1),
                              linewidth=2))

        ax2.add_patch(Rectangle((0, .25), 0.1, .50,
                              edgecolor=(1, 1, 1, 1),
                              facecolor=(1, 1, 1, 1),
                              linewidth=2))
        
        ax3.add_patch(Rectangle((-30, .25), 30, .50,
                              edgecolor=(1, 1, 1, 1),
                              facecolor=(1, 1, 1, 1),
                              linewidth=2))
 
        # ax1.add_patch(Rectangle((0, 0), 1,dist,
        #                       edgecolor=(1, 1, 1, 1),
        #                       facecolor=(1, 1, 1, 1),
        #                       linewidth=2))
        
        # ax2.add_patch(Rectangle((0, 0), 1,dist,
        #                       edgecolor=(1, 1, 1, 1),
        #                       facecolor=(1, 1, 1, 1),
        #                       linewidth=2))
        
        # ax3.add_patch(Rectangle((-30, 0), 30,dist,
        #                       edgecolor=(1, 1, 1, 1),
        #                       facecolor=(1, 1, 1, 1),
        #                       linewidth=2))
        
        
        
        patch1 = ax1.add_patch(Rectangle((0, 0), 1,dist,
                                         edgecolor = blue,
                                         facecolor = blue,
                                         fill=True,
                                         lw=3))
        patch2 = ax2.add_patch(Rectangle((0, 0), 1,dist,
                                         edgecolor = blue,
                                         facecolor = blue,
                                         fill=True,
                                         lw=3))
        patch3 = ax3.add_patch(Rectangle((-30, 0), 30,dist,
                                         edgecolor = blue,
                                         facecolor = blue,
                                         fill=True,
                                         lw=3))
    else:
        ax1.axhline(y=0, c='white',linestyle='-',lw=3) 
        ax2.axhline(y=0, c='white',linestyle='-',lw=3)
        ax3.axhline(y=0, c='white',linestyle='-',lw=3)
        dist = 0
        patch1 = ax1.add_patch(Rectangle((0, 0), 1,dist,
                                         edgecolor = blue,
                                         facecolor = blue,
                                         fill=True,
                                         lw=3))
        patch2 = ax2.add_patch(Rectangle((0, 0), 1,dist,
                                         edgecolor = blue,
                                         facecolor = blue,
                                         fill=True,
                                         lw=3))
        patch3 = ax3.add_patch(Rectangle((0, 0), 1,dist,
                                         edgecolor = blue,
                                         facecolor = blue,
                                         fill=True,
                                         lw=3))
        
    #plt.tight_layout()
    
    
    ax2.set_title(f"{np.datetime_as_string(t_dates[0][i], unit='s')}",loc = 'center', fontsize=18)
    #ax1.legend(loc='lower right', shadow=False, fontsize='medium')
    #ax2.legend(loc='best', shadow=False, fontsize='medium')
    ax3.set_xlim([-30,0])#np.max(T_sol)-273.16])
    ax1.set_xlim(np.min([1-phi_i_sol[~np.isnan(phi_i_sol)]]),np.max(1-phi_i_sol[~np.isnan(phi_i_sol)]))
    ax2.set_xlim([0,0.1])
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
ii=1
ani = animation.FuncAnimation(fig, animate, frn, init_func=init, fargs=(T_sol[:,::ii],s_w_sol[:,::ii],phi_w_sol[:,::ii],phi_i_sol[:,::ii],Yc[:,:],Yf[:,:],tday[::ii],t_dates[::ii])
                               , interval=1/fps)

ani.save(f"../Figures/{simulation_name}_combined_with_seconds_new.mov", writer='ffmpeg', fps=30)

'''


#New plots

fc = k0*k_w0*rho_w*grav/mu_w*phi_L**3 #Infiltration capacity (m/s)

fig, ax1 = plt.subplots(1,1, sharex=True,figsize=(20,6),dpi=50)
rcParams.update({'font.size': 22})
Grid.ymax = 2.6

Flux_array = fc*phi_w_sol**n*(1-phi_i_sol)**(m-n)/phi_L**3 #Flux evaluation
Flux_array[phi_sol>1-non_porous_vol_frac*100] = np.nan

plot = [plt.contourf(t_dates, depth_array, Flux_array[:200,:],cmap="Blues",levels=100,ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(np.linspace(np.min(Flux_array[~np.isnan(Flux_array)]),np.max(Flux_array[~np.isnan(Flux_array)]),10000))
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$Flux$ [m/s]', labelpad=-100, y=0.5, rotation=90)
plt.ylim([Grid.ymax,Grid.ymin])

#yyyymmdd format
chosen_dates = [20160529,20160623,20160718,20160812,20160906]
chosen_dates = [datetime.strptime(str(int(dateee)),'%Y%m%d') for dateee in chosen_dates]

plt.xticks(chosen_dates,rotation = 0) # Rotates X-Axis Ticks by 45-degrees

chosen_dates = [20160524,20160918]
chosen_dates = [datetime.strptime(str(int(dateee)),'%Y%m%d') for dateee in chosen_dates]

plt.xlim(chosen_dates)

plt.tight_layout(w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.ylabel(r'z [m]')
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_rhow{rho_w}_only_Fluxes.pdf',bbox_inches='tight', dpi = 50)


fig, ax1 = plt.subplots(1,1, sharex=True,figsize=(20,6),dpi=50)
rcParams.update({'font.size': 22})
Grid.ymax = 2.6

Flux_array = fc*(1-phi_i_sol)**(m)/phi_L**3 #Flux evaluation
Flux_array[phi_sol>1-non_porous_vol_frac*100] = np.nan

plot = [plt.contourf(t_dates, depth_array, Flux_array[:200,:],cmap="Blues",levels=100,ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(np.linspace(np.min(Flux_array[~np.isnan(Flux_array)]),np.max(Flux_array[~np.isnan(Flux_array)]),10000))
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$Flux$ [m/s]', labelpad=-150, y=0.5, rotation=90)
plt.ylim([Grid.ymax,Grid.ymin])

#yyyymmdd format
chosen_dates = [20160529,20160623,20160718,20160812,20160906]
chosen_dates = [datetime.strptime(str(int(dateee)),'%Y%m%d') for dateee in chosen_dates]

plt.xticks(chosen_dates,rotation = 0) # Rotates X-Axis Ticks by 45-degrees

chosen_dates = [20160524,20160918]
chosen_dates = [datetime.strptime(str(int(dateee)),'%Y%m%d') for dateee in chosen_dates]

plt.xlim(chosen_dates)

plt.tight_layout(w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.ylabel(r'z [m]')
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_rhow{rho_w}_only_Flux2saturate.pdf',bbox_inches='tight', dpi = 50)


#Analytical plot Marc for 50% porosity
fig, ax1 = plt.subplots(2,1, sharex=True,figsize=(6,12),dpi=100)
rcParams.update({'font.size': 22})
phi0 =0.5
s_w_analy = np.linspace(0/phi0,0.5/phi0,1000)
Flux_analy= fc*s_w_analy**n*(phi0)**(m)/phi_L**3 #Flux evaluation

plt.subplot(2,1,1)
plt.plot(s_w_analy,Flux_analy*1e5,'b-',color=blue,label='$s_{wr}=0$')

s_w_analy1= np.linspace(0.07,0.5/phi0,1000)
Flux_analy1= fc*(s_w_analy-0.07)**n*(phi0)**(m)/phi_L**3 #Flux evaluation
plt.plot(s_w_analy1,Flux_analy*1e5,'k-',label='$s_{wr}=0.07$')
plt.xlim([0,1])
plt.ylim([-1e-6*1e5,np.max([Flux_analy*1e5,Flux_analy1*1e5])])
plt.ylabel(r'Flux, $I \times 10^5$ [m/s]')
plt.legend(loc='best')
plt.plot(0.07,0,'k+',markersize=10)

plt.axvline(0.5,ymin=0,ymax=2,color='black',linestyle='-.',alpha=0.5)
plt.plot([0,0.5], [0,1.71], color=blue,linestyle="--")
plt.plot([0,0.5], [0,1.43], color='black',linestyle="--")

plt.subplot(2,1,2)
plt.plot(s_w_analy,Flux_analy*1e5,'b-',color=blue,label='$s_{wr}=0$')

s_w_analy1= np.linspace(0.07,0.5/phi0,1000)
Flux_analy1= fc*(s_w_analy-0.07)**n*(phi0)**(m)/phi_L**3 #Flux evaluation
plt.plot(s_w_analy1,Flux_analy*1e5,'k-',label='$s_{wr}=0.07$')
plt.xlim([0,1])
plt.ylim([-1e-6*1e5,np.max([Flux_analy*1e5,Flux_analy1*1e5])])
plt.ylabel(r'Flux, $I \times 10^5$ [m/s]')

plt.xlabel('Water saturation, $LWC/\phi$ [-]')
plt.plot(0.07,0,'k+',markersize=10)
plt.tight_layout(w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(wspace=0.0, hspace=0.0)

plt.axhline(1.71,xmin=0,xmax=1,color='black',linestyle='-.',alpha=0.5)


plt.plot([0,0.5], [0,1.71], color=blue,linestyle="--")
plt.plot([0,0.532], [0,1.71], color='black',linestyle="--")
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_rhow{rho_w}_res_sat_comparison.pdf',bbox_inches='tight', dpi = 50)

#Peclet number estimation for 50% porosity
K_h = k0*k_w0*rho_w*grav/mu_w #Infiltration capacity (m/s)

LWC = 0.03; Tfirn = -10; phi0 =0.5; t_pulse = 4*day2s
K_bar = 2.22362*(rho_i/rho_w*(1-phi0))**1.885
kk_bar = K_bar/(rho_i*cp_i*(1-phi0))
v= (fc*LWC**n*(phi0)**(m-n)/phi_L**3 - 0)/(0.00573*(1-phi0)*(-Tfirn) + LWC - 0)
grain_size= 2e-3 #grainsize 1mm or 2mm Humphrey et al. (2021)
Pe_grain = v*grain_size/kk_bar
Pe_cell =  v*Grid.dy/kk_bar
Pe_cell_shock =  v*(v*4*day2s)/kk_bar


#Peclet number wrong and rough comparison
fig, ax1 = plt.subplots(1,1, sharex=True,figsize=(20,6),dpi=50)
rcParams.update({'font.size': 22})
Grid.ymax = 2.6

speed_array = (fc*phi_w_sol**n*(1-phi_i_sol)**(m-n)/phi_L**3)/(phi_w_sol + 0.00573*phi_i_sol*(Tm-T_sol)) #Rough speed evaluation
speed_array[phi_sol>0.8] = np.nan
K_bar_array = 2.22362*(rho_i/rho_w*phi_i_sol)**1.885
kk_bar_array = K_bar_array/(rho_i*cp_i*phi_i_sol)

Pe_L_array = speed_array*Grid.dy/kk_bar_array
Pe_grain_array = speed_array*grain_size/kk_bar_array

plot = [plt.contourf(t_dates, depth_array, Pe_L_array[:200,:],cmap="Blues",levels=100,vmin=np.min(Pe_L_array[~np.isnan(Pe_L_array)]),vmax=np.max(Pe_L_array[~np.isnan(Pe_L_array)])/2,ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(np.linspace(np.min(Pe_L_array[~np.isnan(Pe_L_array)]),np.max(Pe_L_array[~np.isnan(Pe_L_array)])/2,10000))
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$Pe_{cell}$', labelpad=-105, y=0.5, rotation=90)
plt.ylim([Grid.ymax,Grid.ymin])

#yyyymmdd format
chosen_dates = [20160529,20160623,20160718,20160812,20160906]
chosen_dates = [datetime.strptime(str(int(dateee)),'%Y%m%d') for dateee in chosen_dates]

plt.xticks(chosen_dates,rotation = 0) # Rotates X-Axis Ticks by 45-degrees

chosen_dates = [20160524,20160918]
chosen_dates = [datetime.strptime(str(int(dateee)),'%Y%m%d') for dateee in chosen_dates]

plt.xlim(chosen_dates)

plt.tight_layout(w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.ylabel(r'z [m]')
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_rhow{rho_w}_only_wrong_PeL.pdf',bbox_inches='tight', dpi = 50)


fig, ax1 = plt.subplots(1,1, sharex=True,figsize=(20,6),dpi=50)
rcParams.update({'font.size': 22})
Grid.ymax = 2.6

speed_array = (fc*phi_w_sol**n*(1-phi_i_sol)**(m-n)/phi_L**3)/(phi_w_sol + 0.00573*phi_i_sol*(Tm-T_sol)) #Rough speed evaluation
speed_array[phi_sol>0.8] = np.nan
K_bar_array = 2.22362*(rho_i/rho_w*phi_i_sol)**1.885
kk_bar_array = K_bar_array/(rho_i*cp_i*phi_i_sol)

Pe_L_array = speed_array*Grid.dy/kk_bar_array
Pe_grain_array = speed_array*grain_size/kk_bar_array

plot = [plt.contourf(t_dates, depth_array, Pe_grain_array[:200,:],cmap="Blues",levels=100,vmin=np.min(Pe_grain_array[~np.isnan(Pe_grain_array)]),vmax=np.max(Pe_grain_array[~np.isnan(Pe_grain_array)])/2,ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(np.linspace(np.min(Pe_grain_array[~np.isnan(Pe_grain_array)]),np.max(Pe_grain_array[~np.isnan(Pe_grain_array)])/2,10000))
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$Pe_{grain}$', labelpad=-150, y=0.5, rotation=90)
plt.ylim([Grid.ymax,Grid.ymin])

#yyyymmdd format
chosen_dates = [20160529,20160623,20160718,20160812,20160906]
chosen_dates = [datetime.strptime(str(int(dateee)),'%Y%m%d') for dateee in chosen_dates]

plt.xticks(chosen_dates,rotation = 0) # Rotates X-Axis Ticks by 45-degrees

chosen_dates = [20160524,20160918]
chosen_dates = [datetime.strptime(str(int(dateee)),'%Y%m%d') for dateee in chosen_dates]

plt.xlim(chosen_dates)

plt.tight_layout(w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.ylabel(r'z [m]')
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_rhow{rho_w}_only_wrong_Pegrain.pdf',bbox_inches='tight', dpi = 50)




'''
# ######
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'font.family': 'Serif'})
phi_w_sol_backup = phi_w_sol.copy()
phi_i_sol_backup = phi_i_sol.copy()
T_sol_backup = T_sol.copy()
# # #Remove 99% melted ice
phi_w_sol_backup[phi_i_sol<non_porous_vol_frac] = 0; phi_w_sol_backup[np.isnan(phi_w_sol_backup)] =0
phi_i_sol_backup[phi_i_sol<non_porous_vol_frac] = 0; phi_i_sol_backup[np.isnan(phi_i_sol_backup)] =0
T_sol_backup[phi_i_sol<non_porous_vol_frac] = Tm-30;T_sol_backup[np.isnan(T_sol_backup)] =Tm-30
T_sol_backup[T_sol_backup<Tm-30] = Tm-30
#####


######################################################################
#Making video
######################################################################
blue = np.array([135, 206, 235])/255
#A fun combined  phi, LWC, T
from datetime import datetime 
from datetime import timedelta
import pandas as pd 
t = np.array(t)
tday = t/day2s
ymax_loc =5
fps = 100000 # frame per sec
#frn = endstop # frame number of the animation
data  = pd.read_csv('/Users/ms6985/Desktop/Research_new/meltwater-percolation/Infiltration/JPL/Samira-data/Achilig_upGPR_data.csv')
dates = data['date'].apply(lambda x: datetime.strptime(x, "'%d-%b-%Y %H:%M:%S.%f'")) 
t_dates = []
t_dates = ([dates[850] + timedelta(seconds = i) for i in t])
t_dates = np.array([[np.datetime64(i) for i in t_dates]])

#New Contour plot combined new
Grid.ymax = 3
horizontal_units = 'km' #m or km
fps = 100000 # frame per sec 
t = np.array(t)
#frn = endstop # frame number of the animation
[N,frn] = np.shape(phi_w_sol_backup) # frame number of the animation from the saved file
t_day = t/day2s
def update_plot(frame_number, phi_w_sol_backup,phi_i_sol_backup,T_sol_backup, plot,t_datess):
    fig.suptitle(f"{np.datetime_as_string(t_datess[frame_number], unit='s')}", fontsize=18)
    plt.subplot(3,1,1)
    if horizontal_units == 'km':
        plt.xlim([Grid.xmin/1e3, Grid.xmax/1e3])
        plot = [plt.contourf(Xc/1e3, Yc, np.transpose(phi_w_sol_backup[:,frame_number].reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin=np.min(phi_w_sol_backup[~np.isnan(phi_w_sol_backup)]),vmax=np.max(phi_w_sol_backup[~np.isnan(phi_w_sol_backup)]),levels=100)]
    else:
        plt.xlim([Grid.xmin, Grid.xmax])    
        plot = [plt.contourf(Xc, Yc, np.transpose(phi_w_sol_backup[:,frame_number].reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin=np.min(phi_w_sol_backup[~np.isnan(phi_w_sol_backup)]),vmax=np.max(phi_w_sol_backup[~np.isnan(phi_w_sol_backup)]),levels=100)]
 
    plt.ylim([Grid.ymax,Grid.ymin])
    ax1.set_ylabel(r'$z$ [m]')
    
    plt.subplot(3,1,2)
    if horizontal_units == 'km':
        plt.xlim([Grid.xmin/1e3, Grid.xmax/1e3])
        plot1 = [plt.contourf(Xc/1e3, Yc, np.transpose((1-phi_i_sol_backup[:,frame_number]).reshape(Grid.Nx,Grid.Ny)),cmap="Greys_r",vmin=np.min(1-phi_i_sol_backup[~np.isnan(phi_i_sol_backup)]),vmax=np.max(1-phi_i_sol_backup[~np.isnan(phi_i_sol_backup)]),levels=100)]
    else:
        plt.xlim([Grid.xmin, Grid.xmax])  
        plot1 = [plt.contourf(Xc, Yc, np.transpose((1-phi_i_sol_backup[:,frame_number]).reshape(Grid.Nx,Grid.Ny)),cmap="Greys_r",vmin=np.min(1-phi_i_sol_backup[~np.isnan(phi_i_sol_backup)]),vmax=np.max(1-phi_i_sol_backup[~np.isnan(phi_i_sol_backup)]),levels=100)]
    
    plt.ylim([Grid.ymax,Grid.ymin])
    ax2.set_ylabel(r'$z$ [m]')
    
    plt.subplot(3,1,3)
    if horizontal_units == 'km':
        plt.xlim([Grid.xmin/1e3, Grid.xmax/1e3])
        ax3.set_xlabel(r'$x$ [km]')
        plot1 = [plt.contourf(Xc/1e3, Yc, np.transpose((T_sol_backup[:,frame_number]-Tm).reshape(Grid.Nx,Grid.Ny)),cmap="Reds",vmin=np.min(T_sol_backup[~np.isnan(T_sol_backup)])-Tm,vmax=np.max(T_sol_backup[~np.isnan(T_sol_backup)])-Tm,levels=100)]
    
    else:
        plt.xlim([Grid.xmin, Grid.xmax])  
        plot1 = [plt.contourf(Xc, Yc, np.transpose((T_sol_backup[:,frame_number]-Tm).reshape(Grid.Nx,Grid.Ny)),cmap="Reds",vmin=np.min(T_sol_backup[~np.isnan(T_sol_backup)])-Tm,vmax=np.max(T_sol_backup[~np.isnan(T_sol_backup)])-Tm,levels=100)]
        ax3.set_xlabel(r'$x$ [m]')
    plt.ylim([Grid.ymax,Grid.ymin])
    ax3.set_ylabel(r'$z$ [m]')


fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True,figsize=(10,10),dpi=100)
#fig.set_size_inches(2000 / 100, 1605 / 100)
plt.subplot(3,1,1)
ax1.set_ylabel(r'$z$')
if horizontal_units == 'km':
    plot = [plt.contourf(Xc/1e3, Yc, np.transpose(phi_w_sol_backup[:,0].reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin=np.min(phi_w_sol_backup[~np.isnan(phi_w_sol_backup)]),vmax=np.max(phi_w_sol_backup[~np.isnan(phi_w_sol_backup)]),levels=100)]
    plt.xlim([Grid.xmin/1e3, Grid.xmax/1e3])
else:
    plot = [plt.contourf(Xc, Yc, np.transpose(phi_w_sol_backup[:,0].reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin=np.min(phi_w_sol_backup[~np.isnan(phi_w_sol_backup)]),vmax=np.max(phi_w_sol_backup[~np.isnan(phi_w_sol_backup)]),levels=100)]
    plt.xlim([Grid.xmin, Grid.xmax]) 
plt.ylim([Grid.ymax,Grid.ymin])
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(phi_w_sol_backup)
mm.set_clim(np.min(phi_w_sol_backup), np.max(phi_w_sol_backup))
clb = plt.colorbar(mm, orientation='vertical',ax=[ax1],aspect=10,pad=0.01)
clb.set_label(r'LWC')
ax1.set_ylabel(r'$z$ [m]')

plt.subplot(3,1,2)
mm = plt.cm.ScalarMappable(cmap=cm.Greys_r)
ax2.set_ylabel(r'$z$')
mm.set_array(1-phi_i_sol_backup)
mm.set_clim(np.min(1-phi_i_sol_backup),np.max(1-phi_i_sol_backup))
clb = plt.colorbar(mm, orientation='vertical',ax=[ax2],aspect=10,pad=0.01)
if horizontal_units == 'km':
    plot1 = [plt.contourf(Xc/1e3, Yc, np.transpose((1-phi_i_sol_backup[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Greys_r",vmin=np.min(1-phi_i_sol_backup[~np.isnan(phi_i_sol_backup)]),vmax=np.max(1-phi_i_sol_backup[~np.isnan(phi_i_sol_backup)]),levels=100)]
    plt.xlim([Grid.xmin/1e3, Grid.xmax/1e3])
else:
    plot1 = [plt.contourf(Xc, Yc, np.transpose((1-phi_i_sol_backup[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Greys_r",vmin=np.min(1-phi_i_sol_backup[~np.isnan(phi_i_sol_backup)]),vmax=np.max(1-phi_i_sol_backup[~np.isnan(phi_i_sol_backup)]),levels=100)]
    plt.xlim([Grid.xmin, Grid.xmax]) 
plt.ylim([Grid.ymax,Grid.ymin])
clb.set_label(r'Porosity')
ax2.set_ylabel(r'$z$ [m]')

plt.subplot(3,1,3)
mm = plt.cm.ScalarMappable(cmap=cm.Reds)
mm.set_array(T_sol_backup-Tm)
mm.set_clim(np.min(T_sol_backup)-Tm, np.max(T_sol_backup)-Tm)
clb = plt.colorbar(mm, orientation='vertical',ax=[ax3],aspect=10,pad=0.01)
clb.set_label(r'T $[^\circ C]$')
if horizontal_units == 'km':
    plot1 = [plt.contourf(Xc/1e3, Yc, np.transpose((T_sol_backup[:,0]-Tm).reshape(Grid.Nx,Grid.Ny)),cmap="Reds",vmin=np.min(T_sol_backup[~np.isnan(T_sol_backup)])-Tm,vmax=np.max(T_sol_backup[~np.isnan(T_sol_backup)])-Tm,levels=100)]
    plt.xlim([Grid.xmin/1e3, Grid.xmax/1e3])
    ax3.set_xlabel(r'$x$ [km]')
else:
    plot1 = [plt.contourf(Xc, Yc, np.transpose((T_sol_backup[:,0]-Tm).reshape(Grid.Nx,Grid.Ny)),cmap="Reds",vmin=np.min(T_sol_backup[~np.isnan(T_sol_backup)])-Tm,vmax=np.max(T_sol_backup[~np.isnan(T_sol_backup)])-Tm,levels=100)]
    plt.xlim([Grid.xmin, Grid.xmax])  
    ax3.set_xlabel(r'$x$ [m]')
plt.ylim([Grid.ymax,Grid.ymin])

ax3.set_ylabel(r'$z$ [m]')

#fig.suptitle("t= %0.2f days" % t_day[0], fontsize=22)

#ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(phi_w_sol[:,:],phi_i_sol[:,:],T_sol[:,:], plot[:0],t_day[:]), interval=1/fps)

ii = int(tf/(4*day2s))
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(phi_w_sol_backup[:,::ii],phi_i_sol_backup[:,::ii],T_sol_backup[:,::ii], plot[::ii],t_dates[0,::ii]), interval=1/fps)

ani.save(f"../Figures/{simulation_name}_{C_L/phi_L}_{Grid.Nx}by{Grid.Ny}_tf{t[frn-1]}_slope_{tilt_angle}deg_new_LWC.mov", writer='ffmpeg', fps=30)

#ani.save(fn+'.gif',writer='imagemagick',fps=fps)
#cmd = 'magick convert %s.gif -fuzz 5%% -layers Optimize %s_r.gif'%(fn,fn)
#subprocess.check_output(cmd)
'''


# #plot contour of porosity and the detected surface
# step = -1#1726#1739#1781
# phi_i = np.transpose([(phi_i_sol[:,step])])
# T     = np.transpose([(T_sol[:,step])])
# H     = np.transpose([(H_sol[:,step])])
# C     = np.transpose([(C_sol[:,step])])


non_porous_vol_frac = 1e-3

#Porosity
plt.figure()
plt.contourf(Xc, Yc, np.transpose((1-phi_i[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Greys_r",vmin=np.min(1-phi_i[~np.isnan(phi_i)]),vmax=np.max(1-phi_i[~np.isnan(phi_i)])+0.1,levels=100)
if np.any(np.isnan(phi_i)):
    nan_points  = np.argwhere(np.isnan(phi_i))
    plt.plot(Xc_col[nan_points[:,0]],Yc_col[nan_points[:,0]],'b*')

if np.any(phi_i<=non_porous_vol_frac):
    dof_complete_melt  = Grid.dof[phi_i[:,0] < non_porous_vol_frac]
    dof_partial_melt   = np.setdiff1d(Grid.dof,dof_complete_melt) #saturated cells 
    dof_f_bnd_melt     = find_faces(dof_complete_melt,D,Grid) 
    dof_f_bnd_melt     = dof_f_bnd_melt[dof_f_bnd_melt>Grid.Nfx]  #location of the actual surface
    ytop,ybot          = find_top_bot_cells(dof_f_bnd_melt,D,Grid)
    plt.plot(Xc_col[ytop-1],Yc_col[ytop-1],'r.')
    plt.plot(Xc_col[dof_complete_melt-1],Yc_col[dof_complete_melt-1],'k.')

plt.colorbar()
plt.ylim([Grid.ymax,Grid.ymin])

#Temperature
plt.figure()
plt.contourf(Xc, Yc, np.transpose((T[:,0]-273.16).reshape(Grid.Nx,Grid.Ny)),cmap="Reds",vmin=np.min(T[~np.isnan(phi_i)]-273.16),vmax=np.max(T[~np.isnan(phi_i)]-273.16),levels=100)
plt.colorbar()
plt.ylim([Grid.ymax,Grid.ymin])
if np.any(np.isnan(phi_i)):
    nan_points  = np.argwhere(np.isnan(phi_i))
    plt.plot(Xc_col[nan_points[:,0]],Yc_col[nan_points[:,0]],'b*')

if np.any(phi_i<=non_porous_vol_frac):
    dof_complete_melt  = Grid.dof[phi_i[:,0] < non_porous_vol_frac]
    dof_partial_melt   = np.setdiff1d(Grid.dof,dof_complete_melt) #saturated cells 
    dof_f_bnd_melt     = find_faces(dof_complete_melt,D,Grid) 
    dof_f_bnd_melt     = dof_f_bnd_melt[dof_f_bnd_melt>Grid.Nfx]  #location of the actual surface
    ytop,ybot          = find_top_bot_cells(dof_f_bnd_melt,D,Grid)
    plt.plot(Xc_col[ytop-1],Yc_col[ytop-1],'r.')
    plt.plot(Xc_col[dof_complete_melt-1],Yc_col[dof_complete_melt-1],'k.')
    
    
plt.ylim([Grid.ymax,Grid.ymin])

#Commposition
plt.figure()
plt.contourf(Xc, Yc, np.transpose((C[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Greys_r",vmin=np.min(C[~np.isnan(C)]),vmax=np.max(C[~np.isnan(C)]),levels=100)
plt.colorbar()
plt.ylim([Grid.ymax,Grid.ymin])
if np.any(np.isnan(phi_i)):
    nan_points  = np.argwhere(np.isnan(phi_i))
    plt.plot(Xc_col[nan_points[:,0]],Yc_col[nan_points[:,0]],'b*')

if np.any(phi_i<=non_porous_vol_frac):
    dof_complete_melt  = Grid.dof[phi_i[:,0] < non_porous_vol_frac]
    dof_partial_melt   = np.setdiff1d(Grid.dof,dof_complete_melt) #saturated cells 
    dof_f_bnd_melt     = find_faces(dof_complete_melt,D,Grid) 
    dof_f_bnd_melt     = dof_f_bnd_melt[dof_f_bnd_melt>Grid.Nfx]  #location of the actual surface
    ytop,ybot          = find_top_bot_cells(dof_f_bnd_melt,D,Grid)
    plt.plot(Xc_col[ytop-1],Yc_col[ytop-1],'r.')
    plt.plot(Xc_col[dof_complete_melt-1],Yc_col[dof_complete_melt-1],'k.')
    
#Enthalpy
plt.figure()
plt.contourf(Xc, Yc, np.transpose((H[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Greys_r",vmin=np.min(H[~np.isnan(H)]),vmax=np.max(H[~np.isnan(H)]),levels=100)
plt.colorbar()
plt.ylim([Grid.ymax,Grid.ymin])
if np.any(np.isnan(phi_i)):
    nan_points  = np.argwhere(np.isnan(phi_i))
    plt.plot(Xc_col[nan_points[:,0]],Yc_col[nan_points[:,0]],'b*')

if np.any(phi_i<=non_porous_vol_frac):
    dof_complete_melt  = Grid.dof[phi_i[:,0] < non_porous_vol_frac]
    dof_partial_melt   = np.setdiff1d(Grid.dof,dof_complete_melt) #saturated cells 
    dof_f_bnd_melt     = find_faces(dof_complete_melt,D,Grid) 
    dof_f_bnd_melt     = dof_f_bnd_melt[dof_f_bnd_melt>Grid.Nfx]  #location of the actual surface
    ytop,ybot          = find_top_bot_cells(dof_f_bnd_melt,D,Grid)
    plt.plot(Xc_col[ytop-1],Yc_col[ytop-1],'r.')
    plt.plot(Xc_col[dof_complete_melt-1],Yc_col[dof_complete_melt-1],'k.')


#ice-layer-evolution
nan_cell = 635
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1, figsize=(10,10),dpi=100,sharex=True)
plt.subplot(5,1,1)
plt.plot(t/day2s,C_sol[nan_cell-1,:],color=blue,linewidth=3)
plt.ylabel('$C$ [kg/m$^3$]')

plt.subplot(5,1,2)
plt.plot(t/day2s,H_sol[nan_cell-1,:],color=blue,linewidth=3)
plt.ylabel('H [J/m$^3$]')

plt.subplot(5,1,3)
plt.plot(t/day2s,phi_i_sol[nan_cell-1,:],color=blue,linewidth=3)
plt.ylabel('$\phi_i$')

plt.subplot(5,1,4)
plt.plot(t/day2s,T_sol[nan_cell-1,:]-Tm,color=blue,linewidth=3)
plt.ylabel('T [$^\circ$C]')

plt.subplot(5,1,5)
plt.plot(t/day2s,phi_w_sol[nan_cell-1,:],color=blue,linewidth=3)
plt.ylabel('LWC')

plt.xlabel('Time [days]')

#phi_w_old,phi_i_old,T_old,dTdH_old = eval_phase_behaviorCwH(H_old,Tm,rho_i,rho_w,rho_nw,cp_i,cp_w,cp_nw,C_old,L_fusion)




'''
#Colors
red    = [190/255 ,30/255 ,45/255 ]
blue   = [ 30/255 ,144/255 , 255/255 ]
green  = [  0/255 , 166/255 ,  81/255]
orange = [247/255 , 148/255 ,  30/255]
purple = [102/255 ,  45/255 , 145/255]
plt.rcParams.update({'font.family': "Serif"})
plt.rcParams.update({'font.size': 26})

######################################################################
#Amp = 0.01, corr_length = 40m
######################################################################
data = np.load('samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx40.0m_ly1m_amp0.01_C0.03_50by200_t10368000.npz')
phi_i_sol =data['phi_i_sol']; porosity_change=np.transpose((phi_i_sol[:,-1]-phi_i_sol[:,0]).reshape(Grid.Nx,Grid.Ny))

final_ice_vol_frac = np.transpose((phi_i_sol[:,-1]).reshape(Grid.Nx,Grid.Ny))


Y_max_array = []

#Porosity
plt.figure()
plt.contourf(Xc, Yc, porosity_change,cmap="Greys_r",levels=100)
plt.colorbar()
plt.ylim([Grid.ymax,Grid.ymin])


##find percolation depth at each Xc
for i in range(0,np.shape(porosity_change)[1]):
    Y_max= np.max(Yc[porosity_change[:,i]>0]) - np.min(Yc[final_ice_vol_frac[:,i]>=0.01])
    Y_max_array=np.append(Y_max_array,Y_max)
    
#porosity_change[porosity_change>0]



fig, ax1 = plt.subplots(1,1, figsize=(8,8),dpi=100)
#rcParams.update({'font.size': 20})

plt.subplot(1,1,1)

#color=['red', 'green', 'blue', 'cyan', 'magenta'],
#       error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2)

plt.errorbar(40, np.mean(Y_max_array),np.std(Y_max_array), linestyle='None',color=red, marker='^', mfc=red,mec=red, ms=20, mew=4)

######################################################################
#Amp = 0.01, corr_length = 400m
######################################################################
data = np.load('samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx400.0m_ly1m_amp0.01_C0.03_50by200_t10368000.npz')
phi_i_sol =data['phi_i_sol']; porosity_change=np.transpose((phi_i_sol[:,-1]-phi_i_sol[:,0]).reshape(Grid.Nx,Grid.Ny))
final_ice_vol_frac = np.transpose((phi_i_sol[:,-1]).reshape(Grid.Nx,Grid.Ny))


Y_max_array = []

##find percolation depth at each Xc
for i in range(0,np.shape(porosity_change)[1]):
    Y_max= np.max(Yc[porosity_change[:,i]>0]) - np.min(Yc[final_ice_vol_frac[:,i]>=0.01])
    Y_max_array=np.append(Y_max_array,Y_max)

#rcParams.update({'font.size': 40})

plt.errorbar(400, np.mean(Y_max_array),np.std(Y_max_array), linestyle='None', marker='^',color=red, mfc=red,mec=red,label='Amp=0.01', ms=20, mew=4)


######################################################################
#Amp = 0.01, corr_length = 4000m
######################################################################
data = np.load('samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx4000.0m_ly1m_amp0.01_C0.03_50by200_t10368000.npz')
phi_i_sol =data['phi_i_sol']; porosity_change=np.transpose((phi_i_sol[:,-1]-phi_i_sol[:,0]).reshape(Grid.Nx,Grid.Ny))
final_ice_vol_frac = np.transpose((phi_i_sol[:,-1]).reshape(Grid.Nx,Grid.Ny))


Y_max_array = []

##find percolation depth at each Xc
for i in range(0,np.shape(porosity_change)[1]):
    Y_max= np.max(Yc[porosity_change[:,i]>0]) - np.min(Yc[final_ice_vol_frac[:,i]>=0.01])
    Y_max_array=np.append(Y_max_array,Y_max)

#rcParams.update({'font.size': 40})

plt.errorbar(4000, np.mean(Y_max_array),np.std(Y_max_array), linestyle='None', marker='^',color=red, mfc=red,mec=red, ms=20, mew=4)


######################################################################
#Amp = 0.05, corr_length = 40m
######################################################################
data = np.load('samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx40.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz')
phi_i_sol =data['phi_i_sol']; porosity_change=np.transpose((phi_i_sol[:,-1]-phi_i_sol[:,0]).reshape(Grid.Nx,Grid.Ny))
final_ice_vol_frac = np.transpose((phi_i_sol[:,-1]).reshape(Grid.Nx,Grid.Ny))


Y_max_array = []

##find percolation depth at each Xc
for i in range(0,np.shape(porosity_change)[1]):
    Y_max= np.max(Yc[porosity_change[:,i]>0]) - np.min(Yc[final_ice_vol_frac[:,i]>=0.01])
    Y_max_array=np.append(Y_max_array,Y_max)

#rcParams.update({'font.size': 40})

plt.subplot(1,1,1)

#color=['red', 'green', 'blue', 'cyan', 'magenta'],
#       error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2)

plt.errorbar(40, np.mean(Y_max_array),np.std(Y_max_array), linestyle='None',color=blue, marker='^', mfc=blue,mec=blue, ms=20, mew=4)

######################################################################
#Amp = 0.05, corr_length = 400m
######################################################################
data = np.load('samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx400.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz')
phi_i_sol =data['phi_i_sol']; porosity_change=np.transpose((phi_i_sol[:,-1]-phi_i_sol[:,0]).reshape(Grid.Nx,Grid.Ny))
final_ice_vol_frac = np.transpose((phi_i_sol[:,-1]).reshape(Grid.Nx,Grid.Ny))


Y_max_array = []

##find percolation depth at each Xc
for i in range(0,np.shape(porosity_change)[1]):
    Y_max= np.max(Yc[porosity_change[:,i]>0]) - np.min(Yc[final_ice_vol_frac[:,i]>=0.01])
    Y_max_array=np.append(Y_max_array,Y_max)

#rcParams.update({'font.size': 40})

plt.errorbar(400, np.mean(Y_max_array),np.std(Y_max_array), linestyle='None', marker='^',color=blue, mfc=blue,mec=blue,label='Amp=0.05', ms=20, mew=4)


#plt.plot(,'bs',color=blue)

plt.ylabel('Percolation depth [m]')
plt.xlabel('Correlation length [m]')
plt.legend()
plt.ylim([3, 0])
plt.tight_layout()
plt.savefig(f'../Figures/correlation_lengths_amp.pdf',bbox_inches='tight', dpi = 300)

######################################################################
#Amp = 0.05, corr_length = 4000m
######################################################################
data = np.load('samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx4000.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz')
phi_i_sol =data['phi_i_sol']; porosity_change=np.transpose((phi_i_sol[:,-1]-phi_i_sol[:,0]).reshape(Grid.Nx,Grid.Ny))
final_ice_vol_frac = np.transpose((phi_i_sol[:,-1]).reshape(Grid.Nx,Grid.Ny))


Y_max_array = []

##find percolation depth at each Xc
for i in range(0,np.shape(porosity_change)[1]):
    Y_max= np.max(Yc[porosity_change[:,i]>0]) - np.min(Yc[final_ice_vol_frac[:,i]>=0.01])
    Y_max_array=np.append(Y_max_array,Y_max)

#rcParams.update({'font.size': 40})

plt.errorbar(4000, np.mean(Y_max_array),np.std(Y_max_array), linestyle='None', marker='^',color=blue, mfc=blue,mec=blue, ms=20, mew=4)



######################################################################
#Amp = 0.1, corr_length = 40m
######################################################################
data = np.load('samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx40.0m_ly1m_amp0.1_C0.03_50by200_t10368000.npz')
phi_i_sol =data['phi_i_sol']; porosity_change=np.transpose((phi_i_sol[:,-1]-phi_i_sol[:,0]).reshape(Grid.Nx,Grid.Ny))
final_ice_vol_frac = np.transpose((phi_i_sol[:,-1]).reshape(Grid.Nx,Grid.Ny))


Y_max_array = []

##find percolation depth at each Xc
for i in range(0,np.shape(porosity_change)[1]):
    Y_max= np.max(Yc[porosity_change[:,i]>0]) - np.min(Yc[final_ice_vol_frac[:,i]>=0.01])
    Y_max_array=np.append(Y_max_array,Y_max)

#rcParams.update({'font.size': 40})

plt.subplot(1,1,1)

#color=['red', 'green', 'blue', 'cyan', 'magenta'],
#       error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2)

plt.errorbar(40, np.mean(Y_max_array),np.std(Y_max_array), linestyle='None',color=green, marker='^', mfc=green,mec=green, ms=20, mew=4)

######################################################################
#Amp = 0.1, corr_length = 400m
######################################################################
data = np.load('samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx400.0m_ly1m_amp0.1_C0.03_50by200_t10368000.npz')
phi_i_sol =data['phi_i_sol']; porosity_change=np.transpose((phi_i_sol[:,-1]-phi_i_sol[:,0]).reshape(Grid.Nx,Grid.Ny))
final_ice_vol_frac = np.transpose((phi_i_sol[:,-1]).reshape(Grid.Nx,Grid.Ny))


Y_max_array = []

##find percolation depth at each Xc
for i in range(0,np.shape(porosity_change)[1]):
    Y_max= np.max(Yc[porosity_change[:,i]>0]) - np.min(Yc[final_ice_vol_frac[:,i]>=0.01])
    Y_max_array=np.append(Y_max_array,Y_max)

#rcParams.update({'font.size': 40})

plt.errorbar(400, np.mean(Y_max_array),np.std(Y_max_array), linestyle='None', marker='^',color=green, mfc=green,mec=green,label='Amp=0.1', ms=20, mew=4)


#plt.plot(,'bs',color=blue)

plt.ylabel('Percolation depth [m]')
plt.xlabel('Correlation length [m]')
plt.legend()
plt.ylim([3, 0])
plt.tight_layout()
plt.savefig(f'../Figures/correlation_lengths_amp.pdf',bbox_inches='tight', dpi = 300)

######################################################################
#Amp = 0.1, corr_length = 4000m
######################################################################
data = np.load('samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx4000.0m_ly1m_amp0.1_C0.03_50by200_t10368000.npz')
phi_i_sol =data['phi_i_sol']; porosity_change=np.transpose((phi_i_sol[:,-1]-phi_i_sol[:,0]).reshape(Grid.Nx,Grid.Ny))
final_ice_vol_frac = np.transpose((phi_i_sol[:,-1]).reshape(Grid.Nx,Grid.Ny))


Y_max_array = []

##find percolation depth at each Xc
for i in range(0,np.shape(porosity_change)[1]):
    Y_max= np.max(Yc[porosity_change[:,i]>0]) - np.min(Yc[final_ice_vol_frac[:,i]>=0.01])
    Y_max_array=np.append(Y_max_array,Y_max)

#rcParams.update({'font.size': 40})

plt.errorbar(4000, np.mean(Y_max_array),np.std(Y_max_array), linestyle='None', marker='^',color=green, mfc=green,mec=green, ms=20, mew=4)


######################################################################
#Without correlated random field
######################################################################
data = np.load('samimi_without_acc_new_C0.03_50by200_t10368000.npz')
phi_i_sol =data['phi_i_sol']; porosity_change=np.transpose((phi_i_sol[:,-1]-phi_i_sol[:,0]).reshape(Grid.Nx,Grid.Ny))
final_ice_vol_frac = np.transpose((phi_i_sol[:,-1]).reshape(Grid.Nx,Grid.Ny))


Y_max_array = []

##find percolation depth at each Xc
for i in range(0,np.shape(porosity_change)[1]):
    Y_max= np.max(Yc[porosity_change[:,i]>0]) - np.min(Yc[final_ice_vol_frac[:,i]>=0.01])
    Y_max_array=np.append(Y_max_array,Y_max)

#rcParams.update({'font.size': 40})

plt.hlines(Y_max_array[0], 40, 4000,color='k',linestyle='--',label='1D')



#plt.plot(,'bs',color=blue)

plt.ylabel('Penetration depth [m]')
plt.xlabel('Correlation length [m]')
plt.legend()
plt.ylim([2.5, 0])
plt.tight_layout()
plt.xscale('log')
plt.savefig(f'../Figures/correlation_lengths_amp_normalized.pdf',bbox_inches='tight', dpi = 600)








#Porosity
plt.figure(figsize=(3,6),dpi=200)
plt.plot(1-phi_i_sol[0:Grid.Ny,0],Yc[0:Grid.Ny],'k-',linewidth=2)
plt.ylim([Grid.ymax,Grid.ymin])
plt.ylim([3,0])
plt.xticks([0,0.3,0.6])
plt.ylabel('Depth, $z$ [m]')
plt.xlabel('$\phi_{meas}(z)$')
plt.tight_layout()
plt.savefig(f'../Figures/1D_porosity_profile.pdf',bbox_inches='tight', dpi = 300)


'''


'''





# ---------------------------------------------------------
# Make a 3×1 figure showing porosity three times
# ---------------------------------------------------------

fig, axes = plt.subplots(
    3, 1,
    figsize=(10, 12),
    dpi=120,
    sharex=True,
    gridspec_kw={"hspace": 0.15, "hspace": 0.0}
)

# setting font sizeto 30
plt.rcParams.update({'font.size': 25})


# If you want specific times, define:
# time_ind = [-1]
# Otherwise, repeat the same field:
time_ind = [-1,-1,-1]      # SAME porosity field three times

for i in range(3):

    ax = axes[i]
    if i ==0:
        ######################################################################
        #Amp = 0.05 corr_length = 40m
        ######################################################################
        data = np.load('samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx40.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz')
        phi_i_sol = data['phi_i_sol']; porosity = 1.-phi_i_sol
        poro_plot = porosity[:, time_ind[i]].reshape(Grid.Nx, Grid.Ny).T
        
    if i ==1:
        ######################################################################
        #Amp = 0.05 corr_length = 400m
        ######################################################################
        data = np.load('samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx400.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz')
        phi_i_sol = data['phi_i_sol']; porosity = 1.-phi_i_sol
        
        poro_plot = porosity[:, time_ind[i]].reshape(Grid.Nx, Grid.Ny).T
        
    if i ==2:
        ######################################################################
        #Amp = 0.05 corr_length = 4000m
        ######################################################################
        data = np.load('samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx4000.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz')
        phi_i_sol =data['phi_i_sol']; porosity = 1.-phi_i_sol
        
        poro_plot = porosity[:, time_ind[i]].reshape(Grid.Nx, Grid.Ny).T


    c = ax.contourf(
        Xc/1e3,
        Yc,
        poro_plot,
        cmap="Greys_r",
        levels=100,
        vmin=0,
        vmax=1
    )

    ax.set_ylim([2.6, Grid.ymin])
    ax.set_ylabel("z [m]")

    if i == 2:
        ax.set_xlabel("x [km]")

from matplotlib.colors import Normalize
# ---------------------------------------------------------
# One colorbar for all panels
# ---------------------------------------------------------
mm = plt.cm.ScalarMappable(
    cmap="Greys_r",
    norm=Normalize(vmin=0, vmax=1)
)
mm.set_array([])

cbar = fig.colorbar(
    mm,
    ax=axes,
    pad=0.02,
    aspect=25
)
cbar.set_label("Porosity ϕ")

plt.show()
plt.savefig(f'../Figures/porosity_profile_comparison_new.pdf',bbox_inches='tight', dpi = 200)

'''




'''
####################################################
# final
####################################################
# heterogeneous
red    = [190/255 ,  30/255, 45/255 ]
# 4 data columns + 1 colorbar column
# TOP ROW: Temperature + LOG Saturation
# BOTTOM ROW: Porosity
# Saturation colorbar has a lower arrow for sw < 1e-2
# Top row panels are labeled with month/date
# Axes share x/y across panels
####################################################
import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, ListedColormap
from matplotlib.gridspec import GridSpec
import datetime

# ----------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------
data = np.load(
    'samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx4000.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz'
)

phi_i_sol = data['phi_i_sol']
phi_w_sol = data['phi_w_sol']
T_sol     = data['T_sol']
time_sol  = data['t']

# ----------------------------------------------------------------------
# DERIVED FIELDS
# ----------------------------------------------------------------------
T_field  = T_sol - Tm
porosity = 1.0 - phi_i_sol

# saturation
sw_sol = phi_w_sol / np.maximum(porosity, 1e-12)

# limits
Tmin, Tmax = -22.0, 0.0
poro_min, poro_max = 0.0, 1.0

# saturation limits
sw_plot_min = 1e-2   # colorbar lower bound
sw_phys_min = 1e-10  # numerical/physical lower threshold for plotting
sw_max      = 1.0

# ----------------------------------------------------------------------
# MASK
# ----------------------------------------------------------------------
phi_sol = 1.0 - phi_i_sol
mask = phi_sol > (1.0 - non_porous_vol_frac)

# ----------------------------------------------------------------------
# TIMES
# ----------------------------------------------------------------------
time_ind = [0, 1935, 2300,-1]

start_date = datetime.datetime(2016, 5, 24)
time_labels = [
    (start_date + datetime.timedelta(seconds=float(time_sol[i]))).strftime("%m/%d")
    for i in time_ind
]

# ----------------------------------------------------------------------
# FIGURE
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(34, 8), dpi=120)

gs = GridSpec(
    nrows=2,
    ncols=5,
    width_ratios=[1, 1, 1, 1, 0.1],
    hspace=0.0,
    wspace=0.05
)

fig.subplots_adjust(top=0.90, right=0.88)

mask_cmap = ListedColormap(["#FFF6CC"])

# Make under-range saturation a very pale blue
cmap_sw = plt.cm.Blues.copy()
cmap_sw.set_under("#F7FBFF")

# ----------------------------------------------------------------------
# CREATE AXES WITH SHARED X/Y
# ----------------------------------------------------------------------
axes_top = []
axes_bottom = []

for j in range(4):

    # =======================
    # TOP: T + saturation
    # =======================
    if j == 0:
        ax = fig.add_subplot(gs[0, j])
        
    else:
        ax = fig.add_subplot(gs[0, j], sharex=axes_top[0], sharey=axes_top[0])

    axes_top.append(ax)
    ax.set_title(time_labels[j], fontsize=20, pad=10)

    mask_j = mask[:, time_ind[j]].reshape(Grid.Nx, Grid.Ny).T

    # mask background
    ax.contourf(
        Xc/1e3, Yc, mask_j.astype(float),
        levels=[0.5, 1.5], cmap=mask_cmap,
        linewidths=0
    )

    # temperature
    Tj = T_field[:, time_ind[j]].copy()
    Tj[mask[:, time_ind[j]]] = np.nan

    ax.contourf(
        Xc/1e3, Yc,
        Tj.reshape(Grid.Nx, Grid.Ny).T,
        cmap="Reds",
        levels=100,
        vmin=Tmin, vmax=Tmax
    )

    # saturation (log)
    swj = sw_sol[:, time_ind[j]].copy()
    swj[swj < sw_phys_min] = np.nan
    swj[swj > 1.0] = 1.0
    swj[mask[:, time_ind[j]]] = np.nan

    ax.contourf(
        Xc/1e3, Yc,
        swj.reshape(Grid.Nx, Grid.Ny).T,
        cmap=cmap_sw,
        levels=np.logspace(np.log10(sw_plot_min), np.log10(sw_max), 100),
        norm=LogNorm(vmin=sw_plot_min, vmax=sw_max),
        extend='min'
    )

    ax.set_ylim([2.6, Grid.ymin])

    if j == 0:
        ax.set_ylabel(r"$z$ [m]")
        rect = patches.Rectangle(
            (x_min, y_bottom),                # lower-left corner
            x_max - x_min,                   # width
            y_top - y_bottom,                # height
            linewidth=2,
            edgecolor='k',
            linestyle='--',
            facecolor='none'
        )
        
        ax.add_patch(rect)
        
    else:
        ax.tick_params(labelleft=False)

    # box limits (in same units as your axes → note Xc/1e3 means km)
    x_min, x_max = 400/1e3, 900/1e3
    y_top, y_bottom = 1.4, 0.92
    

    # =======================
    # BOTTOM: porosity
    # =======================
    if j == 0:
        ax = fig.add_subplot(gs[1, j], sharex=axes_top[0])
        
    else:
        ax = fig.add_subplot(gs[1, j], sharex=axes_top[0], sharey=axes_bottom[0])

    axes_bottom.append(ax)

    ax.contourf(
        Xc/1e3, Yc, mask_j.astype(float),
        levels=[0.5, 1.5], cmap=mask_cmap
    )

    poroj = porosity[:, time_ind[j]].copy()
    poroj[mask[:, time_ind[j]]] = np.nan

    ax.contourf(
        Xc/1e3, Yc,
        poroj.reshape(Grid.Nx, Grid.Ny).T,
        cmap="Greys_r",
        levels=100,
        vmin=poro_min, vmax=poro_max
    )

    ax.set_ylim([2.6, Grid.ymin])

    if j == 0:
        ax.set_ylabel(r"$z$ [m]")
        rect = patches.Rectangle(
            (x_min, y_bottom),                # lower-left corner
            x_max - x_min,                   # width
            y_top - y_bottom,                # height
            linewidth=2,
            edgecolor=red,
            linestyle='--',
            facecolor='none'
        )
        
        ax.add_patch(rect)
    else:
        ax.tick_params(labelleft=False)
        
    # box limits (in same units as your axes → note Xc/1e3 means km)
    x_min, x_max = 400/1e3, 900/1e3
    y_top, y_bottom = 1.4, 0.92
    

    ax.set_xlabel("x [km]")

# Hide top-row x tick labels for clean layout
for ax in axes_top:
    ax.tick_params(labelbottom=False)

# ----------------------------------------------------------------------
# COLORBARS
# ----------------------------------------------------------------------
cax_sw   = fig.add_subplot(gs[0, 4])
cax_T    = fig.add_subplot(gs[1, 4])
cax_poro = fig.add_axes([0.92, 0.15, 0.015, 0.7])

# saturation (log, with lower arrow)
m_sw = plt.cm.ScalarMappable(
    cmap=cmap_sw,
    norm=LogNorm(vmin=sw_plot_min, vmax=sw_max)
)
m_sw.set_array([])

cb_sw = fig.colorbar(
    m_sw,
    cax=cax_sw,
    extend='min',
    extendfrac=0.08
)
cb_sw.set_label(r"$s_w$")
cb_sw.set_ticks([1e-2, 1e-1, 1e0])
cb_sw.set_ticklabels([r"10$^{-2}$", r"10$^{-1}$", r"10$^{0}$"])

# temperature
m_T = plt.cm.ScalarMappable(
    cmap="Reds",
    norm=Normalize(vmin=Tmin, vmax=Tmax)
)
m_T.set_array([])

cb_T = fig.colorbar(m_T, cax=cax_T)
cb_T.set_label(r"$T$ [°C]")

# porosity
m_poro = plt.cm.ScalarMappable(
    cmap="Greys_r",
    norm=Normalize(vmin=poro_min, vmax=poro_max)
)
m_poro.set_array([])

cb_poro = fig.colorbar(m_poro, cax=cax_poro)
cb_poro.set_label(r"Porosity $\phi$")
cb_poro.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

# ----------------------------------------------------------------------
# SAVE
# ----------------------------------------------------------------------
plt.savefig(
    "../Figures/heterogeneous_LOG_saturation_T_phi_arrow_dates_sharedaxes.pdf",
    dpi=600,
    bbox_inches="tight"
)
plt.savefig(
    "../Figures/heterogeneous_LOG_saturation_T_phi_arrow_dates_sharedaxes.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()






import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.gridspec import GridSpec

plt.rcParams.update({'font.size': 20})

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
files = [
    'samimi_without_acc_new_0deg_slopen_ormalizedFalse_C0.03_50by200_t10368000.npz',
    'samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx40.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz',
    'samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx400.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz',
    'samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx4000.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz',
]

# time index for each panel (row-major)
time_ind = [0, -1, 0, -1, 0, -1, 0, -1]

poro_min, poro_max = 0.0, 1.0

mask_cmap = ListedColormap(["#FFF6CC"])

# ---------------------------------------------------------
# FIGURE + GRID
# ---------------------------------------------------------
fig = plt.figure(figsize=(18, 16), dpi=150)

gs = GridSpec(
    nrows=4,
    ncols=3,
    width_ratios=[1, 1, 0.05],
    hspace=0.0,
    wspace=0.0
)

panel = 0

# ---------------------------------------------------------
# PLOTTING LOOP
# ---------------------------------------------------------
for i, fname in enumerate(files):

    data = np.load(fname)
    phi_i_sol = data['phi_i_sol']

    porosity = 1.0 - phi_i_sol
    phi_sol  = 1.0 - phi_i_sol
    mask     = phi_sol > (1.0 - non_porous_vol_frac)

    for j in range(2):

        ax = fig.add_subplot(gs[i, j])

        tidx = time_ind[panel]
        panel += 1

        poro = porosity[:, tidx]
        mask_j = mask[:, tidx]

        poro_plot = poro.reshape(Grid.Nx, Grid.Ny).T
        mask_plot = mask_j.reshape(Grid.Nx, Grid.Ny).T

        # -------------------------------
        # MASK (yellow background)
        # -------------------------------
        ax.contourf(
            Xc / 1e3, Yc,
            mask_plot.astype(float),
            levels=[0.5, 1.5],
            cmap=mask_cmap,
            antialiased=False,
            linewidths=0,
            rasterized=True
        )

        # -------------------------------
        # POROSITY
        # -------------------------------
        poro_plot = poro_plot.copy()
        poro_plot[mask_plot] = np.nan

        ax.contourf(
            Xc / 1e3, Yc,
            poro_plot,
            cmap="Greys_r",
            levels=100,
            vmin=poro_min,
            vmax=poro_max,
            antialiased=False,
            linewidths=0,
            rasterized=True
        )

        # -------------------------------
        # AXIS LIMITS (CRITICAL FIX)
        # -------------------------------
        ax.set_xlim(0, 2.0)          # <-- FIX: applies to ALL panels
        ax.set_ylim(2.6, Grid.ymin)

        # -------------------------------
        # TICKS & LABELS
        # -------------------------------
        if j == 0:
            ax.set_ylabel("z [m]")
            ax.set_yticks([0, 1, 2])
        else:
            ax.set_yticklabels([])

        if i == len(files) - 1:
            ax.set_xlabel("x [km]")
            ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
        else:
            ax.set_xticklabels([])

# ---------------------------------------------------------
# COLORBAR (RIGHT)
# ---------------------------------------------------------
cax = fig.add_subplot(gs[:, 2])

m = plt.cm.ScalarMappable(
    cmap="Greys_r",
    norm=Normalize(vmin=poro_min, vmax=poro_max)
)
m.set_array([])

cb = fig.colorbar(m, cax=cax)
cb.set_label(r"Porosity $\phi$")
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

# ---------------------------------------------------------
# SAVE + SHOW
# ---------------------------------------------------------
plt.savefig(
    "../Figures/porosity_4x2_comparison.pdf",
    bbox_inches="tight"
)
plt.savefig(
    "../Figures/porosity_4x2_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()





import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.gridspec import GridSpec

plt.rcParams.update({'font.size': 20})

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
files = [
    'samimi_without_acc_new_0deg_slopen_ormalizedFalse_C0.03_50by200_t10368000.npz',
    'samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx40.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz',
    'samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx400.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz',
    'samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx4000.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz',
]

# one time per panel (row-major: left, right)
time_ind = [0, -1, 0, -1, 0, -1, 0, -1]

poro_min, poro_max = 0.0, 1.0

# light yellow mask
mask_cmap = ListedColormap(["#FFF6CC"])

# ---------------------------------------------------------
# FIGURE + GRID
# ---------------------------------------------------------
fig = plt.figure(figsize=(10,8), dpi=150)

gs = GridSpec(
    nrows=4,
    ncols=3,
    width_ratios=[1, 1, 0.05],
    hspace=0.0,
    wspace=0.0
)

panel = 0

# ---------------------------------------------------------
# PLOTTING LOOP
# ---------------------------------------------------------
for i, fname in enumerate(files):

    data = np.load(fname)
    phi_i_sol = data['phi_i_sol']

    porosity = 1.0 - phi_i_sol
    phi_sol  = 1.0 - phi_i_sol
    mask     = phi_sol > (1.0 - non_porous_vol_frac)

    for j in range(2):

        ax = fig.add_subplot(gs[i, j])

        tidx = time_ind[panel]
        panel += 1

        poro      = porosity[:, tidx]
        mask_j    = mask[:, tidx]

        poro_plot = poro.reshape(Grid.Nx, Grid.Ny).T
        mask_plot = mask_j.reshape(Grid.Nx, Grid.Ny).T

        # -------------------------------
        # MASK (yellow background)
        # -------------------------------
        ax.contourf(
            Xc / 1e3, Yc,
            mask_plot.astype(float),
            levels=[0.5, 1.5],
            cmap=mask_cmap,
            antialiased=False,
            linewidths=0,
            rasterized=True
        )

        # -------------------------------
        # POROSITY
        # -------------------------------
        poro_plot = poro_plot.copy()
        poro_plot[mask_plot] = np.nan

        ax.contourf(
            Xc / 1e3, Yc,
            poro_plot,
            levels=100,
            cmap="Greys_r",
            vmin=poro_min,
            vmax=poro_max,
            antialiased=False,
            linewidths=0,
            rasterized=True
        )

        # -------------------------------
        # AXIS LIMITS (FOR ALL PANELS)
        # -------------------------------
        ax.set_xlim(0, 2)
        ax.set_ylim([2.6, Grid.ymin])

        # -------------------------------
        # TICKS & LABELS
        # -------------------------------
        if j == 0:
            ax.set_ylabel("z [m]")
            ax.set_yticks([0, 1, 2])
        else:
            ax.set_yticklabels([])

        if i == len(files) - 1:
            ax.set_xlabel("x [km]")
            ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
        else:
            ax.set_xticklabels([])

# ---------------------------------------------------------
# COLORBAR (RIGHT)
# ---------------------------------------------------------
cax = fig.add_subplot(gs[:, 2])

m = plt.cm.ScalarMappable(
    cmap="Greys_r",
    norm=Normalize(vmin=poro_min, vmax=poro_max)
)
m.set_array([])

cb = fig.colorbar(m, cax=cax)
cb.set_label(r"Porosity $\phi$")
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

# ---------------------------------------------------------
# SAVE + SHOW
# ---------------------------------------------------------
plt.savefig(
    "../Figures/porosity_4x2_comparison.pdf",
    bbox_inches="tight"
)
plt.savefig(
    "../Figures/porosity_4x2_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.size': 20,
    'pdf.fonttype': 42,   # avoid Type 3 fonts
})

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
files = [
    'samimi_without_acc_new_0deg_slopen_ormalizedFalse_C0.03_50by200_t10368000.npz',
    'samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx40.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz',
    'samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx400.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz',
    'samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx4000.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz',
]

time_ind = [0, -1, 0, -1, 0, -1, 0, -1]
poro_min, poro_max = 0.0, 1.0

mask_cmap = ListedColormap(["#FFF6CC"])

# ---------------------------------------------------------
# FIGURE
# ---------------------------------------------------------
fig = plt.figure(figsize=(10, 8), dpi=300)

gs = GridSpec(
    nrows=4,
    ncols=3,
    width_ratios=[1, 1, 0.06],
    hspace=0.0,
    wspace=0.0
)

panel = 0

# ---------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------
for i, fname in enumerate(files):

    data = np.load(fname)
    phi_i_sol = data['phi_i_sol']

    porosity = 1.0 - phi_i_sol
    mask = porosity > (1.0 - non_porous_vol_frac)

    for j in range(2):

        ax = fig.add_subplot(gs[i, j])

        tidx = time_ind[panel]
        panel += 1

        poro = porosity[:, tidx].reshape(Grid.Nx, Grid.Ny).T
        mask_plot = mask[:, tidx].reshape(Grid.Nx, Grid.Ny).T

        # --- mask background
        cs_mask = ax.contourf(
            Xc / 1e3, Yc,
            mask_plot.astype(float),
            levels=[0.5, 1.5],
            cmap=mask_cmap,
            antialiased=False
        )
        cs_mask.set_rasterized(True)

        # --- porosity
        poro = poro.copy()
        poro[mask_plot] = np.nan

        cs = ax.contourf(
            Xc / 1e3, Yc,
            poro,
            levels=np.linspace(poro_min, poro_max, 101),
            cmap="Greys_r",
            vmin=poro_min,
            vmax=poro_max,
            antialiased=False
        )
        cs.set_rasterized(True)

        ax.set_xlim(0, 2)
        ax.set_ylim([2.6, Grid.ymin])

        if j == 0:
            ax.set_ylabel("z [m]")
            ax.set_yticks([0, 1, 2])
        else:
            ax.set_yticklabels([])

        if i == len(files) - 1:
            ax.set_xlabel("x [km]")
            ax.set_xticks([0, 0.5, 1, 1.5, 2])
        else:
            ax.set_xticklabels([])

# ---------------------------------------------------------
# COLORBAR
# ---------------------------------------------------------
cax = fig.add_subplot(gs[:, 2])

m = plt.cm.ScalarMappable(
    cmap="Greys_r",
    norm=Normalize(vmin=poro_min, vmax=poro_max)
)
m.set_array([])

cb = fig.colorbar(m, cax=cax)
cb.set_label(r"Porosity $\phi$")
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------
plt.savefig(
    "../Figures/porosity_4x2_comparison_final_corr_length.pdf",
    bbox_inches="tight"
)
plt.savefig(
    "../Figures/porosity_4x2_comparison_final_corr_length.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



#Amplitude
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.size': 20,
    'pdf.fonttype': 42,   # avoid Type 3 fonts
})

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
files = [
    'samimi_without_acc_new_0deg_slopen_ormalizedFalse_C0.03_50by200_t10368000.npz',
    'samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx40.0m_ly1m_amp0.01_C0.03_50by200_t10368000.npz',
    'samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx40.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz',
    'samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx40.0m_ly1m_amp0.1_C0.03_50by200_t10368000.npz',
]

time_ind = [0, -1, 0, -1, 0, -1, 0, -1]
poro_min, poro_max = 0.0, 1.0

mask_cmap = ListedColormap(["#FFF6CC"])

# ---------------------------------------------------------
# FIGURE
# ---------------------------------------------------------
fig = plt.figure(figsize=(10, 8), dpi=150)

gs = GridSpec(
    nrows=4,
    ncols=3,
    width_ratios=[1, 1, 0.06],
    hspace=0.0,
    wspace=0.0
)

panel = 0

# ---------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------
for i, fname in enumerate(files):

    data = np.load(fname)
    phi_i_sol = data['phi_i_sol']

    porosity = 1.0 - phi_i_sol
    mask = porosity > (1.0 - non_porous_vol_frac)

    for j in range(2):

        ax = fig.add_subplot(gs[i, j])

        tidx = time_ind[panel]
        panel += 1

        poro = porosity[:, tidx].reshape(Grid.Nx, Grid.Ny).T
        mask_plot = mask[:, tidx].reshape(Grid.Nx, Grid.Ny).T

        # --- mask background
        cs_mask = ax.contourf(
            Xc / 1e3, Yc,
            mask_plot.astype(float),
            levels=[0.5, 1.5],
            cmap=mask_cmap,
            antialiased=False
        )
        cs_mask.set_rasterized(True)

        # --- porosity
        poro = poro.copy()
        poro[mask_plot] = np.nan

        cs = ax.contourf(
            Xc / 1e3, Yc,
            poro,
            levels=np.linspace(poro_min, poro_max, 101),
            cmap="Greys_r",
            vmin=poro_min,
            vmax=poro_max,
            antialiased=False
        )
        cs.set_rasterized(True)

        ax.set_xlim(0, 2)
        ax.set_ylim([2.6, Grid.ymin])

        if j == 0:
            ax.set_ylabel("z [m]")
            ax.set_yticks([0, 1, 2])
        else:
            ax.set_yticklabels([])

        if i == len(files) - 1:
            ax.set_xlabel("x [km]")
            ax.set_xticks([0, 0.5, 1, 1.5, 2])
        else:
            ax.set_xticklabels([])

# ---------------------------------------------------------
# COLORBAR
# ---------------------------------------------------------
cax = fig.add_subplot(gs[:, 2])

m = plt.cm.ScalarMappable(
    cmap="Greys_r",
    norm=Normalize(vmin=poro_min, vmax=poro_max)
)
m.set_array([])

cb = fig.colorbar(m, cax=cax)
cb.set_label(r"Porosity $\phi$")
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------
plt.savefig(
    "../Figures/porosity_4x2_comparison_final_amplitude_40m.pdf",
    bbox_inches="tight"
)
plt.savefig(
    "../Figures/porosity_4x2_comparison_final_amplitude_40m.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.size': 20,
    'pdf.fonttype': 42,   # avoid Type 3 fonts
})

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
files = [
    'samimi_without_acc_new_0deg_slopen_ormalizedFalse_C0.03_50by200_t10368000.npz',
    'samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx400.0m_ly1m_amp0.01_C0.03_50by200_t10368000.npz',
    'samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx400.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz',
    'samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx400.0m_ly1m_amp0.1_C0.03_50by200_t10368000.npz',
]

time_ind = [0, -1, 0, -1, 0, -1, 0, -1]
poro_min, poro_max = 0.0, 1.0

mask_cmap = ListedColormap(["#FFF6CC"])

# ---------------------------------------------------------
# FIGURE
# ---------------------------------------------------------
fig = plt.figure(figsize=(10, 8), dpi=150)

gs = GridSpec(
    nrows=4,
    ncols=3,
    width_ratios=[1, 1, 0.06],
    hspace=0.0,
    wspace=0.0
)

panel = 0

# ---------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------
for i, fname in enumerate(files):

    data = np.load(fname)
    phi_i_sol = data['phi_i_sol']

    porosity = 1.0 - phi_i_sol
    mask = porosity > (1.0 - non_porous_vol_frac)

    for j in range(2):

        ax = fig.add_subplot(gs[i, j])

        tidx = time_ind[panel]
        panel += 1

        poro = porosity[:, tidx].reshape(Grid.Nx, Grid.Ny).T
        mask_plot = mask[:, tidx].reshape(Grid.Nx, Grid.Ny).T

        # --- mask background
        cs_mask = ax.contourf(
            Xc / 1e3, Yc,
            mask_plot.astype(float),
            levels=[0.5, 1.5],
            cmap=mask_cmap,
            antialiased=False
        )
        cs_mask.set_rasterized(True)

        # --- porosity
        poro = poro.copy()
        poro[mask_plot] = np.nan

        cs = ax.contourf(
            Xc / 1e3, Yc,
            poro,
            levels=np.linspace(poro_min, poro_max, 101),
            cmap="Greys_r",
            vmin=poro_min,
            vmax=poro_max,
            antialiased=False
        )
        cs.set_rasterized(True)

        ax.set_xlim(0, 2)
        ax.set_ylim([2.6, Grid.ymin])

        if j == 0:
            ax.set_ylabel("z [m]")
            ax.set_yticks([0, 1, 2])
        else:
            ax.set_yticklabels([])

        if i == len(files) - 1:
            ax.set_xlabel("x [km]")
            ax.set_xticks([0, 0.5, 1, 1.5, 2])
        else:
            ax.set_xticklabels([])

# ---------------------------------------------------------
# COLORBAR
# ---------------------------------------------------------
cax = fig.add_subplot(gs[:, 2])

m = plt.cm.ScalarMappable(
    cmap="Greys_r",
    norm=Normalize(vmin=poro_min, vmax=poro_max)
)
m.set_array([])

cb = fig.colorbar(m, cax=cax)
cb.set_label(r"Porosity $\phi$")
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------
plt.savefig(
    "../Figures/porosity_4x2_comparison_final_amplitude_400m.pdf",
    bbox_inches="tight"
)
plt.savefig(
    "../Figures/porosity_4x2_comparison_final_amplitude_400m.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

'''






# ############################################################
# Video with:
# (1) bigger time title
# (2) time shown in HOURS only (no minutes/seconds)
# (3) mask region filled SOLID light yellow (overlay drawn LAST)
# ############################################################

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation
from matplotlib.colors import ListedColormap
from datetime import datetime, timedelta
import pandas as pd

# -----------------------------
# Matplotlib style
# -----------------------------
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'font.family': 'Serif'})

# -----------------------------
# Backups
# -----------------------------
phi_w_sol_backup = phi_w_sol.copy()
phi_i_sol_backup = phi_i_sol.copy()
T_sol_backup     = T_sol.copy()

# -----------------------------
# Time handling (build datetime vector)
# -----------------------------
t = np.array(t)

data  = pd.read_csv('/Users/ms6985/Desktop/Research_new/meltwater-percolation/Infiltration/JPL/Samira-data/Achilig_upGPR_data.csv')
dates = data['date'].apply(lambda x: datetime.strptime(x, "'%d-%b-%Y %H:%M:%S.%f'"))

t_dates = [dates.iloc[850] + timedelta(seconds=float(i)) for i in t]
t_dates = np.array(t_dates, dtype='datetime64[ns]')  # shape (Nt,)

# -----------------------------
# Grid / plotting settings
# -----------------------------
Grid.ymax = 3
horizontal_units = 'km'  # 'm' or 'km'

# frame thinning
ii = int(tf / (4 * day2s))
if ii < 1:
    ii = 1

# Decimate arrays in time for animation
phi_w_anim = phi_w_sol_backup[:, ::ii]
phi_i_anim = phi_i_sol_backup[:, ::ii]
T_anim     = T_sol_backup[:, ::ii]
t_anim     = t_dates[::ii]

# frames
N, frn = np.shape(phi_w_anim)

# -----------------------------
# Mask definition
# Choose ONE of these mask definitions:
#   A) mask melted/non-porous: phi_i < non_porous_vol_frac
#   B) mask based on phi_sol = 1-phi_i threshold
# -----------------------------

# A) Melted / non-porous mask (matches your static plot equivalently)
# mask_anim = (phi_i_anim < non_porous_vol_frac)

# B) Your current choice
phi_sol_anim = 1.0 - phi_i_anim
mask_anim = (phi_sol_anim > 1-non_porous_vol_frac)  # adjust threshold as needed

# Include NaN regions too (often helpful)
mask_anim = mask_anim | np.isnan(phi_w_anim) | np.isnan(phi_i_anim) | np.isnan(T_anim)

# Mask colormap (solid light yellow)
mask_cmap = ListedColormap(["#FFF6CC"])

# clamp very cold temps if needed
T_anim = T_anim.copy()
T_anim[T_anim < (Tm - 30)] = (Tm - 30)

# -----------------------------
# Global color limits (nan-safe)
# -----------------------------
LWC_vmin = np.nanmin(phi_w_anim)
LWC_vmax = np.nanmax(phi_w_anim)

poro_anim = 1.0 - phi_i_anim
poro_vmin = np.nanmin(poro_anim)
poro_vmax = np.nanmax(poro_anim)

Tdiff_anim = T_anim - Tm
T_vmin = np.nanmin(Tdiff_anim)
T_vmax = np.nanmax(Tdiff_anim)

# -----------------------------
# Figure
# -----------------------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 10), dpi=100)

def _xcoords():
    return (Xc / 1e3) if horizontal_units == 'km' else Xc

def _xlim():
    return ([Grid.xmin/1e3, Grid.xmax/1e3] if horizontal_units == 'km' else [Grid.xmin, Grid.xmax])

# Colorbars (static mappables)
mLWC  = cm.ScalarMappable(cmap=cm.Blues);   mLWC.set_array([]);  mLWC.set_clim(LWC_vmin, LWC_vmax)
mPoro = cm.ScalarMappable(cmap=cm.Greys_r); mPoro.set_array([]); mPoro.set_clim(poro_vmin, poro_vmax)
mT    = cm.ScalarMappable(cmap=cm.Reds);    mT.set_array([]);    mT.set_clim(T_vmin, T_vmax)

# -----------------------------
# Draw frame (DATA first, MASK last)
# -----------------------------
def draw_frame(frame_number):
    ax1.clear(); ax2.clear(); ax3.clear()

    xplot = _xcoords()
    ax1.set_xlim(_xlim()); ax2.set_xlim(_xlim()); ax3.set_xlim(_xlim())

    # mask as (Ny, Nx)
    mask_j = mask_anim[:, frame_number].reshape(Grid.Nx, Grid.Ny).T

    # =======================
    # 1) LWC
    # =======================
    Z1 = phi_w_anim[:, frame_number].reshape(Grid.Nx, Grid.Ny).T
    ax1.contourf(
        xplot, Yc, Z1,
        cmap="Blues", levels=100,
        vmin=LWC_vmin, vmax=LWC_vmax,
        antialiased=False
    )

    # overlay SOLID yellow mask (draw LAST so it cannot be covered)
    ax1.contourf(
        xplot, Yc, mask_j.astype(int),
        levels=[0.5, 1.5],
        cmap=mask_cmap,
        antialiased=False
    )

    ax1.set_ylim([Grid.ymax, Grid.ymin])
    ax1.set_ylabel(r'$z$ [m]')

    # =======================
    # 2) Porosity
    # =======================
    Z2 = (1.0 - phi_i_anim[:, frame_number]).reshape(Grid.Nx, Grid.Ny).T
    ax2.contourf(
        xplot, Yc, Z2,
        cmap="Greys_r", levels=100,
        vmin=poro_vmin, vmax=poro_vmax,
        antialiased=False
    )

    ax2.contourf(
        xplot, Yc, mask_j.astype(int),
        levels=[0.5, 1.5],
        cmap=mask_cmap,
        antialiased=False
    )

    ax2.set_ylim([Grid.ymax, Grid.ymin])
    ax2.set_ylabel(r'$z$ [m]')

    # =======================
    # 3) Temperature (T - Tm)
    # =======================
    Z3 = (T_anim[:, frame_number] - Tm).reshape(Grid.Nx, Grid.Ny).T
    ax3.contourf(
        xplot, Yc, Z3,
        cmap="Reds", levels=100,
        vmin=T_vmin, vmax=T_vmax,
        antialiased=False
    )

    ax3.contourf(
        xplot, Yc, mask_j.astype(int),
        levels=[0.5, 1.5],
        cmap=mask_cmap,
        antialiased=False
    )

    ax3.set_ylim([Grid.ymax, Grid.ymin])
    ax3.set_ylabel(r'$z$ [m]')
    ax3.set_xlabel(r'$x$ [km]' if horizontal_units == 'km' else r'$x$ [m]')

# Draw first frame (so we can attach colorbars)
draw_frame(0)

# Colorbars (create once)
cb1 = fig.colorbar(mLWC,  ax=ax1, orientation='vertical', aspect=10, pad=0.01)
cb1.set_label(r'LWC')

cb2 = fig.colorbar(mPoro, ax=ax2, orientation='vertical', aspect=10, pad=0.01)
cb2.set_label(r'Porosity')

cb3 = fig.colorbar(mT,    ax=ax3, orientation='vertical', aspect=10, pad=0.01)
cb3.set_label(r'T $[^\circ C]$')

# -----------------------------
# Animation update function
# -----------------------------
def update_plot(frame_number):
    # HOURS ONLY
    title_str = pd.to_datetime(str(t_anim[frame_number])).strftime("%Y-%m-%d")
    fig.suptitle(title_str, fontsize=30)

    draw_frame(frame_number)
    return []

# -----------------------------
# Make animation
# -----------------------------
fps_anim = 30

ani = animation.FuncAnimation(
    fig,
    update_plot,
    frames=frn,
    interval=1,
    blit=False
)

# If you want to save:
ani.save(
     f"../Figures/{simulation_name}_{C_L/phi_L}_{Grid.Nx}by{Grid.Ny}_tf{t[0 + (frn-1)*ii]}_slope_{tilt_angle}deg_new_LWC.mov",
     writer='ffmpeg',
     fps=fps_anim
)



'''
#Colors
red    = [190/255 ,30/255 ,45/255 ]
blue   = [ 30/255 ,144/255 , 255/255 ]
green  = [  0/255 , 166/255 ,  81/255]
orange = [247/255 , 148/255 ,  30/255]
purple = [102/255 ,  45/255 , 145/255]
plt.rcParams.update({'font.family': "Serif"})
plt.rcParams.update({'font.size': 26})

######################################################################
#Amp = 0.01, corr_length = 40m
######################################################################
data = np.load('samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx40.0m_ly1m_amp0.01_C0.03_50by200_t10368000.npz')
phi_i_sol =data['phi_i_sol']; porosity_change=np.transpose((phi_i_sol[:,-1]-phi_i_sol[:,0]).reshape(Grid.Nx,Grid.Ny))

final_ice_vol_frac = np.transpose((phi_i_sol[:,-1]).reshape(Grid.Nx,Grid.Ny))


Y_max_array = []




##find percolation depth at each Xc
for i in range(0,np.shape(porosity_change)[1]):
    Y_max= np.max(Yc[porosity_change[:,i]>0]) - np.min(Yc[final_ice_vol_frac[:,i]>=0.01])
    Y_max_array=np.append(Y_max_array,Y_max)
    
#porosity_change[porosity_change>0]



fig, ax1 = plt.subplots(1,1, figsize=(8,8),dpi=100)
#rcParams.update({'font.size': 20})

plt.subplot(1,1,1)

#color=['red', 'green', 'blue', 'cyan', 'magenta'],
#       error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2)

plt.errorbar(40-10, np.mean(Y_max_array),np.std(Y_max_array), linestyle='None',color=red, marker='^', mfc=red,mec=red, ms=20, mew=4)

######################################################################
#Amp = 0.01, corr_length = 400m
######################################################################
data = np.load('samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx400.0m_ly1m_amp0.01_C0.03_50by200_t10368000.npz')
phi_i_sol =data['phi_i_sol']; porosity_change=np.transpose((phi_i_sol[:,-1]-phi_i_sol[:,0]).reshape(Grid.Nx,Grid.Ny))
final_ice_vol_frac = np.transpose((phi_i_sol[:,-1]).reshape(Grid.Nx,Grid.Ny))


Y_max_array = []

##find percolation depth at each Xc
for i in range(0,np.shape(porosity_change)[1]):
    Y_max= np.max(Yc[porosity_change[:,i]>0]) - np.min(Yc[final_ice_vol_frac[:,i]>=0.01])
    Y_max_array=np.append(Y_max_array,Y_max)

#rcParams.update({'font.size': 40})

plt.errorbar(400-100, np.mean(Y_max_array),np.std(Y_max_array), linestyle='None', marker='^',color=red, mfc=red,mec=red,label='Amp=0.01', ms=20, mew=4)


######################################################################
#Amp = 0.01, corr_length = 4000m
######################################################################
data = np.load('samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx4000.0m_ly1m_amp0.01_C0.03_50by200_t10368000.npz')
phi_i_sol =data['phi_i_sol']; porosity_change=np.transpose((phi_i_sol[:,-1]-phi_i_sol[:,0]).reshape(Grid.Nx,Grid.Ny))
final_ice_vol_frac = np.transpose((phi_i_sol[:,-1]).reshape(Grid.Nx,Grid.Ny))


Y_max_array = []

##find percolation depth at each Xc
for i in range(0,np.shape(porosity_change)[1]):
    Y_max= np.max(Yc[porosity_change[:,i]>0]) - np.min(Yc[final_ice_vol_frac[:,i]>=0.01])
    Y_max_array=np.append(Y_max_array,Y_max)

#rcParams.update({'font.size': 40})

plt.errorbar(4000-1000, np.mean(Y_max_array),np.std(Y_max_array), linestyle='None', marker='^',color=red, mfc=red,mec=red, ms=20, mew=4)


######################################################################
#Amp = 0.05, corr_length = 40m
######################################################################
data = np.load('samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx40.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz')
phi_i_sol =data['phi_i_sol']; porosity_change=np.transpose((phi_i_sol[:,-1]-phi_i_sol[:,0]).reshape(Grid.Nx,Grid.Ny))
final_ice_vol_frac = np.transpose((phi_i_sol[:,-1]).reshape(Grid.Nx,Grid.Ny))


Y_max_array = []

##find percolation depth at each Xc
for i in range(0,np.shape(porosity_change)[1]):
    Y_max= np.max(Yc[porosity_change[:,i]>0]) - np.min(Yc[final_ice_vol_frac[:,i]>=0.01])
    Y_max_array=np.append(Y_max_array,Y_max)

#rcParams.update({'font.size': 40})

plt.subplot(1,1,1)

#color=['red', 'green', 'blue', 'cyan', 'magenta'],
#       error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2)

plt.errorbar(40, np.mean(Y_max_array),np.std(Y_max_array), linestyle='None',color=blue, marker='o', mfc=blue,mec=blue, ms=20, mew=4)

######################################################################
#Amp = 0.05, corr_length = 400m
######################################################################
data = np.load('samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx400.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz')
phi_i_sol =data['phi_i_sol']; porosity_change=np.transpose((phi_i_sol[:,-1]-phi_i_sol[:,0]).reshape(Grid.Nx,Grid.Ny))
final_ice_vol_frac = np.transpose((phi_i_sol[:,-1]).reshape(Grid.Nx,Grid.Ny))


Y_max_array = []

##find percolation depth at each Xc
for i in range(0,np.shape(porosity_change)[1]):
    Y_max= np.max(Yc[porosity_change[:,i]>0]) - np.min(Yc[final_ice_vol_frac[:,i]>=0.01])
    Y_max_array=np.append(Y_max_array,Y_max)

#rcParams.update({'font.size': 40})

plt.errorbar(400, np.mean(Y_max_array),np.std(Y_max_array), linestyle='None', marker='o',color=blue, mfc=blue,mec=blue,label='Amp=0.05', ms=20, mew=4)


#plt.plot(,'bs',color=blue)

plt.ylabel('Percolation depth [m]')
plt.xlabel('Correlation length [m]')
plt.legend()
plt.ylim([3, 0])
plt.tight_layout()
plt.savefig(f'../Figures/correlation_lengths_amp.pdf',bbox_inches='tight', dpi = 300)

######################################################################
#Amp = 0.05, corr_length = 4000m
######################################################################
data = np.load('samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx4000.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz')
phi_i_sol =data['phi_i_sol']; porosity_change=np.transpose((phi_i_sol[:,-1]-phi_i_sol[:,0]).reshape(Grid.Nx,Grid.Ny))
final_ice_vol_frac = np.transpose((phi_i_sol[:,-1]).reshape(Grid.Nx,Grid.Ny))


Y_max_array = []

##find percolation depth at each Xc
for i in range(0,np.shape(porosity_change)[1]):
    Y_max= np.max(Yc[porosity_change[:,i]>0]) - np.min(Yc[final_ice_vol_frac[:,i]>=0.01])
    Y_max_array=np.append(Y_max_array,Y_max)

#rcParams.update({'font.size': 40})

plt.errorbar(4000, np.mean(Y_max_array),np.std(Y_max_array), linestyle='None', marker='o',color=blue, mfc=blue,mec=blue, ms=20, mew=4)



######################################################################
#Amp = 0.1, corr_length = 40m
######################################################################
data = np.load('samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx40.0m_ly1m_amp0.1_C0.03_50by200_t10368000.npz')
phi_i_sol =data['phi_i_sol']; porosity_change=np.transpose((phi_i_sol[:,-1]-phi_i_sol[:,0]).reshape(Grid.Nx,Grid.Ny))
final_ice_vol_frac = np.transpose((phi_i_sol[:,-1]).reshape(Grid.Nx,Grid.Ny))


Y_max_array = []

##find percolation depth at each Xc
for i in range(0,np.shape(porosity_change)[1]):
    Y_max= np.max(Yc[porosity_change[:,i]>0]) - np.min(Yc[final_ice_vol_frac[:,i]>=0.01])
    Y_max_array=np.append(Y_max_array,Y_max)

#rcParams.update({'font.size': 40})

plt.subplot(1,1,1)

#color=['red', 'green', 'blue', 'cyan', 'magenta'],
#       error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2)

plt.errorbar(40+10, np.mean(Y_max_array),np.std(Y_max_array), linestyle='None',color=green, marker='s', mfc=green,mec=green, ms=20, mew=4)

######################################################################
#Amp = 0.1, corr_length = 400m
######################################################################
data = np.load('samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx400.0m_ly1m_amp0.1_C0.03_50by200_t10368000.npz')
phi_i_sol =data['phi_i_sol']; porosity_change=np.transpose((phi_i_sol[:,-1]-phi_i_sol[:,0]).reshape(Grid.Nx,Grid.Ny))
final_ice_vol_frac = np.transpose((phi_i_sol[:,-1]).reshape(Grid.Nx,Grid.Ny))


Y_max_array = []

##find percolation depth at each Xc
for i in range(0,np.shape(porosity_change)[1]):
    Y_max= np.max(Yc[porosity_change[:,i]>0]) - np.min(Yc[final_ice_vol_frac[:,i]>=0.01])
    Y_max_array=np.append(Y_max_array,Y_max)

#rcParams.update({'font.size': 40})

plt.errorbar(400+100, np.mean(Y_max_array),np.std(Y_max_array), linestyle='None', marker='s',color=green, mfc=green,mec=green,label='Amp=0.1', ms=20, mew=4)


#plt.plot(,'bs',color=blue)

plt.ylabel('Percolation depth [m]')
plt.xlabel('Correlation length [m]')
plt.legend()
plt.ylim([3, 0])
plt.tight_layout()
plt.savefig(f'../Figures/correlation_lengths_amp.pdf',bbox_inches='tight', dpi = 300)

######################################################################
#Amp = 0.1, corr_length = 4000m
######################################################################
data = np.load('samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx4000.0m_ly1m_amp0.1_C0.03_50by200_t10368000.npz')
phi_i_sol =data['phi_i_sol']; porosity_change=np.transpose((phi_i_sol[:,-1]-phi_i_sol[:,0]).reshape(Grid.Nx,Grid.Ny))
final_ice_vol_frac = np.transpose((phi_i_sol[:,-1]).reshape(Grid.Nx,Grid.Ny))


Y_max_array = []

##find percolation depth at each Xc
for i in range(0,np.shape(porosity_change)[1]):
    Y_max= np.max(Yc[porosity_change[:,i]>0]) - np.min(Yc[final_ice_vol_frac[:,i]>=0.01])
    Y_max_array=np.append(Y_max_array,Y_max)

#rcParams.update({'font.size': 40})

plt.errorbar(4000+1000, np.mean(Y_max_array),np.std(Y_max_array), linestyle='None', marker='s',color=green, mfc=green,mec=green, ms=20, mew=4)


######################################################################
#Without correlated random field
######################################################################
data = np.load('samimi_without_acc_new_C0.03_50by200_t10368000.npz')
phi_i_sol =data['phi_i_sol']; porosity_change=np.transpose((phi_i_sol[:,-1]-phi_i_sol[:,0]).reshape(Grid.Nx,Grid.Ny))
final_ice_vol_frac = np.transpose((phi_i_sol[:,-1]).reshape(Grid.Nx,Grid.Ny))


Y_max_array = []

##find percolation depth at each Xc
for i in range(0,np.shape(porosity_change)[1]):
    Y_max= np.max(Yc[porosity_change[:,i]>0]) - np.min(Yc[final_ice_vol_frac[:,i]>=0.01])
    Y_max_array=np.append(Y_max_array,Y_max)

#rcParams.update({'font.size': 40})

plt.hlines(Y_max_array[0], 30, 5000,color='k',linestyle='--',label='1D')



#plt.plot(,'bs',color=blue)

plt.ylabel('Penetration depth [m]')
plt.xlabel('Correlation length [m]')
plt.legend()
plt.ylim([2.5, 0])
plt.tight_layout()
plt.xscale('log')
plt.savefig(f'../Figures/correlation_lengths_amp_normalized_offset.pdf',bbox_inches='tight', dpi = 600)


'''








# =========================================================
# Static 3x4 multiphysics figure (not zoomed)
# Rows:
#   1) Porosity
#   2) Combined Temperature + Saturation
#   3) Classification
#
# Colorbars:
#   Row 1 -> phi
#   Row 2 -> T
#   Row 3 -> sw (log scale, arrow below 1e-2)
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.ticker import FormatStrFormatter
import datetime
import pandas as pd

plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'font.family': 'Serif'})


# ----------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------
data = np.load(
    'samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx4000.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz'
)

phi_i_sol = data['phi_i_sol']
phi_w_sol = data['phi_w_sol']
T_sol     = data['T_sol']
t  = data['t']

# -------------------------------------------------
# Build grid
# -------------------------------------------------
Grid = build_grid(Grid)
[D, G, I] = build_ops(Grid)

# -------------------------------------------------
# Derived fields
# -------------------------------------------------
phi_sol = 1.0 - phi_i_sol

# choose times
times = [0, 1935, 2300,-1]

# -------------------------------------------------
# Dates
# -------------------------------------------------
date_csv_path = "/Users/ms6985/Desktop/Research_new/meltwater-percolation/Infiltration/JPL/Samira-data/Achilig_upGPR_data.csv"
date_start_index = 850

data_dates = pd.read_csv(date_csv_path)
dates = data_dates['date'].apply(lambda x: datetime.datetime.strptime(x, "'%d-%b-%Y %H:%M:%S.%f'"))

time_labels = [
    (dates.iloc[date_start_index] + datetime.timedelta(seconds=float(t[i]))).strftime("%m/%d")
    for i in times
]

print([dates.iloc[date_start_index] + datetime.timedelta(seconds=float(t[i])) for i in times])

# -------------------------------------------------
# Plot scales
# -------------------------------------------------
sw_vmin, sw_vmax = 1e-2, 1.0
sw_phys_min = 1e-10

phi_vmin = float(np.nanmin(phi_sol))
phi_vmax = float(np.nanmax(phi_sol))

T_anom = T_sol - Tm
T_vmin = -22
T_vmax = float(np.nanmax(T_anom))

ticks_sw = [1e-2, 1e-1, 1e0]
ticks_phi = [phi_vmin, 0.5 * (phi_vmin + phi_vmax), phi_vmax]
ticks_T = [T_vmin, 0.5 * (T_vmin + T_vmax), T_vmax]

# -------------------------------------------------
# Mask settings
# -------------------------------------------------
mask_cmap = ListedColormap(["#FFF6CC"])
mask_threshold = 1.0 - non_porous_vol_frac

cmap_sw = plt.cm.Blues.copy()
cmap_sw.set_under("#F7FBFF")

# -------------------------------------------------
# Figure
# -------------------------------------------------
fig = plt.figure(figsize=(14, 9), dpi=120)

# 4 data columns + 1 narrow colorbar column
gs = fig.add_gridspec(
    nrows=3,
    ncols=5,
    width_ratios=[1, 1, 1, 1, 0.08],
    hspace=0.0,
    wspace=0.05
)

fig.subplots_adjust(top=0.90, right=0.88)

# -------------------------------------------------
# ScalarMappables for colorbars
# -------------------------------------------------
mm_phi = cm.ScalarMappable(
    cmap="Greys_r",
    norm=plt.Normalize(vmin=phi_vmin, vmax=phi_vmax)
)
mm_phi.set_array([])

mm_T = cm.ScalarMappable(
    cmap="Reds",
    norm=plt.Normalize(vmin=T_vmin, vmax=T_vmax)
)
mm_T.set_array([])

mm_sw = cm.ScalarMappable(
    cmap=cmap_sw,
    norm=LogNorm(vmin=sw_vmin, vmax=sw_vmax)
)
mm_sw.set_array([])

# -------------------------------------------------
# Shared axes containers
# -------------------------------------------------
axes_row1 = []
axes_row2 = []
axes_row3 = []

# -------------------------------------------------
# Plot panels
# -------------------------------------------------
for j, tt in enumerate(times):

    # =================================================
    # Common fields
    # =================================================
    phi_now = phi_sol[:, tt]
    phi_w_now = phi_w_sol[:, tt]
    phi_i_now = phi_i_sol[:, tt]
    T_now = T_anom[:, tt]

    # mask based on near-surface porous removal criterion
    mask_grid = (phi_now > mask_threshold).reshape(Grid.Nx, Grid.Ny).T
    mask_bool = mask_grid.astype(bool)
    mask_grid_int = mask_grid.astype(int)

    phi_grid = phi_now.reshape(Grid.Nx, Grid.Ny).T
    T_grid = T_now.reshape(Grid.Nx, Grid.Ny).T

    # safe saturation
    sw_cell = np.divide(
        phi_w_now,
        phi_now * (1.0 - s_gr),
        out=np.full_like(phi_w_now, np.nan, dtype=float),
        where=(phi_now * (1.0 - s_gr)) > 1e-12
    )
    sw_grid = sw_cell.reshape(Grid.Nx, Grid.Ny).T

    # -------------------------------------------------
    # Saturated and ice-layer cell sets
    # -------------------------------------------------
    dof_inact = Grid.dof[sw_cell >= sat_threshold]
    dof_ice_layer = Grid.dof[phi_i_now >= phi_i_threshold]

    dof_f_saturated = find_faces(dof_inact, D, Grid)
    dof_sat_faces = find_all_faces(dof_inact, D, Grid)

    X_sat, Y_sat = comp_face_coords(dof_sat_faces, Grid)
    X_bnd, Y_bnd = comp_face_coords(dof_f_saturated, Grid)
    X_all, Y_all = comp_face_coords(Grid.dof_f, Grid)

    dof_ice_layer_faces = find_all_faces(dof_ice_layer, D, Grid)
    X_ice, Y_ice = comp_face_coords(dof_ice_layer_faces, Grid)
    dof_f_ice = find_faces(dof_ice_layer, D, Grid)
    X_ice_bnd, Y_ice_bnd = comp_face_coords(dof_f_ice, Grid)

    # =================================================
    # Row 1: Porosity
    # =================================================
    if j == 0:
        ax = fig.add_subplot(gs[1, j])
    else:
        ax = fig.add_subplot(gs[1, j], sharex=axes_row1[0], sharey=axes_row1[0])
    axes_row1.append(ax)

    ax.set_title(time_labels[j], fontsize=20, pad=10)

    phi_plot = np.where(mask_bool, np.nan, phi_grid)

    ax.pcolormesh(
        Xc, Yc, phi_plot,
        cmap="Greys_r",
        vmin=phi_vmin,
        vmax=phi_vmax,
        shading="nearest"
    )

    ax.contourf(
        Xc, Yc, mask_grid_int,
        levels=[0.5, 1.5],
        cmap=mask_cmap,
        antialiased=False
    )

    ax.set_xlim([400, 900])
    ax.set_ylim([1.4,0.92])
    ax.plot(X_all, Y_all, "k-", linewidth=0.4)
    if j == 0:
        ax.set_ylabel(r"$z$ [m]")
    else:
        ax.tick_params(labelleft=False)

    # =================================================
    # Row 2: Combined T + sw
    # =================================================
    if j == 0:
        ax = fig.add_subplot(gs[0, j], sharex=axes_row1[0], sharey=axes_row1[0])
    else:
        ax = fig.add_subplot(gs[0, j], sharex=axes_row1[0], sharey=axes_row2[0])
    axes_row2.append(ax)

    ax.contourf(
        Xc, Yc, mask_grid_int,
        levels=[0.5, 1.5],
        cmap=mask_cmap,
        antialiased=False
    )
    
    T_plot = np.where(mask_bool, np.nan, T_grid)

    ax.contourf(
        Xc, Yc, T_plot,
        cmap="Reds",
        vmin=T_vmin, vmax=T_vmax,
        levels=100
    )

    sw_overlay = np.where((sw_grid >= sw_phys_min) & (~mask_bool), sw_grid, np.nan)

    ax.pcolormesh(
        Xc, Yc, sw_overlay,
        cmap=cmap_sw,
        norm=LogNorm(vmin=sw_vmin, vmax=sw_vmax),
        shading="nearest"
    )
    
    
    ax.plot(X_all, Y_all, "k-", linewidth=0.4)
    
    
    
    ax.set_xlim([400, 900])
    ax.set_ylim([1.4,0.92])





    if j == 0:
        ax.set_ylabel(r"$z$ [m]")
    else:
        ax.tick_params(labelleft=False)

    # =================================================
    # Row 3: Classification
    # =================================================
    if j == 0:
        ax = fig.add_subplot(gs[2, j], sharex=axes_row1[0], sharey=axes_row1[0])
    else:
        ax = fig.add_subplot(gs[2, j], sharex=axes_row1[0], sharey=axes_row3[0])
    axes_row3.append(ax)

    ax.plot(X_all, Y_all, "k-", linewidth=0.4)
    ax.plot(X_sat, Y_sat, "r-", linewidth=0.4)
    ax.plot(X_bnd, Y_bnd, "r-", linewidth=2.0)
    ax.plot(X_ice, Y_ice, "b-", linewidth=0.4)
    ax.plot(X_ice_bnd, Y_ice_bnd, "b-", linewidth=2.0)

    if len(dof_inact) > 0:
        ax.plot(
            Xc_col[dof_inact - 1, 0],
            Yc_col[dof_inact - 1, 0],
            "ro",
            markerfacecolor="None",
            markersize=2
        )

    if len(dof_ice_layer) > 0:
        ax.plot(
            Xc_col[dof_ice_layer - 1, 0],
            Yc_col[dof_ice_layer - 1, 0],
            "bo",
            markerfacecolor="None",
            markersize=2
        )

    ax.set_xlim([400, 900])
    ax.set_ylim([1.4,0.92])

    if j == 0:
        ax.set_ylabel(r"$z$ [m]")
    else:
        ax.tick_params(labelleft=False)

    ax.set_xlabel(r"$x$ [m]")

# -------------------------------------------------
# Axis formatting
# -------------------------------------------------
#yticks = [0, 1, 2]

for r, row_axes in enumerate([axes_row1, axes_row2, axes_row3]):
    for ax in row_axes:
        #ax.set_yticks(yticks)
        ax.tick_params(
            axis='both',
            direction='inout',
            length=6,
            width=1
        )

        if r != 2:
            ax.tick_params(labelbottom=False)
        else:
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(30)
                lbl.set_ha("right")

# -------------------------------------------------
# Colorbars (ORDER: sw → phi → T)
# -------------------------------------------------

# 1️⃣ Saturation FIRST
cbar_sw = fig.colorbar(
    mm_sw,
    ax=axes_row2,   # sw is plotted in combined row
    fraction=0.025,
    pad=0.03,
    extend='min',
    extendfrac=0.08
)
cbar_sw.set_ticks(ticks_sw)
cbar_sw.set_label(r"$s_w$", rotation=0, labelpad=10)
cbar_sw.set_ticklabels([r"10$^{-2}$", r"10$^{-1}$", r"10$^{0}$"])


# 2️⃣ Porosity SECOND
cbar_phi = fig.colorbar(
    mm_phi,
    ax=axes_row1,
    fraction=0.025,
    pad=0.03
)
cbar_phi.set_ticks(ticks_phi)
cbar_phi.set_label(r"$\phi$", rotation=0, labelpad=10)
cbar_phi.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


# 3️⃣ Temperature THIRD
cbar_T = fig.colorbar(
    mm_T,
    ax=axes_row3,   # or axes_row2 if you want it tied to T visually
    fraction=0.025,
    pad=0.03
)
cbar_T.set_ticks(ticks_T)
cbar_T.set_label(r"$T$ [$^\circ$C]", rotation=0, labelpad=10)
cbar_T.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

# -------------------------------------------------
# Save figure
# -------------------------------------------------
plt.savefig(
    f"../Figures/samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx4000.0m_ly1m_amp0.05_C0.03_50by200_t10368000_3x4_phi_Tsw_classification_not_zoomed_restyled.pdf",
    bbox_inches="tight",
    dpi=200
)
plt.savefig(
    f"../Figures/samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx4000.0m_ly1m_amp0.05_C0.03_50by200_t10368000_3x4_phi_Tsw_classification_not_zoomed_restyled.png",
    bbox_inches="tight",
    dpi=600
)

plt.show()



####################################################
# final
####################################################
# heterogeneous
red    = [190/255 ,  30/255, 45/255 ]
# 4 data columns + 1 colorbar column
# TOP ROW: Temperature + LOG Saturation
# BOTTOM ROW: Porosity
# Saturation colorbar has a lower arrow for sw < 1e-2
# Top row panels are labeled with month/date
# Axes share x/y across panels
####################################################
import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, ListedColormap
from matplotlib.gridspec import GridSpec
import datetime

# ----------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------
data = np.load(
    'samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx4000.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz'
)

phi_i_sol = data['phi_i_sol']
phi_w_sol = data['phi_w_sol']
T_sol     = data['T_sol']
t  = data['t']

# ----------------------------------------------------------------------
# DERIVED FIELDS
# ----------------------------------------------------------------------
T_field  = T_sol - Tm
porosity = 1.0 - phi_i_sol

# saturation
sw_sol = phi_w_sol / np.maximum(porosity, 1e-12)

# limits
Tmin, Tmax = -22.0, 0.0
poro_min, poro_max = 0.0, 1.0

# saturation limits
sw_plot_min = 1e-2   # colorbar lower bound
sw_phys_min = 1e-10  # numerical/physical lower threshold for plotting
sw_max      = 1.0

# ----------------------------------------------------------------------
# MASK
# ----------------------------------------------------------------------
phi_sol = 1.0 - phi_i_sol
mask = phi_sol > (1.0 - non_porous_vol_frac)

# ----------------------------------------------------------------------
# TIMES
# ----------------------------------------------------------------------
time_ind = [0, 1935, 2300,-1]

start_date = datetime.datetime(2016, 5, 24)
time_labels = [
    (start_date + datetime.timedelta(seconds=float(time_sol[i]))).strftime("%m/%d")
    for i in time_ind
]

# ----------------------------------------------------------------------
# FIGURE
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(34, 8), dpi=120)

gs = GridSpec(
    nrows=2,
    ncols=5,
    width_ratios=[1, 1, 1, 1, 0.1],
    hspace=0.0,
    wspace=0.05
)

fig.subplots_adjust(top=0.90, right=0.88)

mask_cmap = ListedColormap(["#FFF6CC"])

# Make under-range saturation a very pale blue
cmap_sw = plt.cm.Blues.copy()
cmap_sw.set_under("#F7FBFF")

# ----------------------------------------------------------------------
# CREATE AXES WITH SHARED X/Y
# ----------------------------------------------------------------------
axes_top = []
axes_bottom = []

for j in range(4):
    # box limits (in same units as your axes → note Xc/1e3 means km)
    x_min, x_max = 400/1e3, 900/1e3
    y_top, y_bottom = 1.4, 0.92
    
    # =======================
    # TOP: T + saturation
    # =======================
    if j == 0:
        ax = fig.add_subplot(gs[0, j])
        
    else:
        ax = fig.add_subplot(gs[0, j], sharex=axes_top[0], sharey=axes_top[0])

    axes_top.append(ax)
    ax.set_title(time_labels[j], fontsize=20, pad=10)

    mask_j = mask[:, time_ind[j]].reshape(Grid.Nx, Grid.Ny).T

    # mask background
    ax.contourf(
        Xc/1e3, Yc, mask_j.astype(float),
        levels=[0.5, 1.5], cmap=mask_cmap,
        linewidths=0
    )

    # temperature
    Tj = T_field[:, time_ind[j]].copy()
    Tj[mask[:, time_ind[j]]] = np.nan

    ax.contourf(
        Xc/1e3, Yc,
        Tj.reshape(Grid.Nx, Grid.Ny).T,
        cmap="Reds",
        levels=100,
        vmin=Tmin, vmax=Tmax
    )

    # saturation (log)
    swj = sw_sol[:, time_ind[j]].copy()
    swj[swj < sw_phys_min] = np.nan
    swj[swj > 1.0] = 1.0
    swj[mask[:, time_ind[j]]] = np.nan

    ax.contourf(
        Xc/1e3, Yc,
        swj.reshape(Grid.Nx, Grid.Ny).T,
        cmap=cmap_sw,
        levels=np.logspace(np.log10(sw_plot_min), np.log10(sw_max), 100),
        norm=LogNorm(vmin=sw_plot_min, vmax=sw_max),
        extend='min'
    )

    ax.set_ylim([2.6, Grid.ymin])

    if j == 0:
        ax.set_ylabel(r"$z$ [m]")
        rect = patches.Rectangle(
            (x_min, y_bottom),                # lower-left corner
            x_max - x_min,                   # width
            y_top - y_bottom,                # height
            linewidth=2,
            edgecolor='k',
            linestyle='--',
            facecolor='none'
        )
        
        ax.add_patch(rect)
        
    else:
        ax.tick_params(labelleft=False)


    # =======================
    # BOTTOM: porosity
    # =======================
    if j == 0:
        ax = fig.add_subplot(gs[1, j], sharex=axes_top[0])
        
    else:
        ax = fig.add_subplot(gs[1, j], sharex=axes_top[0], sharey=axes_bottom[0])

    axes_bottom.append(ax)

    ax.contourf(
        Xc/1e3, Yc, mask_j.astype(float),
        levels=[0.5, 1.5], cmap=mask_cmap
    )

    poroj = porosity[:, time_ind[j]].copy()
    poroj[mask[:, time_ind[j]]] = np.nan

    ax.contourf(
        Xc/1e3, Yc,
        poroj.reshape(Grid.Nx, Grid.Ny).T,
        cmap="Greys_r",
        levels=100,
        vmin=poro_min, vmax=poro_max
    )

    ax.set_ylim([2.6, Grid.ymin])

    if j == 0:
        ax.set_ylabel(r"$z$ [m]")
        rect = patches.Rectangle(
            (x_min, y_bottom),                # lower-left corner
            x_max - x_min,                   # width
            y_top - y_bottom,                # height
            linewidth=2,
            edgecolor=red,
            linestyle='--',
            facecolor='none'
        )
        
        ax.add_patch(rect)
    else:
        ax.tick_params(labelleft=False)
        
    # box limits (in same units as your axes → note Xc/1e3 means km)
    x_min, x_max = 400/1e3, 900/1e3
    y_top, y_bottom = 1.4, 0.92
    

    ax.set_xlabel("x [km]")

# Hide top-row x tick labels for clean layout
for ax in axes_top:
    ax.tick_params(labelbottom=False)

# ----------------------------------------------------------------------
# COLORBARS
# ----------------------------------------------------------------------
cax_sw   = fig.add_subplot(gs[0, 4])
cax_T    = fig.add_subplot(gs[1, 4])
cax_poro = fig.add_axes([0.92, 0.15, 0.015, 0.7])

# saturation (log, with lower arrow)
m_sw = plt.cm.ScalarMappable(
    cmap=cmap_sw,
    norm=LogNorm(vmin=sw_plot_min, vmax=sw_max)
)
m_sw.set_array([])

cb_sw = fig.colorbar(
    m_sw,
    cax=cax_sw,
    extend='min',
    extendfrac=0.08
)
cb_sw.set_label(r"$s_w$")
cb_sw.set_ticks([1e-2, 1e-1, 1e0])
cb_sw.set_ticklabels([r"10$^{-2}$", r"10$^{-1}$", r"10$^{0}$"])

# temperature
m_T = plt.cm.ScalarMappable(
    cmap="Reds",
    norm=Normalize(vmin=Tmin, vmax=Tmax)
)
m_T.set_array([])

cb_T = fig.colorbar(m_T, cax=cax_T)
cb_T.set_label(r"$T$ [°C]")

# porosity
m_poro = plt.cm.ScalarMappable(
    cmap="Greys_r",
    norm=Normalize(vmin=poro_min, vmax=poro_max)
)
m_poro.set_array([])

cb_poro = fig.colorbar(m_poro, cax=cax_poro)
cb_poro.set_label(r"Porosity $\phi$")
cb_poro.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

# ----------------------------------------------------------------------
# SAVE
# ----------------------------------------------------------------------
plt.savefig(
    "../Figures/heterogeneous_LOG_saturation_T_phi_arrow_dates_sharedaxes.pdf",
    dpi=600,
    bbox_inches="tight"
)
plt.savefig(
    "../Figures/heterogeneous_LOG_saturation_T_phi_arrow_dates_sharedaxes.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()




####################################################
# final (pcolormesh version)
####################################################
import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, ListedColormap
from matplotlib.gridspec import GridSpec
import datetime

# ----------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------
data = np.load(
    'samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx4000.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz'
)

phi_i_sol = data['phi_i_sol']
phi_w_sol = data['phi_w_sol']
T_sol     = data['T_sol']
t         = data['t']

# ----------------------------------------------------------------------
# DERIVED FIELDS
# ----------------------------------------------------------------------
T_field  = T_sol - Tm
porosity = 1.0 - phi_i_sol

sw_sol = phi_w_sol / np.maximum(porosity, 1e-12)

Tmin, Tmax = -22.0, 0.0
poro_min, poro_max = 0.0, 1.0

sw_plot_min = 1e-2
sw_phys_min = 1e-10
sw_max      = 1.0

# ----------------------------------------------------------------------
# MASK
# ----------------------------------------------------------------------
phi_sol = 1.0 - phi_i_sol
mask = phi_sol > (1.0 - non_porous_vol_frac)

# ----------------------------------------------------------------------
# TIMES
# ----------------------------------------------------------------------
time_ind = [0, 1935, 2300, -1]

start_date = datetime.datetime(2016, 5, 24)
time_labels = [
    (start_date + datetime.timedelta(seconds=float(t[i]))).strftime("%m/%d")
    for i in time_ind
]

# ----------------------------------------------------------------------
# FIGURE
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(34, 8), dpi=120)

gs = GridSpec(
    nrows=2,
    ncols=5,
    width_ratios=[1, 1, 1, 1, 0.1],
    hspace=0.0,
    wspace=0.05
)

fig.subplots_adjust(top=0.90, right=0.88)

mask_cmap = ListedColormap(["#FFF6CC"])

cmap_sw = plt.cm.Blues.copy()
cmap_sw.set_under("#F7FBFF")

axes_top = []
axes_bottom = []

# ----------------------------------------------------------------------
# LOOP PANELS
# ----------------------------------------------------------------------
for j in range(4):

    x_min, x_max = 400/1e3, 900/1e3
    y_top, y_bottom = 1.4, 0.92

    # =======================
    # TOP ROW
    # =======================
    if j == 0:
        ax = fig.add_subplot(gs[0, j])
    else:
        ax = fig.add_subplot(gs[0, j], sharex=axes_top[0], sharey=axes_top[0])

    axes_top.append(ax)
    ax.set_title(time_labels[j], fontsize=20, pad=10)

    mask_j = mask[:, time_ind[j]].reshape(Grid.Nx, Grid.Ny).T

    # --- mask ---
    ax.pcolormesh(
        Xc/1e3, Yc,
        mask_j.astype(float),
        cmap=mask_cmap,
        shading='auto'
    )

    # --- temperature ---
    Tj = T_field[:, time_ind[j]].copy()
    Tj[mask[:, time_ind[j]]] = np.nan
    Tj_plot = Tj.reshape(Grid.Nx, Grid.Ny).T

    ax.pcolormesh(
        Xc/1e3, Yc,
        Tj_plot,
        cmap="Reds",
        vmin=Tmin, vmax=Tmax,
        shading='auto'
    )

    # --- saturation ---
    swj = sw_sol[:, time_ind[j]].copy()
    swj[swj < sw_phys_min] = np.nan
    swj[swj > 1.0] = 1.0
    swj[mask[:, time_ind[j]]] = np.nan
    swj_plot = swj.reshape(Grid.Nx, Grid.Ny).T
    swj_plot[swj_plot < sw_plot_min] = np.nan

    ax.pcolormesh(
        Xc/1e3, Yc,
        swj_plot,
        cmap=cmap_sw,
        norm=LogNorm(vmin=sw_plot_min, vmax=sw_max),
        shading='auto'
    )

    ax.set_ylim([2.6, Grid.ymin])

    if j == 0:
        ax.set_ylabel(r"$z$ [m]")
        rect = patches.Rectangle(
            (x_min, y_bottom),
            x_max - x_min,
            y_top - y_bottom,
            linewidth=2,
            edgecolor='k',
            linestyle='--',
            facecolor='none'
        )
        ax.add_patch(rect)
    else:
        ax.tick_params(labelleft=False)

    # =======================
    # BOTTOM ROW (porosity)
    # =======================
    if j == 0:
        ax = fig.add_subplot(gs[1, j], sharex=axes_top[0])
    else:
        ax = fig.add_subplot(gs[1, j], sharex=axes_top[0], sharey=axes_bottom[0])

    axes_bottom.append(ax)

    # --- mask ---
    ax.pcolormesh(
        Xc/1e3, Yc,
        mask_j.astype(float),
        cmap=mask_cmap,
        shading='auto'
    )

    # --- porosity ---
    poroj = porosity[:, time_ind[j]].copy()
    poroj[mask[:, time_ind[j]]] = np.nan
    poroj_plot = poroj.reshape(Grid.Nx, Grid.Ny).T

    ax.pcolormesh(
        Xc/1e3, Yc,
        poroj_plot,
        cmap="Greys_r",
        vmin=poro_min, vmax=poro_max,
        shading='auto'
    )

    ax.set_ylim([2.6, Grid.ymin])

    if j == 0:
        ax.set_ylabel(r"$z$ [m]")
        rect = patches.Rectangle(
            (x_min, y_bottom),
            x_max - x_min,
            y_top - y_bottom,
            linewidth=2,
            edgecolor=red,
            linestyle='--',
            facecolor='none'
        )
        ax.add_patch(rect)
    else:
        ax.tick_params(labelleft=False)

    ax.set_xlabel("x [km]")

# hide top x labels
for ax in axes_top:
    ax.tick_params(labelbottom=False)
    
    # ------------------------------------------------------
# SET X TICKS (shared across all panels)
# ------------------------------------------------------
xticks = [0, 0.5, 1.0, 1.5]

for ax in axes_bottom:
    ax.set_xticks(xticks)

# ----------------------------------------------------------------------
# COLORBARS
# ----------------------------------------------------------------------
cax_sw   = fig.add_subplot(gs[0, 4])
cax_T    = fig.add_subplot(gs[1, 4])
cax_poro = fig.add_axes([0.92, 0.15, 0.015, 0.7])

m_sw = plt.cm.ScalarMappable(
    cmap=cmap_sw,
    norm=LogNorm(vmin=sw_plot_min, vmax=sw_max)
)
cb_sw = fig.colorbar(m_sw, cax=cax_sw, extend='min')
cb_sw.set_label(r"$s_w$")
cb_sw.set_ticks([1e-2, 1e-1, 1e0])
cb_sw.set_ticklabels([r"10$^{-2}$", r"10$^{-1}$", r"10$^{0}$"])

m_T = plt.cm.ScalarMappable(
    cmap="Reds",
    norm=Normalize(vmin=Tmin, vmax=Tmax)
)
cb_T = fig.colorbar(m_T, cax=cax_T)
cb_T.set_label(r"$T$ [°C]")

m_poro = plt.cm.ScalarMappable(
    cmap="Greys_r",
    norm=Normalize(vmin=poro_min, vmax=poro_max)
)
cb_poro = fig.colorbar(m_poro, cax=cax_poro)
cb_poro.set_label(r"Porosity $\phi$")

# ----------------------------------------------------------------------
# SAVE
# ----------------------------------------------------------------------
plt.savefig(
    "../Figures/heterogeneous_LOG_saturation_T_phi_arrow_dates_sharedaxepcolormesh.pdf",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "../Figures/heterogeneous_LOG_saturation_T_phi_arrow_dates_sharedaxepcolormesh.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()







######################################################################
# Load data 1D result
######################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.dates as mdates
from matplotlib import cm, rcParams

data = np.load('2D_SamiraA_withSamimiflux_lowres_absolute_k0_onetimes_phiexp1.885_cond__C0.03_50by200_t10368000.npz')
t = data['t']
phi_w_sol = data['phi_w_sol']
phi_i_sol = data['phi_i_sol']
H_sol = data['H_sol']
T_sol = data['T_sol']
s_w_sol = data['s_w_sol']

phi = data['phi']
Xc = data['Xc']
Yc = data['Yc']
Xc_col = data['Xc_col']
Yc_col = data['Yc_col']
Grid.Nx = data['Grid_Nx']
Grid.Ny = data['Grid_Ny']
Grid.xc = data['Grid_xc']
Grid.yc = data['Grid_yc']
Grid.xf = data['Grid_xf']
Grid.yf = data['Grid_yf']

phi_sol = 1 - phi_i_sol
phi_w_sol_backup = phi_w_sol

######################################################################
# External data
######################################################################
df = pd.read_excel('./Samira-data/TDRA_MaytoSept2016.xlsx', header=(1))

data_upgpr = pd.read_csv(
    '/Users/ms6985/Desktop/Research_new/meltwater-percolation/Infiltration/JPL/Samira-data/Achilig_upGPR_data.csv'
)
dates = data_upgpr['date'].apply(
    lambda x: datetime.strptime(x, "'%d-%b-%Y %H:%M:%S.%f'")
)

######################################################################
# Plot styling
######################################################################
rcParams.update({'font.size': 22})

######################################################################
# Constants / masks
######################################################################
Grid.ymax = 2.6

phi_sol = 1 - phi_i_sol
tday = t / day2s
depth_array = np.kron(np.ones(len(tday)), np.transpose([Grid.yc]))


######################################################################
# Time arrays
######################################################################
t_dates = np.array([dates[850] + timedelta(seconds=float(i)) for i in t])
t_dates = np.array([[np.datetime64(i) for i in t_dates]])
t_dates = np.repeat(t_dates, Grid.Ny, axis=0)

######################################################################
# Extract center column
######################################################################
mid = int(Grid.Nx * Grid.Ny / 2)

phi_w_array = phi_w_sol_backup[mid:mid + Grid.Ny, :]
phi_i_array = phi_i_sol[mid:mid + Grid.Ny, :]
T_array = T_sol[mid:mid + Grid.Ny, :]

T_array[T_array > Tm] = Tm
T_array[T_array < Tm - 22] = Tm - 22

# Porosity and saturation on center column
phi_array = 1 - phi_i_array
sw_array = np.divide(
    phi_w_array,
    phi_array * (1 - s_gr),
    out=np.full_like(phi_w_array, np.nan),
    where=(phi_array * (1 - s_gr)) > 1e-12
)
sw_array[sw_array < 1e-10] = np.nan
sw_array[sw_array > 1.0] = 1.0

######################################################################
# Yellow mask for non-porous region
######################################################################
mask_1d_bool = phi_array > 1 - non_porous_vol_frac
mask_1d_plot = np.where(mask_1d_bool, 1.0, np.nan)

mask_cmap = ListedColormap(["#FFF6CC"])

# hide masked values from plotted fields so yellow mask shows through
T_array[mask_1d_bool] = np.nan
sw_array[mask_1d_bool] = np.nan
phi_array_plot = phi_array.copy()
phi_array_plot[mask_1d_bool] = np.nan

######################################################################
# Vertical dashed lines ONLY at t[times]
######################################################################
times = [0, 1935, 2300, -1]
line_dates = [dates[850] + timedelta(seconds=float(t[i])) for i in times]

print(line_dates)

def add_vertical_lines(ax):
    for d in line_dates:
        ax.axvline(d, color='k', linestyle='--', linewidth=1)

######################################################################
# Uniform x ticks between 05/24 and 09/21 (5 ticks, keep endpoints)
######################################################################
x_start = datetime(2016, 5, 24)
x_end   = datetime(2016, 9, 21)

n_uniform_ticks = 5
uniform_tick_nums = np.linspace(mdates.date2num(x_start), mdates.date2num(x_end), n_uniform_ticks)
uniform_tick_dates = mdates.num2date(uniform_tick_nums)
uniform_tick_labels = [d.strftime('%m/%d') for d in uniform_tick_dates]

######################################################################
# Figure with explicit colorbar columns so panel widths match
######################################################################
fig = plt.figure(figsize=(15, 10), dpi=50)

gs = fig.add_gridspec(
    nrows=3,
    ncols=2,
    width_ratios=[1.0, 0.045],
    height_ratios=[1, 1, 1],
    wspace=0.08,
    hspace=0.0
)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # T + sw
ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)  # phi

# colorbars: top sw, middle T, bottom phi
cax1 = fig.add_subplot(gs[0, 1])  # sw
cax2 = fig.add_subplot(gs[1, 1])  # T
cax3 = fig.add_subplot(gs[2, 1])  # phi

###############################################################################
# PANEL 1 — Qnet + Accumulation
###############################################################################
plt.sca(ax1)

ax1.plot(
    [dates[850] + timedelta(days=i) for i in day],
    Qnet,
    'r-',
    color=red,
    linewidth=2
)

# Fixed y-limits/ticks
ax1.set_ylim([-25, 80])
ax1.set_yticks([-20, 0, 20, 40, 60])

ax1.set_ylabel(r'Q$_{net}$ [W/m$^2$]', color=red)

# saturation colorbar on panel 1
mm_sw = plt.cm.ScalarMappable(
    cmap="Blues",
    norm=LogNorm(vmin=1e-2, vmax=1.0)
)
mm_sw.set_array([])

clb_sw = fig.colorbar(
    mm_sw,
    cax=cax1,
    orientation='vertical',
    extend='min'
)
clb_sw.set_label(r'$s_w$')
clb_sw.set_ticks([1e-2, 1e-1, 1.0])
clb_sw.set_ticklabels([r'10$^{-2}$', r'10$^{-1}$', r'10$^{0}$'])

ax1.spines['left'].set_color(red)
ax1.tick_params(axis='y', colors=red)

ax1new = ax1.twinx()
ax1new.spines['right'].set_color(blue)
ax1new.set_ylabel(r'$a$ [mm.w.e./day]', color=blue)
ax1new.plot(
    [dates[850] + timedelta(days=i) for i in day],
    RetMIP_acc_rate(day) * 1e3 * day2s,
    color=blue
)
ax1new.tick_params(axis='y', colors=blue)
ax1new.set_yticks([0, 5, 10, 15])

add_vertical_lines(ax1)
add_vertical_lines(ax1new)

###############################################################################
# PANEL 2 — TEMPERATURE FIELD + SATURATION OVERLAY
###############################################################################
plt.sca(ax2)

# Temperature below melting only
T_array2 = T_array.copy()
T_array2[T_array2 >= Tm] = np.nan

# yellow mask first
ax2.contourf(
    t_dates,
    depth_array,
    mask_1d_plot,
    levels=[0.5, 1.5],
    cmap=mask_cmap
)

ax2.contourf(
    t_dates,
    depth_array,
    T_array2 - Tm,
    cmap="Reds",
    levels=100
)

mm_T = plt.cm.ScalarMappable(cmap=cm.Reds)
mm_T.set_array(np.linspace(
    np.nanmin(T_array2 - Tm),
    np.nanmax(T_array2 - Tm),
    10000
))

ax2.set_ylabel("z [m]")
ax2.set_ylim([Grid.ymax, Grid.ymin])

clb_T = fig.colorbar(mm_T, cax=cax2, orientation='vertical')
clb_T.set_label(r'T[$^\circ$C]')

# Saturation overlay: log scale from 1e-2 to 1 with lower arrow
ax2.contourf(
    t_dates,
    depth_array,
    sw_array,
    cmap="Blues",
    levels=np.logspace(-2, 0, 100),
    norm=LogNorm(vmin=1e-2, vmax=1.0),
    extend='min'
)

# upGPR line
ax2.plot(
    dates[850:4200],
    data_upgpr['height'][850] - data_upgpr['height'][850:4200],
    'k-'
)

add_vertical_lines(ax2)

###############################################################################
# PANEL 3 — POROSITY
###############################################################################
plt.sca(ax3)

# yellow mask first
ax3.contourf(
    t_dates,
    depth_array,
    mask_1d_plot,
    levels=[0.5, 1.5],
    cmap=mask_cmap
)

ax3.contourf(
    t_dates,
    depth_array,
    phi_array_plot,
    cmap="Greys_r",
    levels=100
)

mm_phi = plt.cm.ScalarMappable(cmap=cm.Greys_r)
mm_phi.set_array(np.linspace(
    np.nanmin(phi_array_plot),
    np.nanmax(phi_array_plot),
    10000
))

ax3.set_ylabel("z [m]")
ax3.set_ylim([Grid.ymax, Grid.ymin])

clb_phi = fig.colorbar(mm_phi, cax=cax3, orientation='vertical')
clb_phi.set_label(r'$\phi$')

add_vertical_lines(ax3)

###############################################################################
# X-AXIS FORMATTING
###############################################################################
ax3.set_xlabel("Calendar date")

for ax in [ax1, ax2, ax3]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_xlim([x_start, x_end])

# uniformly spaced ticks only
ax3.set_xticks(uniform_tick_dates)
ax3.set_xticklabels(uniform_tick_labels, rotation=0)

# hide upper x tick labels
ax1.tick_params(labelbottom=False)
ax2.tick_params(labelbottom=False)

plt.tight_layout()
plt.subplots_adjust(wspace=0.08, hspace=0.0)

plt.savefig(
    f'../Figures/1D{simulation_name}_{Grid.Nx}by{Grid.Ny}_Qnet_TplusSw_phi_uniform_ticks_lines_only_at_times.pdf',
    bbox_inches='tight',
    dpi=50
)

plt.savefig(
    f'../Figures/1D{simulation_name}_{Grid.Nx}by{Grid.Ny}_Qnet_TplusSw_phi_uniform_ticks_lines_only_at_times.png',
    bbox_inches="tight",
    dpi=600
)

plt.show()







####################################################
# VIDEO VERSION with 3 panels:
#   1) T + sw
#   2) phi
#   3) classification
# Colorbars:
#   sw, T, phi
#
# Fixes:
# - subpanels do not drift with frames
# - uses fixed axes + fixed colorbars
# - skips 10 time steps each frame
####################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, ListedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
import datetime

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
data = np.load(
    'samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx4000.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz'
)

phi_i_sol = data['phi_i_sol']
phi_w_sol = data['phi_w_sol']
T_sol     = data['T_sol']
t         = data['t']

# --------------------------------------------------
# GRID / OPS
# --------------------------------------------------
Grid = build_grid(Grid)
[D, G, I] = build_ops(Grid)

# --------------------------------------------------
# DERIVED FIELDS
# --------------------------------------------------
T_field  = T_sol - Tm
porosity = 1.0 - phi_i_sol
phi_sol  = porosity.copy()

sw_sol = np.divide(
    phi_w_sol,
    np.maximum(porosity * (1.0 - s_gr), 1e-12),
    out=np.full_like(phi_w_sol, np.nan, dtype=float),
    where=np.maximum(porosity * (1.0 - s_gr), 1e-12) > 0
)

# --------------------------------------------------
# LIMITS
# --------------------------------------------------
Tmin, Tmax = -22.0, 0.0
poro_min, poro_max = 0.0, 1.0

sw_plot_min = 1e-2
sw_phys_min = 1e-10
sw_max      = 1.0

# --------------------------------------------------
# MASK
# --------------------------------------------------
mask = phi_sol > (1.0 - non_porous_vol_frac)

mask_cmap = ListedColormap(["#FFF6CC"])
cmap_sw = plt.cm.Blues.copy()
cmap_sw.set_under("#F7FBFF")

# --------------------------------------------------
# TIME SETTINGS
# skip 10 time steps each frame
# --------------------------------------------------
frame_stride = 10
time_indices = np.arange(0, len(t), frame_stride)
if time_indices[-1] != len(t) - 1:
    time_indices = np.append(time_indices, len(t) - 1)

start_date = datetime.datetime(2016, 5, 24)

def make_time_label(tt):
    return (start_date + datetime.timedelta(seconds=float(t[tt]))).strftime("%m/%d/%y")

# --------------------------------------------------
# FIGURE SETUP
# Use fixed axes and fixed colorbar axes.
# Avoid tight_layout/subplots_adjust after animation starts.
# --------------------------------------------------
fig = plt.figure(figsize=(16, 8), dpi=120)
gs = GridSpec(
    3, 2,
    width_ratios=[1.0, 0.05],
    height_ratios=[1, 1, 1],
    left=0.08, right=0.92, bottom=0.08, top=0.92,
    hspace=0.06, wspace=0.10
)

ax_top = fig.add_subplot(gs[0, 0])
ax_mid = fig.add_subplot(gs[1, 0], sharex=ax_top, sharey=ax_top)
ax_bot = fig.add_subplot(gs[2, 0], sharex=ax_top, sharey=ax_top)

cax_sw  = fig.add_subplot(gs[0, 1])
cax_T   = fig.add_subplot(gs[1, 1])
cax_phi = fig.add_subplot(gs[2, 1])

# --------------------------------------------------
# FIXED COLORBARS
# --------------------------------------------------
m_sw = plt.cm.ScalarMappable(
    cmap=cmap_sw,
    norm=LogNorm(vmin=sw_plot_min, vmax=sw_max)
)
m_sw.set_array([])

cb_sw = fig.colorbar(
    m_sw,
    cax=cax_sw,
    orientation='vertical',
    extend='min'
)
cb_sw.set_label(r"$s_w$")
cb_sw.set_ticks([1e-2, 1e-1, 1e0])
cb_sw.set_ticklabels([r"10$^{-2}$", r"10$^{-1}$", r"10$^{0}$"])

m_T = plt.cm.ScalarMappable(
    cmap="Reds",
    norm=Normalize(vmin=Tmin, vmax=Tmax)
)
m_T.set_array([])

cb_T = fig.colorbar(
    m_T,
    cax=cax_T,
    orientation='vertical'
)
cb_T.set_label(r"$T$ [$^\circ$C]")

m_phi = plt.cm.ScalarMappable(
    cmap="Greys_r",
    norm=Normalize(vmin=poro_min, vmax=poro_max)
)
m_phi.set_array([])

cb_phi = fig.colorbar(
    m_phi,
    cax=cax_phi,
    orientation='vertical'
)
cb_phi.set_label(r"Porosity $\phi$")

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def style_axis(ax, show_xlabel=False, show_ylabel=False):
    ax.set_xlim([Xc.min()/1e3, Xc.max()/1e3])
    ax.set_ylim([2.6, Grid.ymin])
    if show_ylabel:
        ax.set_ylabel("z [m]")
    if show_xlabel:
        ax.set_xlabel("x [km]")
    ax.tick_params(axis='both', direction='inout', length=6, width=1)

# --------------------------------------------------
# ANIMATION FUNCTION
# --------------------------------------------------
def update(frame):

    tt = time_indices[frame]

    # Clear only plotting axes
    for ax in [ax_top, ax_mid, ax_bot]:
        ax.cla()

    label = make_time_label(tt)
    ax_top.set_title(label)

    mask_j = mask[:, tt].reshape(Grid.Nx, Grid.Ny).T

    # ==================================================
    # TOP: T + sw
    # ==================================================
    ax_top.contourf(
        Xc / 1e3, Yc, mask_j.astype(float),
        levels=[0.5, 1.5],
        cmap=mask_cmap
    )

    Tj = T_field[:, tt].copy()
    Tj[mask[:, tt]] = np.nan

    ax_top.contourf(
        Xc / 1e3, Yc,
        Tj.reshape(Grid.Nx, Grid.Ny).T,
        cmap="Reds",
        levels=100,
        vmin=Tmin,
        vmax=Tmax
    )

    swj = sw_sol[:, tt].copy()
    swj[swj < sw_phys_min] = np.nan
    swj[swj > 1.0] = 1.0
    swj[mask[:, tt]] = np.nan

    ax_top.contourf(
        Xc / 1e3, Yc,
        swj.reshape(Grid.Nx, Grid.Ny).T,
        cmap=cmap_sw,
        levels=np.logspace(-2, 0, 100),
        norm=LogNorm(vmin=sw_plot_min, vmax=sw_max),
        extend='min'
    )

    style_axis(ax_top, show_xlabel=False, show_ylabel=True)
    ax_top.tick_params(labelbottom=False)

    # ==================================================
    # MIDDLE: phi
    # ==================================================
    ax_mid.contourf(
        Xc / 1e3, Yc,
        mask_j.astype(float),
        levels=[0.5, 1.5],
        cmap=mask_cmap
    )

    poroj = porosity[:, tt].copy()
    poroj[mask[:, tt]] = np.nan

    ax_mid.contourf(
        Xc / 1e3, Yc,
        poroj.reshape(Grid.Nx, Grid.Ny).T,
        cmap="Greys_r",
        levels=100,
        vmin=poro_min,
        vmax=poro_max
    )

    style_axis(ax_mid, show_xlabel=False, show_ylabel=True)
    ax_mid.tick_params(labelbottom=False)

    # ==================================================
    # BOTTOM: classification
    # ==================================================
    phi_now   = phi_sol[:, tt]
    phi_w_now = phi_w_sol[:, tt]
    phi_i_now = phi_i_sol[:, tt]

    sw_cell = np.divide(
        phi_w_now,
        np.maximum(phi_now * (1.0 - s_gr), 1e-12),
        out=np.full_like(phi_w_now, np.nan, dtype=float),
        where=np.maximum(phi_now * (1.0 - s_gr), 1e-12) > 0
    )

    dof_inact = Grid.dof[np.isfinite(sw_cell) & (sw_cell >= sat_threshold)]
    dof_ice_layer = Grid.dof[phi_i_now >= phi_i_threshold]

    dof_f_saturated = find_faces(dof_inact, D, Grid)
    dof_sat_faces   = find_all_faces(dof_inact, D, Grid)

    X_sat, Y_sat = comp_face_coords(dof_sat_faces, Grid)
    X_bnd, Y_bnd = comp_face_coords(dof_f_saturated, Grid)
    X_all, Y_all = comp_face_coords(Grid.dof_f, Grid)

    dof_ice_layer_faces = find_all_faces(dof_ice_layer, D, Grid)
    X_ice, Y_ice = comp_face_coords(dof_ice_layer_faces, Grid)
    dof_f_ice = find_faces(dof_ice_layer, D, Grid)
    X_ice_bnd, Y_ice_bnd = comp_face_coords(dof_f_ice, Grid)

    ax_bot.plot(X_all / 1e3, Y_all, "k-", linewidth=0.4)
    ax_bot.plot(X_sat / 1e3, Y_sat, "r-", linewidth=0.4)
    ax_bot.plot(X_bnd / 1e3, Y_bnd, "r-", linewidth=2.0)
    ax_bot.plot(X_ice / 1e3, Y_ice, "b-", linewidth=0.4)
    ax_bot.plot(X_ice_bnd / 1e3, Y_ice_bnd, "b-", linewidth=2.0)

    if len(dof_inact) > 0:
        ax_bot.plot(
            Xc_col[dof_inact - 1, 0] / 1e3,
            Yc_col[dof_inact - 1, 0],
            "ro",
            markerfacecolor="None",
            markersize=2
        )

    if len(dof_ice_layer) > 0:
        ax_bot.plot(
            Xc_col[dof_ice_layer - 1, 0] / 1e3,
            Yc_col[dof_ice_layer - 1, 0],
            "bo",
            markerfacecolor="None",
            markersize=2
        )

    style_axis(ax_bot, show_xlabel=True, show_ylabel=True)

    return []

# --------------------------------------------------
# CREATE VIDEO
# --------------------------------------------------
ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(time_indices),
    interval=100,
    blit=False
)

# --------------------------------------------------
# SAVE
# --------------------------------------------------
Writer = animation.writers['ffmpeg']
writer = Writer(
    fps=10,
    metadata=dict(artist='OpenAI'),
    bitrate=1800,
    extra_args=[
        '-vf', 'pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2',
        '-pix_fmt', 'yuv420p'
    ]
)

ani.save(
    "../Figures/firn_video_4000m_with_classification.mp4",
    writer=writer,
    dpi=200
)

plt.close(fig)





####################################################
# VIDEO VERSION with 3 panels: (pcolormesh)
#   1) T + sw
#   2) phi
#   3) classification
# Colorbars:
#   sw, T, phi
####################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, ListedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
import datetime

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
data = np.load(
    'samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx4000.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz'
)

phi_i_sol = data['phi_i_sol']
phi_w_sol = data['phi_w_sol']
T_sol     = data['T_sol']
t         = data['t']

# --------------------------------------------------
# GRID / OPS
# --------------------------------------------------
Grid = build_grid(Grid)
[D, G, I] = build_ops(Grid)

# --------------------------------------------------
# DERIVED FIELDS
# --------------------------------------------------
T_field  = T_sol - Tm
porosity = 1.0 - phi_i_sol
phi_sol  = porosity.copy()

sw_sol = np.divide(
    phi_w_sol,
    np.maximum(porosity * (1.0 - s_gr), 1e-12),
    out=np.full_like(phi_w_sol, np.nan, dtype=float),
    where=np.maximum(porosity * (1.0 - s_gr), 1e-12) > 0
)

# --------------------------------------------------
# LIMITS
# --------------------------------------------------
Tmin, Tmax = -22.0, 0.0
poro_min, poro_max = 0.0, 1.0

sw_plot_min = 1e-2
sw_phys_min = 1e-10
sw_max      = 1.0

# --------------------------------------------------
# MASK
# --------------------------------------------------
mask = phi_sol > (1.0 - non_porous_vol_frac)

mask_cmap = ListedColormap(["#FFF6CC"])
cmap_sw = plt.cm.Blues.copy()
cmap_sw.set_under("#F7FBFF")

# --------------------------------------------------
# TIME SETTINGS
# --------------------------------------------------
frame_stride = 10
time_indices = np.arange(0, len(t), frame_stride)
if time_indices[-1] != len(t) - 1:
    time_indices = np.append(time_indices, len(t) - 1)

start_date = datetime.datetime(2016, 5, 24)

def make_time_label(tt):
    return (start_date + datetime.timedelta(seconds=float(t[tt]))).strftime("%m/%d/%y")

# --------------------------------------------------
# FIGURE SETUP
# --------------------------------------------------
fig = plt.figure(figsize=(16, 8), dpi=120)
gs = GridSpec(
    3, 2,
    width_ratios=[1.0, 0.05],
    height_ratios=[1, 1, 1],
    left=0.08, right=0.92, bottom=0.08, top=0.92,
    hspace=0.06, wspace=0.10
)

ax_top = fig.add_subplot(gs[0, 0])
ax_mid = fig.add_subplot(gs[1, 0], sharex=ax_top, sharey=ax_top)
ax_bot = fig.add_subplot(gs[2, 0], sharex=ax_top, sharey=ax_top)

cax_sw  = fig.add_subplot(gs[0, 1])
cax_T   = fig.add_subplot(gs[1, 1])
cax_phi = fig.add_subplot(gs[2, 1])

# --------------------------------------------------
# FIXED COLORBARS
# --------------------------------------------------
m_sw = plt.cm.ScalarMappable(
    cmap=cmap_sw,
    norm=LogNorm(vmin=sw_plot_min, vmax=sw_max)
)
m_sw.set_array([])

cb_sw = fig.colorbar(
    m_sw,
    cax=cax_sw,
    orientation='vertical',
    extend='min'
)
cb_sw.set_label(r"$s_w$")
cb_sw.set_ticks([1e-2, 1e-1, 1e0])
cb_sw.set_ticklabels([r"10$^{-2}$", r"10$^{-1}$", r"10$^{0}$"])

m_T = plt.cm.ScalarMappable(
    cmap="Reds",
    norm=Normalize(vmin=Tmin, vmax=Tmax)
)
m_T.set_array([])

cb_T = fig.colorbar(
    m_T,
    cax=cax_T,
    orientation='vertical'
)
cb_T.set_label(r"$T$ [$^\circ$C]")

m_phi = plt.cm.ScalarMappable(
    cmap="Greys_r",
    norm=Normalize(vmin=poro_min, vmax=poro_max)
)
m_phi.set_array([])

cb_phi = fig.colorbar(
    m_phi,
    cax=cax_phi,
    orientation='vertical'
)
cb_phi.set_label(r"Porosity $\phi$")

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def style_axis(ax, show_xlabel=False, show_ylabel=False):
    ax.set_xlim([Xc.min() / 1e3, Xc.max() / 1e3])
    ax.set_ylim([2.6, Grid.ymin])
    if show_ylabel:
        ax.set_ylabel("z [m]")
    if show_xlabel:
        ax.set_xlabel("x [km]")
    ax.tick_params(axis='both', direction='inout', length=6, width=1)

# --------------------------------------------------
# ANIMATION FUNCTION
# --------------------------------------------------
def update(frame):

    tt = time_indices[frame]

    # clear plotting axes only
    for ax in [ax_top, ax_mid, ax_bot]:
        ax.cla()

    label = make_time_label(tt)
    ax_top.set_title(label)

    mask_j = mask[:, tt].reshape(Grid.Nx, Grid.Ny).T

    # ================================================
    # TOP: T + sw
    # ================================================
    ax_top.pcolormesh(
        Xc / 1e3,
        Yc,
        mask_j.astype(float),
        cmap=mask_cmap,
        shading='auto'
    )

    Tj = T_field[:, tt].copy()
    Tj[mask[:, tt]] = np.nan
    Tj = Tj.reshape(Grid.Nx, Grid.Ny).T

    ax_top.pcolormesh(
        Xc / 1e3,
        Yc,
        Tj,
        cmap="Reds",
        vmin=Tmin,
        vmax=Tmax,
        shading='auto'
    )

    swj = sw_sol[:, tt].copy()
    swj[swj < sw_phys_min] = np.nan
    swj[swj > 1.0] = 1.0
    swj[mask[:, tt]] = np.nan
    swj = swj.reshape(Grid.Nx, Grid.Ny).T

    ax_top.pcolormesh(
        Xc / 1e3,
        Yc,
        swj,
        cmap=cmap_sw,
        norm=LogNorm(vmin=sw_plot_min, vmax=sw_max),
        shading='auto'
    )

    style_axis(ax_top, show_xlabel=False, show_ylabel=True)
    ax_top.tick_params(labelbottom=False)

    # ================================================
    # MIDDLE: phi
    # ================================================
    ax_mid.pcolormesh(
        Xc / 1e3,
        Yc,
        mask_j.astype(float),
        cmap=mask_cmap,
        shading='auto'
    )

    poroj = porosity[:, tt].copy()
    poroj[mask[:, tt]] = np.nan
    poroj = poroj.reshape(Grid.Nx, Grid.Ny).T

    ax_mid.pcolormesh(
        Xc / 1e3,
        Yc,
        poroj,
        cmap="Greys_r",
        vmin=poro_min,
        vmax=poro_max,
        shading='auto'
    )

    style_axis(ax_mid, show_xlabel=False, show_ylabel=True)
    ax_mid.tick_params(labelbottom=False)

    # ================================================
    # BOTTOM: classification
    # ================================================
    phi_now   = phi_sol[:, tt]
    phi_w_now = phi_w_sol[:, tt]
    phi_i_now = phi_i_sol[:, tt]

    sw_cell = np.divide(
        phi_w_now,
        np.maximum(phi_now * (1.0 - s_gr), 1e-12),
        out=np.full_like(phi_w_now, np.nan, dtype=float),
        where=np.maximum(phi_now * (1.0 - s_gr), 1e-12) > 0
    )

    dof_inact = Grid.dof[np.isfinite(sw_cell) & (sw_cell >= sat_threshold)]
    dof_ice_layer = Grid.dof[phi_i_now >= phi_i_threshold]

    dof_f_saturated = find_faces(dof_inact, D, Grid)
    dof_sat_faces   = find_all_faces(dof_inact, D, Grid)

    X_sat, Y_sat = comp_face_coords(dof_sat_faces, Grid)
    X_bnd, Y_bnd = comp_face_coords(dof_f_saturated, Grid)
    X_all, Y_all = comp_face_coords(Grid.dof_f, Grid)

    dof_ice_layer_faces = find_all_faces(dof_ice_layer, D, Grid)
    X_ice, Y_ice = comp_face_coords(dof_ice_layer_faces, Grid)
    dof_f_ice = find_faces(dof_ice_layer, D, Grid)
    X_ice_bnd, Y_ice_bnd = comp_face_coords(dof_f_ice, Grid)

    ax_bot.plot(X_all / 1e3, Y_all, "k-", linewidth=0.4)
    ax_bot.plot(X_sat / 1e3, Y_sat, "r-", linewidth=0.4)
    ax_bot.plot(X_bnd / 1e3, Y_bnd, "r-", linewidth=2.0)
    ax_bot.plot(X_ice / 1e3, Y_ice, "b-", linewidth=0.4)
    ax_bot.plot(X_ice_bnd / 1e3, Y_ice_bnd, "b-", linewidth=2.0)

    if len(dof_inact) > 0:
        ax_bot.plot(
            Xc_col[dof_inact - 1, 0] / 1e3,
            Yc_col[dof_inact - 1, 0],
            "ro",
            markerfacecolor="None",
            markersize=2
        )

    if len(dof_ice_layer) > 0:
        ax_bot.plot(
            Xc_col[dof_ice_layer - 1, 0] / 1e3,
            Yc_col[dof_ice_layer - 1, 0],
            "bo",
            markerfacecolor="None",
            markersize=2
        )

    style_axis(ax_bot, show_xlabel=True, show_ylabel=True)

    return []

# --------------------------------------------------
# CREATE VIDEO
# --------------------------------------------------
ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(time_indices),
    interval=100,
    blit=False
)

# --------------------------------------------------
# SAVE VIDEO
# --------------------------------------------------
Writer = animation.writers['ffmpeg']
writer = Writer(
    fps=10,
    metadata=dict(artist='OpenAI'),
    bitrate=1800,
    extra_args=[
        '-vf', 'pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2',
        '-pix_fmt', 'yuv420p'
    ]
)

ani.save(
    "../Figures/firn_video_4000m_with_classification_pcolormesh.mp4",
    writer=writer,
    dpi=200
)

plt.close(fig)




# ==================================================
# Combined 1D/2D HydroFirn video
# Top 3 panels: static plots
# Bottom 2 panels: evolving plots
# Yellow mask in static panels 2 and 3
# Gap only between panels 3 and 4
# Uses GridSpec spacer row to avoid clipping / cropped labels
# Panels 4 and 5 use pcolormesh
# Original dpi restored to 120
# ==================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, ListedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.dates as mdates
from matplotlib import rcParams

# ==================================================
# LOAD STATIC-PLOT DATA
# ==================================================
data_static = np.load(
    '2D_SamiraA_withSamimiflux_lowres_absolute_k0_onetimes_phiexp1.885_cond__C0.03_50by200_t10368000.npz'
)

t_static      = data_static['t']
phi_w_sol_1d  = data_static['phi_w_sol'].copy()
phi_i_sol_1d  = data_static['phi_i_sol'].copy()
H_sol         = data_static['H_sol']
T_sol_1d      = data_static['T_sol'].copy()
s_w_sol       = data_static['s_w_sol']

phi           = data_static['phi']
Xc_static     = data_static['Xc']
Yc_static     = data_static['Yc']
Xc_col_static = data_static['Xc_col']
Yc_col_static = data_static['Yc_col']

# grid info for static panels
Grid1D = type("Grid1D", (), {})()
Grid1D.Nx   = int(data_static['Grid_Nx'])
Grid1D.Ny   = int(data_static['Grid_Ny'])
Grid1D.xc   = data_static['Grid_xc']
Grid1D.yc   = data_static['Grid_yc']
Grid1D.xf   = data_static['Grid_xf']
Grid1D.yf   = data_static['Grid_yf']
Grid1D.ymax = 2.6
Grid1D.ymin = np.min(Grid1D.yf)

phi_sol_1d = 1.0 - phi_i_sol_1d

# ==================================================
# LOAD EVOLVING 2D VIDEO DATA
# ==================================================
data_dyn = np.load(
    'samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx4000.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz'
)

phi_i_sol_2d = data_dyn['phi_i_sol'].copy()
phi_w_sol_2d = data_dyn['phi_w_sol'].copy()
T_sol_2d     = data_dyn['T_sol'].copy()
t_dyn        = data_dyn['t']

# ==================================================
# GRID / OPS FOR EVOLVING PANELS
# ==================================================
Grid = build_grid(Grid)
[D, G, I] = build_ops(Grid)

# ==================================================
# DERIVED FIELDS FOR EVOLVING PANELS
# ==================================================
phi_sol_2d  = 1.0 - phi_i_sol_2d
porosity_2d = phi_sol_2d.copy()
T_field_2d  = T_sol_2d - Tm

sw_sol_2d = np.divide(
    phi_w_sol_2d,
    np.maximum(porosity_2d * (1.0 - s_gr), 1e-12),
    out=np.full_like(phi_w_sol_2d, np.nan, dtype=float),
    where=np.maximum(porosity_2d * (1.0 - s_gr), 1e-12) > 0
)

# ==================================================
# LIMITS / COLORMAPS
# ==================================================
Tmin, Tmax = -22.0, 0.0
poro_min, poro_max = 0.0, 1.0
sw_plot_min = 1e-2
sw_phys_min = 1e-10
sw_max = 1.0

mask_cmap = ListedColormap(["#FFF6CC"])
cmap_sw = plt.cm.Blues.copy()
cmap_sw.set_under("#F7FBFF")

# ==================================================
# MASKS
# ==================================================
mask_2d = phi_sol_2d > (1.0 - non_porous_vol_frac)

# ==================================================
# EXTERNAL DATA FOR STATIC PANELS
# ==================================================
df = pd.read_excel('./Samira-data/TDRA_MaytoSept2016.xlsx', header=(1))

data_upgpr = pd.read_csv(
    '/Users/ms6985/Desktop/Research_new/meltwater-percolation/Infiltration/JPL/Samira-data/Achilig_upGPR_data.csv'
)

dates = data_upgpr['date'].apply(
    lambda x: datetime.strptime(x, "'%d-%b-%Y %H:%M:%S.%f'")
)

rcParams.update({'font.size': 18})

# ==================================================
# STATIC PANEL ARRAYS
# ==================================================
tday = t_static / day2s

depth_array = np.kron(np.ones(len(tday)), np.transpose([Grid1D.yc]))

phi_w_sol_static = phi_w_sol_1d.copy()
phi_i_sol_static = phi_i_sol_1d.copy()
T_sol_static     = T_sol_1d.copy()

mid = int(Grid1D.Nx * Grid1D.Ny / 2)

phi_w_array = phi_w_sol_static[mid:mid + Grid1D.Ny, :].copy()
phi_i_array = phi_i_sol_static[mid:mid + Grid1D.Ny, :].copy()
T_array     = T_sol_static[mid:mid + Grid1D.Ny, :].copy()

phi_array = 1.0 - phi_i_array

nan_mask_1d = np.isnan(phi_i_array) | np.isnan(phi_w_array) | np.isnan(T_array)
phi_w_array[nan_mask_1d] = np.nan
phi_i_array[nan_mask_1d] = np.nan
T_array[nan_mask_1d]     = np.nan
phi_array[nan_mask_1d]   = np.nan

# ==================================================
# STATIC YELLOW MASK: porosity < threshold
# ==================================================
mask_1d_bool = phi_array > 1 - non_porous_vol_frac
mask_1d_plot = np.where(mask_1d_bool, 1.0, np.nan)

T_array[T_array > Tm] = Tm
T_array[T_array < Tm - 22] = Tm - 22

T_array2 = T_array.copy()
T_array2[T_array2 >= Tm] = np.nan
T_array2[mask_1d_bool] = np.nan

sw_array = np.divide(
    phi_w_array,
    np.maximum(phi_array * (1.0 - s_gr), 1e-12),
    out=np.full_like(phi_w_array, np.nan),
    where=np.maximum(phi_array * (1.0 - s_gr), 1e-12) > 1e-12
)
sw_array[sw_array < 1e-10] = np.nan
sw_array[sw_array > 1.0] = 1.0
sw_array[nan_mask_1d] = np.nan
sw_array[mask_1d_bool] = np.nan

phi_array_plot = phi_array.copy()
phi_array_plot[mask_1d_bool] = np.nan

# datetime grid
t_dates_1d = np.array([dates[850] + timedelta(seconds=float(i)) for i in t_static])
t_dates_1d = np.array([[np.datetime64(i) for i in t_dates_1d]])
t_dates_1d = np.repeat(t_dates_1d, Grid1D.Ny, axis=0)

# ==================================================
# TIME SETTINGS
# ==================================================
frame_stride = 10
time_indices = np.arange(0, len(t_dyn), frame_stride)
if time_indices[-1] != len(t_dyn) - 1:
    time_indices = np.append(time_indices, len(t_dyn) - 1)

start_date = dates[850]

def make_time_label(tt):
    return (start_date + timedelta(seconds=float(t_dyn[tt]))).strftime("%m/%d/%y")

# ==================================================
# STATIC X AXIS SETTINGS
# ==================================================
x_start = datetime(2016, 5, 24)
x_end   = datetime(2016, 9, 21)

n_uniform_ticks = 5
uniform_tick_nums = np.linspace(mdates.date2num(x_start), mdates.date2num(x_end), n_uniform_ticks)
uniform_tick_dates = mdates.num2date(uniform_tick_nums)
uniform_tick_labels = [d.strftime('%m/%d') for d in uniform_tick_dates]

# ==================================================
# FIGURE SETUP
# ==================================================
fig = plt.figure(figsize=(16, 14), dpi=90)

gs = GridSpec(
    6, 2,
    width_ratios=[1.0, 0.05],
    height_ratios=[1, 1, 1, 0.28, 1, 1],
    left=0.08, right=0.92, bottom=0.12, top=0.94,
    hspace=0.0, wspace=0.08
)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

ax_gap = fig.add_subplot(gs[3, 0])
ax_gap.axis("off")

ax4 = fig.add_subplot(gs[4, 0])
ax5 = fig.add_subplot(gs[5, 0], sharex=ax4, sharey=ax4)

cax_sw  = fig.add_subplot(gs[0, 1])
cax_T   = fig.add_subplot(gs[1, 1])
cax_phi = fig.add_subplot(gs[2, 1])

ax_blank_gap = fig.add_subplot(gs[3, 1]); ax_blank_gap.axis("off")
ax_blank1    = fig.add_subplot(gs[4, 1]); ax_blank1.axis("off")
ax_blank2    = fig.add_subplot(gs[5, 1]); ax_blank2.axis("off")

# ==================================================
# SINGLE COLORBAR SET
# ==================================================
m_sw = plt.cm.ScalarMappable(
    cmap=cmap_sw,
    norm=LogNorm(vmin=sw_plot_min, vmax=sw_max)
)
m_sw.set_array([])
cb_sw = fig.colorbar(m_sw, cax=cax_sw, orientation='vertical', extend='min')
cb_sw.set_label(r"$s_w$")
cb_sw.set_ticks([1e-2, 1e-1, 1e0])
cb_sw.set_ticklabels([r"10$^{-2}$", r"10$^{-1}$", r"10$^{0}$"])

m_T = plt.cm.ScalarMappable(
    cmap="Reds",
    norm=Normalize(vmin=Tmin, vmax=Tmax)
)
m_T.set_array([])
cb_T = fig.colorbar(m_T, cax=cax_T, orientation='vertical')
cb_T.set_ticks([-5, -10, -15, -20])
cb_T.set_label(r"$T$ [$^\circ$C]")

m_phi = plt.cm.ScalarMappable(
    cmap="Greys_r",
    norm=Normalize(vmin=poro_min, vmax=poro_max)
)
m_phi.set_array([])
cb_phi = fig.colorbar(m_phi, cax=cax_phi, orientation='vertical')
cb_phi.set_label(r"Porosity $\phi$")

# ==================================================
# STATIC PANEL 1: Qnet + accumulation
# ==================================================
ax1.plot(
    [dates[850] + timedelta(days=i) for i in day],
    Qnet,
    color=red,
    linewidth=2
)
ax1.set_ylim([-25, 80])
ax1.set_yticks([0, 20, 40, 60])
ax1.set_ylabel("")
ax1.spines['left'].set_color(red)
ax1.tick_params(axis='y', colors=red)

ax1new = ax1.twinx()
ax1new.spines['right'].set_color(blue)
ax1new.plot(
    [dates[850] + timedelta(days=i) for i in day],
    RetMIP_acc_rate(day) * 1e3 * day2s,
    color=blue
)
ax1new.tick_params(axis='y', colors=blue)
ax1new.set_yticks([0, 5, 10, 15])
ax1new.set_ylabel("")

ax1.text(
    0.02, 0.96,
    r'Q$_{net}$ [W/m$^2$]',
    transform=ax1.transAxes,
    ha='left', va='top',
    color=red
)
ax1.text(
    0.98, 0.96,
    r'$a$ [mm.w.e./day]',
    transform=ax1.transAxes,
    ha='right', va='top',
    color=blue
)

# ==================================================
# STATIC PANEL 2: T + sw + yellow mask
# ==================================================
ax2.contourf(
    t_dates_1d,
    depth_array,
    mask_1d_plot,
    levels=[0.5, 1.5],
    cmap=mask_cmap
)

ax2.contourf(
    t_dates_1d,
    depth_array,
    T_array2 - Tm,
    cmap="Reds",
    levels=100,
    vmin=Tmin,
    vmax=Tmax
)

ax2.contourf(
    t_dates_1d,
    depth_array,
    sw_array,
    cmap="Blues",
    levels=np.logspace(-2, 0, 100),
    norm=LogNorm(vmin=1e-2, vmax=1.0),
    extend='min'
)

ax2.plot(
    dates[850:4200],
    data_upgpr['height'][850] - data_upgpr['height'][850:4200],
    'k-'
)

ax2.set_ylabel("z [m]")
ax2.set_ylim([Grid1D.ymax, Grid1D.ymin])

# ==================================================
# STATIC PANEL 3: porosity + yellow mask
# ==================================================
ax3.contourf(
    t_dates_1d,
    depth_array,
    mask_1d_plot,
    levels=[0.5, 1.5],
    cmap=mask_cmap
)

ax3.contourf(
    t_dates_1d,
    depth_array,
    phi_array_plot,
    cmap="Greys_r",
    levels=100,
    vmin=poro_min,
    vmax=poro_max
)

ax3.set_ylabel("z [m]")
ax3.set_ylim([Grid1D.ymax, Grid1D.ymin])

for ax in [ax1, ax2, ax3]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_xlim([x_start, x_end])

ax3.set_xticks(uniform_tick_dates)
ax3.set_xticklabels(uniform_tick_labels, rotation=0)

ax1.tick_params(labelbottom=False)
ax2.tick_params(labelbottom=False)

# ==================================================
# MOVING VERTICAL TIMESTAMP LINES
# ==================================================
current_date0 = start_date + timedelta(seconds=float(t_dyn[time_indices[0]]))

vline1  = ax1.axvline(current_date0, color='k', linestyle='--', linewidth=1.5)
vline1b = ax1new.axvline(current_date0, color='k', linestyle='--', linewidth=1.5)
vline2  = ax2.axvline(current_date0, color='k', linestyle='--', linewidth=1.5)
vline3  = ax3.axvline(current_date0, color='k', linestyle='--', linewidth=1.5)

# ==================================================
# HELPER
# ==================================================
def style_axis(ax, show_xlabel=False, show_ylabel=False):
    ax.set_xlim([Xc_static.min() / 1e3, Xc_static.max() / 1e3])
    ax.set_ylim([2.6, Grid.ymin])
    if show_ylabel:
        ax.set_ylabel("z [m]")
    if show_xlabel:
        ax.set_xlabel("x [km]")
    ax.tick_params(axis='both', direction='inout', length=6, width=1)

# ==================================================
# ANIMATION FUNCTION
# ==================================================
def update(frame):
    tt = time_indices[frame]
    current_date = start_date + timedelta(seconds=float(t_dyn[tt]))

    vline1.set_xdata([current_date, current_date])
    vline1b.set_xdata([current_date, current_date])
    vline2.set_xdata([current_date, current_date])
    vline3.set_xdata([current_date, current_date])

    label = make_time_label(tt)
    fig.suptitle(f"Date: {label}", y=0.985, fontsize=24)

    for ax in [ax4, ax5]:
        ax.cla()

    mask_j = mask_2d[:, tt].reshape(Grid.Nx, Grid.Ny).T

    # ================================================
    # PANEL 4: evolving T + sw (pcolormesh)
    # ================================================
    ax4.pcolormesh(
        Xc_static / 1e3,
        Yc_static,
        mask_j.astype(float),
        cmap=mask_cmap,
        shading='auto'
    )

    Tj = T_field_2d[:, tt].copy()
    Tj[mask_2d[:, tt]] = np.nan
    Tj_plot = Tj.reshape(Grid.Nx, Grid.Ny).T

    ax4.pcolormesh(
        Xc_static / 1e3,
        Yc_static,
        Tj_plot,
        cmap="Reds",
        vmin=Tmin,
        vmax=Tmax,
        shading='auto'
    )

    swj = sw_sol_2d[:, tt].copy()
    swj[swj < sw_phys_min] = np.nan
    swj[swj > 1.0] = 1.0
    swj[mask_2d[:, tt]] = np.nan
    swj_plot = swj.reshape(Grid.Nx, Grid.Ny).T
    swj_plot[swj_plot < sw_plot_min] = np.nan

    ax4.pcolormesh(
        Xc_static / 1e3,
        Yc_static,
        swj_plot,
        cmap=cmap_sw,
        norm=LogNorm(vmin=sw_plot_min, vmax=sw_max),
        shading='auto'
    )

    style_axis(ax4, show_xlabel=False, show_ylabel=True)
    ax4.tick_params(labelbottom=False)

    # ================================================
    # PANEL 5: evolving porosity (pcolormesh)
    # ================================================
    ax5.pcolormesh(
        Xc_static / 1e3,
        Yc_static,
        mask_j.astype(float),
        cmap=mask_cmap,
        shading='auto'
    )

    poroj = porosity_2d[:, tt].copy()
    poroj[mask_2d[:, tt]] = np.nan
    poroj_plot = poroj.reshape(Grid.Nx, Grid.Ny).T

    ax5.pcolormesh(
        Xc_static / 1e3,
        Yc_static,
        poroj_plot,
        cmap="Greys_r",
        vmin=poro_min,
        vmax=poro_max,
        shading='auto'
    )

    style_axis(ax5, show_xlabel=True, show_ylabel=True)
    ax5.tick_params(axis='x', labelbottom=True, pad=5)
    ax5.xaxis.set_label_position('bottom')
    ax5.set_xlabel("x [km]", labelpad=10)

    return [vline1, vline1b, vline2, vline3]

# ==================================================
# CREATE VIDEO
# ==================================================
ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(time_indices),
    interval=100,
    blit=False
)

# ==================================================
# SAVE VIDEO
# ==================================================
Writer = animation.writers['ffmpeg']
writer = Writer(
    fps=10,
    metadata=dict(artist='Shadab'),
    bitrate=1800,
    extra_args=[
        '-vf', 'pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2',
        '-pix_fmt', 'yuv420p'
    ]
)

ani.save(
    "../Figures/HydroFirn_5panel_static_top_evolving_bottom.mp4",
    writer=writer,
    dpi=90
)

plt.close(fig)




# ==================================================
# Combined 1D/2D HydroFirn video
# Top 3 panels: static plots
# Bottom 2 panels: evolving plots
# Yellow mask in static panels 2 and 3
# Gap only between panels 3 and 4
# Uses GridSpec spacer row to avoid clipping / cropped labels
# Panels 4 and 5 use pcolormesh
# Qnet is forced above the accumulation line
# dpi kept at 100
# ==================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, ListedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.dates as mdates
from matplotlib import rcParams

# ==================================================
# LOAD STATIC-PLOT DATA
# ==================================================
data_static = np.load(
    '2D_SamiraA_withSamimiflux_lowres_absolute_k0_onetimes_phiexp1.885_cond__C0.03_50by200_t10368000.npz'
)

t_static      = data_static['t']
phi_w_sol_1d  = data_static['phi_w_sol'].copy()
phi_i_sol_1d  = data_static['phi_i_sol'].copy()
H_sol         = data_static['H_sol']
T_sol_1d      = data_static['T_sol'].copy()
s_w_sol       = data_static['s_w_sol']

phi           = data_static['phi']
Xc_static     = data_static['Xc']
Yc_static     = data_static['Yc']
Xc_col_static = data_static['Xc_col']
Yc_col_static = data_static['Yc_col']

# grid info for static panels
Grid1D = type("Grid1D", (), {})()
Grid1D.Nx   = int(data_static['Grid_Nx'])
Grid1D.Ny   = int(data_static['Grid_Ny'])
Grid1D.xc   = data_static['Grid_xc']
Grid1D.yc   = data_static['Grid_yc']
Grid1D.xf   = data_static['Grid_xf']
Grid1D.yf   = data_static['Grid_yf']
Grid1D.ymax = 2.6
Grid1D.ymin = np.min(Grid1D.yf)

phi_sol_1d = 1.0 - phi_i_sol_1d

# ==================================================
# LOAD EVOLVING 2D VIDEO DATA
# ==================================================
data_dyn = np.load(
    'samimi_without_acc_new_0deg_slopen_ormalizedFalsecorryes_lx4000.0m_ly1m_amp0.05_C0.03_50by200_t10368000.npz'
)

phi_i_sol_2d = data_dyn['phi_i_sol'].copy()
phi_w_sol_2d = data_dyn['phi_w_sol'].copy()
T_sol_2d     = data_dyn['T_sol'].copy()
t_dyn        = data_dyn['t']

# ==================================================
# GRID / OPS FOR EVOLVING PANELS
# ==================================================
Grid = build_grid(Grid)
[D, G, I] = build_ops(Grid)

# ==================================================
# DERIVED FIELDS FOR EVOLVING PANELS
# ==================================================
phi_sol_2d  = 1.0 - phi_i_sol_2d
porosity_2d = phi_sol_2d.copy()
T_field_2d  = T_sol_2d - Tm

sw_sol_2d = np.divide(
    phi_w_sol_2d,
    np.maximum(porosity_2d * (1.0 - s_gr), 1e-12),
    out=np.full_like(phi_w_sol_2d, np.nan, dtype=float),
    where=np.maximum(porosity_2d * (1.0 - s_gr), 1e-12) > 0
)

# ==================================================
# LIMITS / COLORMAPS
# ==================================================
Tmin, Tmax = -22.0, 0.0
poro_min, poro_max = 0.0, 1.0
sw_plot_min = 1e-2
sw_phys_min = 1e-10
sw_max = 1.0

mask_cmap = ListedColormap(["#FFF6CC"])
cmap_sw = plt.cm.Blues.copy()
cmap_sw.set_under("#F7FBFF")

# ==================================================
# MASKS
# ==================================================
mask_2d = phi_sol_2d > (1.0 - non_porous_vol_frac)

# ==================================================
# EXTERNAL DATA FOR STATIC PANELS
# ==================================================
df = pd.read_excel('./Samira-data/TDRA_MaytoSept2016.xlsx', header=(1))

data_upgpr = pd.read_csv(
    '/Users/ms6985/Desktop/Research_new/meltwater-percolation/Infiltration/JPL/Samira-data/Achilig_upGPR_data.csv'
)

dates = data_upgpr['date'].apply(
    lambda x: datetime.strptime(x, "'%d-%b-%Y %H:%M:%S.%f'")
)

rcParams.update({'font.size': 18})

# ==================================================
# STATIC PANEL ARRAYS
# ==================================================
tday = t_static / day2s

depth_array = np.kron(np.ones(len(tday)), np.transpose([Grid1D.yc]))

phi_w_sol_static = phi_w_sol_1d.copy()
phi_i_sol_static = phi_i_sol_1d.copy()
T_sol_static     = T_sol_1d.copy()

mid = int(Grid1D.Nx * Grid1D.Ny / 2)

phi_w_array = phi_w_sol_static[mid:mid + Grid1D.Ny, :].copy()
phi_i_array = phi_i_sol_static[mid:mid + Grid1D.Ny, :].copy()
T_array     = T_sol_static[mid:mid + Grid1D.Ny, :].copy()

phi_array = 1.0 - phi_i_array

nan_mask_1d = np.isnan(phi_i_array) | np.isnan(phi_w_array) | np.isnan(T_array)
phi_w_array[nan_mask_1d] = np.nan
phi_i_array[nan_mask_1d] = np.nan
T_array[nan_mask_1d]     = np.nan
phi_array[nan_mask_1d]   = np.nan

# ==================================================
# STATIC YELLOW MASK
# ==================================================
mask_1d_bool = phi_array > 1 - non_porous_vol_frac
mask_1d_plot = np.where(mask_1d_bool, 1.0, np.nan)

T_array[T_array > Tm] = Tm
T_array[T_array < Tm - 22] = Tm - 22

T_array2 = T_array.copy()
T_array2[T_array2 >= Tm] = np.nan
T_array2[mask_1d_bool] = np.nan

sw_array = np.divide(
    phi_w_array,
    np.maximum(phi_array * (1.0 - s_gr), 1e-12),
    out=np.full_like(phi_w_array, np.nan),
    where=np.maximum(phi_array * (1.0 - s_gr), 1e-12) > 1e-12
)
sw_array[sw_array < 1e-10] = np.nan
sw_array[sw_array > 1.0] = 1.0
sw_array[nan_mask_1d] = np.nan
sw_array[mask_1d_bool] = np.nan

phi_array_plot = phi_array.copy()
phi_array_plot[mask_1d_bool] = np.nan

# datetime grid
t_dates_1d = np.array([dates[850] + timedelta(seconds=float(i)) for i in t_static])
t_dates_1d = np.array([[np.datetime64(i) for i in t_dates_1d]])
t_dates_1d = np.repeat(t_dates_1d, Grid1D.Ny, axis=0)

# ==================================================
# TIME SETTINGS
# ==================================================
frame_stride = 10
time_indices = np.arange(0, len(t_dyn), frame_stride)
if time_indices[-1] != len(t_dyn) - 1:
    time_indices = np.append(time_indices, len(t_dyn) - 1)

start_date = dates[850]

def make_time_label(tt):
    return (start_date + timedelta(seconds=float(t_dyn[tt]))).strftime("%m/%d/%y")

# ==================================================
# STATIC X AXIS SETTINGS
# ==================================================
x_start = datetime(2016, 5, 24)
x_end   = datetime(2016, 9, 21)

n_uniform_ticks = 5
uniform_tick_nums = np.linspace(mdates.date2num(x_start), mdates.date2num(x_end), n_uniform_ticks)
uniform_tick_dates = mdates.num2date(uniform_tick_nums)
uniform_tick_labels = [d.strftime('%m/%d') for d in uniform_tick_dates]

# ==================================================
# FIGURE SETUP
# ==================================================
fig = plt.figure(figsize=(16, 14), dpi=100)

gs = GridSpec(
    6, 2,
    width_ratios=[1.0, 0.05],
    height_ratios=[1, 1, 1, 0.28, 1, 1],
    left=0.08, right=0.92, bottom=0.12, top=0.94,
    hspace=0.0, wspace=0.08
)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

ax_gap = fig.add_subplot(gs[3, 0])
ax_gap.axis("off")

ax4 = fig.add_subplot(gs[4, 0])
ax5 = fig.add_subplot(gs[5, 0], sharex=ax4, sharey=ax4)

cax_sw  = fig.add_subplot(gs[0, 1])
cax_T   = fig.add_subplot(gs[1, 1])
cax_phi = fig.add_subplot(gs[2, 1])

ax_blank_gap = fig.add_subplot(gs[3, 1]); ax_blank_gap.axis("off")
ax_blank1    = fig.add_subplot(gs[4, 1]); ax_blank1.axis("off")
ax_blank2    = fig.add_subplot(gs[5, 1]); ax_blank2.axis("off")

# ==================================================
# SINGLE COLORBAR SET
# ==================================================
m_sw = plt.cm.ScalarMappable(
    cmap=cmap_sw,
    norm=LogNorm(vmin=sw_plot_min, vmax=sw_max)
)
m_sw.set_array([])
cb_sw = fig.colorbar(m_sw, cax=cax_sw, orientation='vertical', extend='min')
cb_sw.set_label(r"$s_w$")
cb_sw.set_ticks([1e-2, 1e-1, 1e0])
cb_sw.set_ticklabels([r"10$^{-2}$", r"10$^{-1}$", r"10$^{0}$"])

m_T = plt.cm.ScalarMappable(
    cmap="Reds",
    norm=Normalize(vmin=Tmin, vmax=Tmax)
)
m_T.set_array([])
cb_T = fig.colorbar(m_T, cax=cax_T, orientation='vertical')
cb_T.set_ticks([-5, -10, -15, -20])
cb_T.set_label(r"$T$ [$^\circ$C]")

m_phi = plt.cm.ScalarMappable(
    cmap="Greys_r",
    norm=Normalize(vmin=poro_min, vmax=poro_max)
)
m_phi.set_array([])
cb_phi = fig.colorbar(m_phi, cax=cax_phi, orientation='vertical')
cb_phi.set_ticks([0,0.25,0.5,0.75])
cb_phi.set_label(r"Porosity $\phi$")

# ==================================================
# STATIC PANEL 1: Qnet + accumulation
# ==================================================
ax1.plot(
    [dates[850] + timedelta(days=i) for i in day],
    Qnet,
    color=red,
    linewidth=2,
    zorder=5
)
ax1.set_ylim([-25, 80])
ax1.set_yticks([0, 20, 40, 60])
ax1.set_ylabel("")
ax1.spines['left'].set_color(red)
ax1.tick_params(axis='y', colors=red)

ax1new = ax1.twinx()
ax1new.plot(
    [dates[850] + timedelta(days=i) for i in day],
    RetMIP_acc_rate(day) * 1e3 * day2s,
    color=blue,
    zorder=3
)
ax1new.spines['right'].set_color(blue)
ax1new.tick_params(axis='y', colors=blue)
ax1new.set_yticks([0, 5, 10, 15])
ax1new.set_ylabel("")

# layering fix so Qnet stays above
ax1.set_zorder(2)
ax1new.set_zorder(1)
ax1.patch.set_visible(False)

ax1.text(
    0.02, 0.96,
    r'Q$_{net}$ [W/m$^2$]',
    transform=ax1.transAxes,
    ha='left', va='top',
    color=red,
    zorder=10
)
ax1.text(
    0.98, 0.96,
    r'$a$ [mm.w.e./day]',
    transform=ax1.transAxes,
    ha='right', va='top',
    color=blue,
    zorder=10
)

# ==================================================
# STATIC PANEL 2: T + sw + yellow mask
# ==================================================
ax2.contourf(
    t_dates_1d,
    depth_array,
    mask_1d_plot,
    levels=[0.5, 1.5],
    cmap=mask_cmap
)

ax2.contourf(
    t_dates_1d,
    depth_array,
    T_array2 - Tm,
    cmap="Reds",
    levels=100,
    vmin=Tmin,
    vmax=Tmax
)

ax2.contourf(
    t_dates_1d,
    depth_array,
    sw_array,
    cmap="Blues",
    levels=np.logspace(-2, 0, 100),
    norm=LogNorm(vmin=1e-2, vmax=1.0),
    extend='min'
)

ax2.plot(
    dates[850:4200],
    data_upgpr['height'][850] - data_upgpr['height'][850:4200],
    'k-'
)

ax2.set_ylabel("z [m]")
ax2.set_ylim([Grid1D.ymax, Grid1D.ymin])

# ==================================================
# STATIC PANEL 3: porosity + yellow mask
# ==================================================
ax3.contourf(
    t_dates_1d,
    depth_array,
    mask_1d_plot,
    levels=[0.5, 1.5],
    cmap=mask_cmap
)

ax3.contourf(
    t_dates_1d,
    depth_array,
    phi_array_plot,
    cmap="Greys_r",
    levels=100,
    vmin=poro_min,
    vmax=poro_max
)

ax3.set_ylabel("z [m]")
ax3.set_ylim([Grid1D.ymax, Grid1D.ymin])

for ax in [ax1, ax2, ax3]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_xlim([x_start, x_end])

ax3.set_xticks(uniform_tick_dates)
ax3.set_xticklabels(uniform_tick_labels, rotation=0)

ax1.tick_params(labelbottom=False)
ax2.tick_params(labelbottom=False)

# ==================================================
# MOVING VERTICAL TIMESTAMP LINES
# ==================================================
current_date0 = start_date + timedelta(seconds=float(t_dyn[time_indices[0]]))

vline1  = ax1.axvline(current_date0, color='k', linestyle='--', linewidth=1.5, zorder=8)
vline1b = ax1new.axvline(current_date0, color='k', linestyle='--', linewidth=1.5, zorder=8)
vline2  = ax2.axvline(current_date0, color='k', linestyle='--', linewidth=1.5)
vline3  = ax3.axvline(current_date0, color='k', linestyle='--', linewidth=1.5)

# ==================================================
# HELPER
# ==================================================
def style_axis(ax, show_xlabel=False, show_ylabel=False):
    ax.set_xlim([Xc_static.min() / 1e3, Xc_static.max() / 1e3])
    ax.set_ylim([2.6, Grid.ymin])
    if show_ylabel:
        ax.set_ylabel("z [m]")
    if show_xlabel:
        ax.set_xlabel("x [km]")
    ax.tick_params(axis='both', direction='inout', length=6, width=1)

# ==================================================
# ANIMATION FUNCTION
# ==================================================
def update(frame):
    tt = time_indices[frame]
    current_date = start_date + timedelta(seconds=float(t_dyn[tt]))

    vline1.set_xdata([current_date, current_date])
    vline1b.set_xdata([current_date, current_date])
    vline2.set_xdata([current_date, current_date])
    vline3.set_xdata([current_date, current_date])

    label = make_time_label(tt)
    fig.suptitle(f"Date: {label}", y=0.985, fontsize=18)

    for ax in [ax4, ax5]:
        ax.cla()

    mask_j = mask_2d[:, tt].reshape(Grid.Nx, Grid.Ny).T

    # ================================================
    # PANEL 4: evolving T + sw (pcolormesh)
    # ================================================
    ax4.pcolormesh(
        Xc_static / 1e3,
        Yc_static,
        mask_j.astype(float),
        cmap=mask_cmap,
        shading='auto'
    )

    Tj = T_field_2d[:, tt].copy()
    Tj[mask_2d[:, tt]] = np.nan
    Tj_plot = Tj.reshape(Grid.Nx, Grid.Ny).T

    ax4.pcolormesh(
        Xc_static / 1e3,
        Yc_static,
        Tj_plot,
        cmap="Reds",
        vmin=Tmin,
        vmax=Tmax,
        shading='auto'
    )

    swj = sw_sol_2d[:, tt].copy()
    swj[swj < sw_phys_min] = np.nan
    swj[swj > 1.0] = 1.0
    swj[mask_2d[:, tt]] = np.nan
    swj_plot = swj.reshape(Grid.Nx, Grid.Ny).T
    swj_plot[swj_plot < sw_plot_min] = np.nan

    ax4.pcolormesh(
        Xc_static / 1e3,
        Yc_static,
        swj_plot,
        cmap=cmap_sw,
        norm=LogNorm(vmin=sw_plot_min, vmax=sw_max),
        shading='auto'
    )

    style_axis(ax4, show_xlabel=False, show_ylabel=True)
    ax4.tick_params(labelbottom=False)

    # ================================================
    # PANEL 5: evolving porosity (pcolormesh)
    # ================================================
    ax5.pcolormesh(
        Xc_static / 1e3,
        Yc_static,
        mask_j.astype(float),
        cmap=mask_cmap,
        shading='auto'
    )

    poroj = porosity_2d[:, tt].copy()
    poroj[mask_2d[:, tt]] = np.nan
    poroj_plot = poroj.reshape(Grid.Nx, Grid.Ny).T

    ax5.pcolormesh(
        Xc_static / 1e3,
        Yc_static,
        poroj_plot,
        cmap="Greys_r",
        vmin=poro_min,
        vmax=poro_max,
        shading='auto'
    )

    style_axis(ax5, show_xlabel=True, show_ylabel=True)
    ax5.tick_params(axis='x', labelbottom=True, pad=5)
    ax5.xaxis.set_label_position('bottom')
    ax5.set_xlabel("x [km]", labelpad=10)

    return [vline1, vline1b, vline2, vline3]

# ==================================================
# CREATE VIDEO
# ==================================================
ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(time_indices),
    interval=100,
    blit=False
)

# ==================================================
# SAVE VIDEO
# ==================================================
Writer = animation.writers['ffmpeg']
writer = Writer(
    fps=15,
    metadata=dict(artist='Shadab'),
    bitrate=1800,
    extra_args=[
        '-vf', 'pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2',
        '-pix_fmt', 'yuv420p'
    ]
)

ani.save(
    "../Figures/HydroFirn_5panel_static_top_evolving_bottom.mp4",
    writer=writer,
    dpi=100
)

plt.close(fig)