######################################################################
#Gravity current in porous medium
#Mohammad Afzal Shadab
#Date modified: 10/27/2024
######################################################################

######################################################################
#import libraries
######################################################################
import time as time_contour

start_time = time_contour.perf_counter()

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
from Barenblatt_cartesian import Barenblatt_cartesian
from scipy.interpolate import interp1d

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
    fC = phi**m / phi_L**m           #Power law porosity
    return fC

#Rel perm of water: Upwinded
def f_Cn(C,phi,n):
    fC = np.zeros_like(phi)
    fC = ((C/phi-s_wr)/(1-s_gr-s_wr))**n    #Power law rel perm
    fC[C<=0]  = 0.0      
    return fC

######################################################################
#parameters
######################################################################
##simulation
simulation_name = f'test_Gravity_current_lowres_absolute_k0_onetimes_phiexp1.885_cond_well_high_res'
diffusion = 'no'  
#CFL    = 0.8     #CFL number
tilt_angle = 0   #angle of the slope
T_constant = -30#1e-14   #temperature of cold firn [C]
simulation_name = f'{simulation_name}_{diffusion}diff_{T_constant}C_new'

##hydrology
m      = 3 #Cozeny-Karman coefficient for numerator K = K0 (1-phi_i)^m
n      = 2 #Corey-Brooks coefficient krw = krw0 * sw^n
s_wr   = 0.0   #Residual water saturation
s_gr   = 0.0   #Residual gas saturation
k_w0   = 1.0  #relative permeability threshold: wetting phase
mu_w   = 1e-3 #dynamic viscosity: wetting phase    
grav   = 9.81    #acceleration due to gravity
k0     = 5.6e-11#absolute permeability m^2 in pore space Meyer and Hewitt 2017
CFL    = 1.0   #CFL number
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

#domain details
z0 = 3.75 #characteristic height (m)

fc = k0*k_w0*rho_w*grav/mu_w*phi_L**3 #Infiltration capacity (m/s)

sat_threshold = 1-1e-3 #threshold for saturated region formation

#injection
Param.xleft_inj= 0e3;  Param.xright_inj= 1000e3


############################################################
#new code (cold firn aquifer)
############################################################
phi_orig = 0.7
T  = -30 # Temperature of the aquifer [C]
Delta_theta = rho_i*cp_i*(phi_orig-1)/(rho_w*L_fusion)*(T_constant)  #calculating refrozen water (From Clark et al.,2017)
Delta_phi   = rho_w*Delta_theta/rho_i
#Delta_phi = 0.06350981451791826  #from the numeric simulation

phi0 = phi_orig - Delta_phi
Delta_rho = rho_w - rho_nw

S  = phi0**(m-1)*Delta_rho*grav*k0/mu_w  #Constant in constant porosity model (Huppert and Woods, 1994)
S1 = phi0**(m-1)*Delta_rho*grav*k0/mu_w  #Constant in constant porosity model (Huppert and Woods, 1994)
S2 = phi0**(m)*Delta_rho*grav*k0/(mu_w*(phi0+Delta_theta))  #Constant in constant porosity model (Huppert and Woods, 1994)
kappabykappa1 = S2/S1 #the kappa ratio

##Analytic solution initialize
g_solution, beta = Barenblatt_cartesian(kappabykappa1)
BB  = 10
kappa1 = S1/2
AA   = BB**2/kappa1
alpha = 1 - 2*beta
t_init = 1*yr2s  #initial time -- 1 day

h_init = g_solution*AA/t_init**alpha
r_init = BB*t_init**beta
r_init_arr = np.linspace(0,r_init,len(h_init))

L = 2.5*r_init   #Length of the glacier (m)
h_analy_func = interp1d(r_init_arr,h_init, bounds_error=False, fill_value=0)
############################################################

#temporal
tf     = 10*yr2s
tmax = tf   #time scaling with respect to fc
#t_interest = [0,0.25,0.5399999999999999 + 0.005,0.6,0.7116505061468059,1.0] #swr,sgr=0.05
t_interest = np.linspace(t_init,tmax,int(tf/day2s*2)+1)   #swr,sgr=0

Nt   = 10000
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
Grid.xmin =  0*z0; Grid.xmax =L; Grid.Nx = 80;    #Horizontal direction %%%%%Nx =100, xmax = 35 km
Grid.ymin =  0*z0; Grid.ymax =250;Grid.Ny = 40;    #Vertically downward direction
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
phi_nw            = phi_orig*np.ones_like(Yc_col)#%%%%%%  only gas in the firn

phi_nw[(Grid.ymax-Yc_col)<=h_analy_func((Xc_col))] = 0.0

###############################

#phi_nw[Xc_col<1e3]= 0.0
#phi_nw  = fit_porosity(Yc_col) #volume fraction of gas 

phi_w   = np.zeros_like(phi_nw)  #volume fraction of water phase #No water
phi_w[(Grid.ymax-Yc_col)<=h_analy_func((Xc_col))] = phi0

C       = rho_w * phi_w + (1 - phi_w - phi_nw) * rho_i
T =       T_constant*np.ones_like(Yc_col) + Tm


#T       = fit_temp(Yc_col)+Tm %%%%%
#T[Yc_col>3.7] = -15.7 + Tm    %%%%%
H       = enthalpyfromT(T_constant*np.ones_like(Yc_col)+Tm,Tm,rho_i,rho_w,0,cp_i,cp_w,0,phi_w,phi_nw,L_fusion)

#T[Xc_col<1e3] = Tm #%%%%%% Setting the saturated region in between 0 to 2 km.
#H[Xc_col<1e3] = rho_w * phi_w[Xc_col<1e3] * L_fusion

T[(Grid.ymax-Yc_col)<=h_analy_func((Xc_col))] = Tm #%%%%%% Setting the saturated region in between 0 to 2 km.
H[(Grid.ymax-Yc_col)<=h_analy_func((Xc_col))] = rho_w * phi_w[(Grid.ymax-Yc_col)<=h_analy_func((Xc_col))] * L_fusion



phi_w,phi_i,T,dTdH = eval_phase_behaviorCwH(H,Tm,rho_i,rho_w,rho_nw,cp_i,cp_w,cp_nw,C,L_fusion)
s_w_init= s_wfunc(phi_w,phi_nw)
phi = (phi_w+ phi_nw)*np.ones((Grid.N,1))#np.exp(-(1-Yc_col/(grid.ymax-grid.ymin)))#phi*np.ones((grid.N,1)) #porosity in each cell
s_w = s_w_init.copy()#s_wp *np.ones((grid.N,1))
fs_theta = 0.0*np.ones((Grid.N,1))                     #RHS of heat equation


#################################################################
#Plotting#############
plt.figure()
plot = [plt.contourf(Xc, Yc, np.transpose((phi_w[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1)]
plt.colorbar()
plt.ylabel(r'$z$')
#plt.axis('scaled')

plt.figure()
plot = [plt.contourf(Xc, Yc, np.transpose((1-phi_i[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Greys",vmin = 0, vmax = 1)]
plt.colorbar()
plt.ylabel(r'$z$')
#plt.axis('scaled')
#################################################################



######################################################################
#initializing arrays
######################################################################
s_w_sol = np.copy(s_w) #for water saturation
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
BC.C_g    = np.array([])#C[dof_inj-1] #+ rho_w*C_L*np.ones((len(dof_inj),1))
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

t    =[t_init]
time = t_init
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
    #Param.H.qb= Qnet_func(time/day2s)*np.ones((len(dof_inj),1)) 
    #[H_B,H_N,H_fn] = build_bnd(Param.H,Grid,I)

    #BC.C_g    = C[dof_inj-1] 
    #Param.H.qb= Qnet_func(time/day2s)*np.ones((Grid.Nx,1))

    #Param.H.qb= Qnet_func(time/day2s)*np.transpose([np.concatenate([np.flip(np.linspace(1,1.1,int(Grid.Nx/2))),np.linspace(1,1.1,int(Grid.Nx/2))])])
    G = G_original.copy()
    D = D_original.copy()
    # ################################################################################################
    non_porous_vol_frac = 1e-3
    # if np.any(phi_i<=non_porous_vol_frac):
    #     dof_complete_melt  = Grid.dof[phi_i[:,0] < non_porous_vol_frac]
    #     dof_partial_melt   = np.setdiff1d(Grid.dof,dof_complete_melt) #saturated cells 
    #     dof_f_bnd_melt     = find_faces(dof_complete_melt,D,Grid) 
    #     ytop,ybot          = find_top_bot_cells(dof_f_bnd_melt,D,Grid)
        
    #     Param.H.dof_neu = ytop
    #     Param.H.dof_f_neu = dof_f_bnd_melt
    #     Param.H.qb = Qnet_func(time/day2s)*np.ones((Grid.Nx,1))
    #     [H_B,H_N,H_fn] = build_bnd(Param.H,Grid,I)
        
    #     G = zero_rows(G,dof_f_bnd_melt-1)
    #     D = -np.transpose(G)
    #     D  = zero_rows(D,dof_complete_melt-1) #
    
    #     ############################################################
    #     #location of the surface
    #     surf_loc = (Yc_col[ytop[0]-1] + Yc_col[ybot[0]-1]) / 2
    # ################################################################################################     

    phi_w_old = phi_w.copy() 
    C_old     = C.copy()      
    flux      = (comp_harmonicmean(Avg,f_Cm(phi,m))*(flux_upwind(v, Grid) @ f_Cn(phi_w_old,phi,n)))*np.cos(tilt_angle*np.pi/180)
    flux_vert = flux.copy()
    flux_vert[Grid.dof_f<=Grid.Nfx,0] = flux_vert[Grid.dof_f<=Grid.Nfx,0]*np.tan(tilt_angle*np.pi/180)  #making gravity based flux in x direction
    
    if i<=0:
        speed = np.min(f_Cm(1-phi_i,m)*f_Cn(np.array([C_L]),np.array([phi_L]),n)[0]/C_L)
    else:
        speed = f_Cm(phi,m)*f_Cn(phi_w,np.array(phi),n)/(phi_w)  
        speed[np.isnan(speed)] = 0
    
    if np.any(speed[speed>0]):
        dt1   = 1e16#CFL*Grid.dy/np.max(speed[speed>0]) #Calculating the time step from the filling of volume
    else:
        dt1   = 1e16 #Calculating the time step from the filling of volume

    #flux_vert_mat = sp.dia_matrix((flux_vert[:,0],  np.array([0])), shape=(Grid.Nf, Grid.Nf))
    res = D@flux_vert  #since the gradients are only zero and 1    
    res_vert = res.copy()
    ######
    #Taking out the domain to cut off single phase region
    #dof_act  = Grid.dof[C_old[:,0] / (phi[:,0]*(1-s_gr)) < 0.99999]
    dof_act  = Grid.dof[phi_w_old[:,0] / (phi[:,0]*(1-s_gr)) < sat_threshold]
    dof_inact= np.setdiff1d(Grid.dof,dof_act) #saturated cells
    if len(dof_act)< Grid.N: #when atleast one layer is present
        #############################################
        dof_f_saturated = find_faces(dof_inact,D,Grid)       
        #Step 1: Modify the gradient to import natural BC at the crater
        #G_small = zero_rows(G,dof_f_saturated-1)
        
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
        
        #upwinding boundary y-directional flux
        if tilt_angle != 90:
            #finding boundary faces
            dof_ysat_faces = dof_f_saturated[dof_f_saturated>Grid.Nfx]
            
            #removing boundary faces
            dof_ysat_faces = np.setdiff1d(dof_ysat_faces,np.append(Grid.dof_f_ymin,Grid.dof_f_ymax))
            
            ytop,ybot               = find_top_bot_cells(dof_ysat_faces,D,Grid)
            q_w[dof_ysat_faces-1,0] = comp_sat_unsat_bnd_flux(q_w[dof_ysat_faces-1],flux_vert[dof_ysat_faces-1],ytop,ybot,phi_w,phi,sat_threshold)
            
        #upwinding boundary x-directional flux   ####new line
        if tilt_angle != 0:
            #finding boundary faces
            dof_xsat_faces = dof_f_saturated[dof_f_saturated<=Grid.Nfx]
            
            #removing boundary faces
            dof_xsat_faces = np.setdiff1d(dof_xsat_faces,np.append(Grid.dof_f_xmin,Grid.dof_f_xmax))
            
            xleft,xright            = find_left_right_cells(dof_xsat_faces,D,Grid)
            q_w[dof_xsat_faces-1,0] = comp_sat_unsat_bnd_flux(q_w[dof_xsat_faces-1],flux_vert[dof_xsat_faces-1],xright,xleft,phi_w,phi,sat_threshold)
             
        #find all saturated faces
        dof_sat_faces = find_all_faces(dof_inact,D,Grid)  
        
        flux_vert[dof_sat_faces-1] = q_w[dof_sat_faces-1]
        
        res = D @ flux_vert

    #dt2   = np.abs((phi*sat_threshold - phi*s_gr - phi_w_old)/(res)) #Calculating the time step from the filling of volume
    #dt2   = np.abs(dt2[~np.isinf(dt2)])
    #dt2   = np.min(dt2[dt2>0])
    #dt2  =  CFL*np.min(dt2[dt2>1e-4*z0/fc])


    #Time step of diffusion
    #dt3  = np.min(CFL*(Grid.dy**2)/(2*(phi_w*k_w + phi_i*k_i)/(rho_i*phi_i*cp_i+rho_w*phi_w*cp_w))) 
    #if np.isnan(dt3): dt3=1e5
    #dt3= 1e3
    #dt = np.min([dt1,dt2,dt3])
    
        
    #dt   = np.abs((phi*sat_threshold - phi*s_gr - phi_w_old)/(res)) 
    dt   = CFL*np.abs((phi - phi*s_gr - phi_w_old)/(res)) ####HURRAYYYYY
    
    dt[dt<1e-10] = 0.0
    dt  =  np.min(dt[dt>0])
    if i<1000: 
        dt = tmax/(Nt*1000)
    elif time+dt >= t_interest[np.max(np.argwhere(time+dt >= t_interest))] and time < t_interest[np.max(np.argwhere(time+dt >= t_interest))]:
        dt = t_interest[np.max(np.argwhere(time+dt >= t_interest))] - time   #To have the results at a specific time
    
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
    C   = solve_lbvp(I,RHS,B,BC.C_g,N)

    h_i,h_w,h_nw = eval_h(Tm,T,rho_i,rho_w,0,cp_i,cp_w,0,L_fusion)
    
    if diffusion == 'yes':
        RHS    =  H - dt * (D @ (rho_w * flux_vert * comp_harmonicmean(Avg,h_w)  -K_bar_edge @ G @ T  ) -(fs_theta+H_fn)) # -kappa_edge @ G @ H 
    else:
        RHS    =  H - dt * (D @ (rho_w * flux_vert * comp_harmonicmean(Avg,h_w)) -(fs_theta+H_fn)) # -kappa_edge @ G @ H 

    H      =  solve_lbvp(I,RHS,H_B,Param.H.g,H_N)

    phi_w,phi_i,T,dTdH = eval_phase_behaviorCwH(H,Tm,rho_i,rho_w,rho_nw,cp_i,cp_w,cp_nw,C,L_fusion)
    phi    = 1-phi_i
    phi_nw = 1 -phi_i -phi_w
    time = time + dt   
    
    #adding accumulation
    if (RetMIP_acc_rate(time/day2s) > 0) and (surf_loc > 0):  #if accumulation > 0 and surface is below top
        #length_added = RetMIP_acc_rate(time/day2s) * dt * rho_w /315 #'meter' ice equivalent of solid ice accumulation
        #length_added = RetMIP_acc_rate(time/day2s) * dt * rho_w /315 #'meter' ice equivalent of solid ice accumulation
        length_added = integrate.qmc_quad(RetMIP_acc_rate, (time-dt)/day2s, (time)/day2s)[0] * day2s * rho_w /315 #'meter' ice equivalent of solid ice accumulation #dt is involved
        
        
        if length_added_final < Grid.dy:  #insufficient to fill one cell
            length_added_final = length_added_final + length_added 
                
        else: #sufficient to fill one cell atleast
            if  length_added_final == 0:   length_added_final =  length_added 

            surf_loc = surf_loc - length_added_final     #new surface location [m]        
            dof_fresh = dof_complete_melt[np.ravel(Yc_col[dof_complete_melt-1] > surf_loc)] #cells of freshly fallen snow
            if np.any(dof_fresh): print(dof_fresh)
            H[dof_fresh-1] = 0.0   #adding temperate ice on top
            C[dof_fresh-1] = 315   #adding temperate, freshly fallen snow on top [kg/m3]
            phi_i[dof_fresh-1]   = 315/rho_i
            phi_w[dof_fresh-1]   = 0 
            phi_nw[dof_fresh-1]  = 1 - 315/rho_i
            print(time,'\t',length_added_final)
            length_added_final = 0  #re-setting the length to zero for next cycle
    ####################################################################################

    if np.isin(time,t_interest):
        t.append(time) 
        s_w_sol = np.concatenate((s_w_sol,s_w), axis=1) 
        H_sol   = np.concatenate((H_sol,H), axis=1) 
        T_sol   = np.concatenate((T_sol,T), axis=1) 
        phi_w_sol    = np.concatenate((phi_w_sol,phi_w), axis=1) 
        phi_i_sol   = np.concatenate((phi_i_sol,phi_i), axis=1) 
        q_w_new_sol  = np.concatenate((q_w_new_sol,flux_vert), axis=1) 
        
        '''
        if time%yr2s== 0:
            t_save = np.array(t.copy())
    
            ######################################################################
            #Saving the data
            ######################################################################
            np.savez(f'{simulation_name}_C{C_L}_{Grid.Nx}by{Grid.Ny}_t{tmax}.npz', t=t_save,q_w_new_sol=q_w_new_sol,H_sol=H_sol,T_sol=T_sol,s_w_sol=s_w_sol,phi_w_sol =phi_w_sol,phi_i_sol =phi_i_sol,phi=phi,Xc=Xc,Yc=Yc,Xc_col=Xc_col,Yc_col=Yc_col,Grid_Nx=Grid.Nx,Grid_Ny=Grid.Ny,Grid_xc=Grid.xc,Grid_yc=Grid.yc,Grid_xf=Grid.xf,Grid_yf=Grid.yf)
        '''      
        if len(dof_act)< Grid.N:
            print(i,time/day2s,'Saturated cells',Grid.N-len(dof_act))      
        else:    
            print(i,time/day2s)
    i = i+1
    

t = np.array(t)
tday = t/day2s


end_time = time_contour.perf_counter()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time:.4f} seconds")


######################################################################
#Saving the data
######################################################################
np.savez(f'{simulation_name}_C{C_L}_{Grid.Nx}by{Grid.Ny}_t{tmax}.npz', t=t,q_w_new_sol=q_w_new_sol,H_sol=H_sol,T_sol=T_sol,s_w_sol=s_w_sol,phi_w_sol =phi_w_sol,phi_i_sol =phi_i_sol,phi=phi,Xc=Xc,Yc=Yc,Xc_col=Xc_col,Yc_col=Yc_col,Grid_Nx=Grid.Nx,Grid_Ny=Grid.Ny,Grid_xc=Grid.xc,Grid_yc=Grid.yc,Grid_xf=Grid.xf,Grid_yf=Grid.yf)


np.savez(f'{simulation_name}_T_surface.npz', tday=tday,T_surf=T_sol[0,:])

'''
######################################################################
#for loading data
######################################################################
#data = np.load('Gravity_current_lowres_absolute_k0_onetimes_phiexp1.885_cond_well_high_res_nodiff_-30C_new_C0.03_80by40_t315576000.0_OG.npz') #no conduction
#data = np.load('test_Gravity_current_lowres_absolute_k0_onetimes_phiexp1.885_cond_well_high_res_yesdiff_-30C_new_C0.03_80by40_t315576000.0.npz') #with conduction
data = np.load('final_por_test_Gravity_current_lowres_absolute_k0_onetimes_phiexp1.885_cond_well_high_res_nodiff_-30C_new_C0.03_80by40_t315576000.0.npz') #without conduction porosity correct

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

import matplotlib.animation as animation
from matplotlib.ticker import LinearLocator, FormatStrFormatter
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'font.family': 'Serif'})
#Colors
brown  = [181/255 , 101/255, 29/255]
red    = [190/255 ,30/255 ,45/255 ]
blue   = [ 30/255 ,144/255 , 255/255 ]
green  = [  0/255 , 166/255 ,  81/255]
orange = [247/255 , 148/255 ,  30/255]
purple = [102/255 ,  45/255 , 145/255]
brown  = [155/255 ,  118/255 ,  83/255]
tan    = [199/255 , 178/255 , 153/255]
gray   = [100/255 , 100/255 , 100/255]


'''
#Multiple_time_plots
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,figsize=(12,8))
plt.subplot(2,2,1)
ii = np.argwhere(t==1*yr2s)[0,0]
cf = plt.contourf(Xc, Grid.ymax-Yc, np.transpose((phi_w_sol[:,ii]/(1-phi_i_sol[:,ii])).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1)
plt.ylabel(r'$z$ [m]')
#plt.xlim([Grid.xmin, 10])
plt.ylim([Grid.ymin,Grid.ymax])
plt.yticks([0, 50, 100,150,200])
#plt.axis('scaled')

plt.subplot(2,2,2)
ii = np.argwhere(t<3*yr2s)[-1]+1#np.argwhere(t<2*yr2s)[-1]+1#np.argwhere(t==2*yr2s)[0,0]#2 year
plot = [plt.contourf(Xc, Grid.ymax-Yc, np.transpose((phi_w_sol[:,ii]/(1-phi_i_sol[:,ii])).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1)]
plt.ylim([Grid.ymin,Grid.ymax])

plt.subplot(2,2,3)
ii = np.argwhere(t<5*yr2s)[-1]+1#np.argwhere(t==5*yr2s)[0,0]
plot = [plt.contourf(Xc, Grid.ymax-Yc, np.transpose((phi_w_sol[:,ii]/(1-phi_i_sol[:,ii])).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1)]
plt.ylim([Grid.ymin,Grid.ymax])
plt.yticks([0, 50, 100,150,200])
plt.ylabel(r'$z$ [m]')
plt.xlabel(r'$x$ [m]')
plt.subplot(2,2,4)
ii = np.argwhere(t==10*yr2s)[0,0]
plot = [plt.contourf(Xc, Grid.ymax-Yc, np.transpose((phi_w_sol[:,ii]/(1-phi_i_sol[:,ii])).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1)]
plt.ylim([Grid.ymin,Grid.ymax])
plt.xlabel(r'$x$ [m]')

plt.tight_layout(rect=[0, 0, 0.92, 1])
plt.subplots_adjust(wspace=0, hspace=0)
cbar = fig.colorbar(
    cf,
    ax=axes,
    orientation='vertical',
    fraction=0.04,
    pad=0.02
)
cbar.ax.set_title(r'$s_w$', fontsize=24, pad=3)
plt.savefig(f'../Figures/{simulation_name}_Marcs_way_Saturation.pdf',bbox_inches='tight', dpi = 600)
plt.savefig(f'../Figures/{simulation_name}_Marcs_way_Saturation.png',bbox_inches='tight', dpi = 600)
'''

#Multiple_time_plots
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,figsize=(12,8))
plt.subplot(2,2,1)
ii = np.argwhere(t==1*yr2s)[0,0]
cf = plt.contourf(
    Xc, Grid.ymax - Yc,
    np.transpose((phi_w_sol[:,ii]/(1-phi_i_sol[:,ii])).reshape(Grid.Nx,Grid.Ny)),
    levels=200,              # ← increase this
    cmap="Blues",
    vmin=0, vmax=1
)
h_analy = h_init*(t_init/t[ii])**alpha
x_analy = np.linspace(0,r_init,len(h_init))*(t[ii]/t_init)**beta
plt.plot(x_analy, h_analy,'k--',linewidth=6,color=red,label='Ana')
plt.ylabel(r'$z$ [m]')
#plt.xlim([Grid.xmin, 10])
plt.ylim([Grid.ymin,Grid.ymax])
plt.yticks([0, 50, 100,150,200])
#plt.axis('scaled')
plt.legend(loc='upper right',framealpha=0.0)

plt.subplot(2,2,2)
ii = np.argwhere(t<3*yr2s)[-1]+1#np.argwhere(t<2*yr2s)[-1]+1#np.argwhere(t==2*yr2s)[0,0]#2 year
plot = [plt.contourf(Xc, Grid.ymax-Yc, np.transpose((phi_w_sol[:,ii]/(1-phi_i_sol[:,ii])).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1)]
h_analy = h_init*(t_init/t[ii])**alpha
x_analy = np.linspace(0,r_init,len(h_init))*(t[ii]/t_init)**beta
plt.plot(x_analy, h_analy,'k--',linewidth=6,color=red)
plt.ylim([Grid.ymin,Grid.ymax])

plt.subplot(2,2,3)
ii = np.argwhere(t<5*yr2s)[-1]+1#np.argwhere(t==5*yr2s)[0,0]
plot = [plt.contourf(Xc, Grid.ymax-Yc, np.transpose((phi_w_sol[:,ii]/(1-phi_i_sol[:,ii])).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1)]
h_analy = h_init*(t_init/t[ii])**alpha
x_analy = np.linspace(0,r_init,len(h_init))*(t[ii]/t_init)**beta
plt.plot(x_analy, h_analy,'k--',linewidth=6,color=red)
plt.ylim([Grid.ymin,Grid.ymax])
plt.yticks([0, 50, 100,150,200])
plt.ylabel(r'$z$ [m]')
plt.xlabel(r'$x$ [m]')
plt.subplot(2,2,4)
ii = np.argwhere(t==10*yr2s)[0,0]
plot = [plt.contourf(Xc, Grid.ymax-Yc, np.transpose((phi_w_sol[:,ii]/(1-phi_i_sol[:,ii])).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1)]
h_analy = h_init*(t_init/t[ii])**alpha
x_analy = np.linspace(0,r_init,len(h_init))*(t[ii]/t_init)**beta
plt.plot(x_analy, h_analy,'k--',linewidth=6,color=red)
plt.ylim([Grid.ymin,Grid.ymax])
plt.xlabel(r'$x$ [m]')

plt.tight_layout(rect=[0, 0, 0.92, 1])
plt.subplots_adjust(wspace=0, hspace=0)
cbar = fig.colorbar(
    cf,
    ax=axes,
    orientation='vertical',
    fraction=0.04,
    pad=0.02
)
cbar.set_ticks(np.linspace(0, 1, 6))
cbar.ax.set_title(r'$s_w$', fontsize=24, pad=3)
plt.savefig(f'../Figures/{simulation_name}_Marcs_way_Saturation.pdf',bbox_inches='tight', dpi = 600)
plt.savefig(f'../Figures/{simulation_name}_Marcs_way_Saturation.png',bbox_inches='tight', dpi = 600)

##############################
#Estimate Volume and radius super slow
skip=100
Q_num=np.zeros(np.floor(len(t)/skip).astype(int));t_num=np.zeros(np.floor(len(t)/skip).astype(int))
h_num=np.zeros(np.floor(len(t)/skip).astype(int));x_num=np.zeros(np.floor(len(t)/skip).astype(int))
for iii in range(len(t_num)):
    Q_num[iii] = np.sum(phi_w_sol[:,iii*skip]*Grid.V)
    t_num[iii] = t[iii*skip]
    h_num[iii] = Grid.ymax-np.min(Yc_col[phi_w_sol[:,iii*skip]>1e-2])
    x_num[iii] = np.max(Xc_col[phi_w_sol[:,iii*skip]>1e-3])    
##############################


######################################################################
#Saving the data
######################################################################
np.savez(f'analysis_{simulation_name}_C{C_L}_{Grid.Nx}by{Grid.Ny}_t{tmax}.npz', Q_num,t_num,h_num,x_num)



############################################################
#new code (cold firn aquifer)
############################################################
phi_orig = 0.7
T  = -30 # Temperature of the aquifer [C]
Delta_phi_ = rho_i*cp_i*(phi_orig-1)/(rho_w*L_fusion)*(T_constant)  #calculating refrozen water (From Clark et al.,2017)

phi0 = phi_orig - Delta_phi_
Delta_rho = rho_w - rho_nw

S  = phi0**(m-1)*Delta_rho*grav*k0/mu_w  #Constant in constant porosity model (Huppert and Woods, 1994)
S1 = phi0**(m-1)*Delta_rho*grav*k0/mu_w  #Constant in constant porosity model (Huppert and Woods, 1994)
S2 = phi0**(m)*Delta_rho*grav*k0/(mu_w*(phi0+Delta_phi_))  #Constant in constant porosity model (Huppert and Woods, 1994)
kappabykappa1 = S2/S1 #the kappa ratio

##Analytic solution initialize
g_solution_, beta_ = Barenblatt_cartesian(kappabykappa1)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
plt.subplot(1,3,1)
#plt.loglog(t,h_max_analy,'r-',linewidth=3)
plt.plot(np.log10(t_num/t_num[0]), np.log10((t_num[0]/t_num)**(1 - 2*beta_)),'k--',linewidth=3,color=red,label=r'Ana')
plt.plot(np.log10(t_num/t_num[0]),np.log10(h_num/h_num[0]),'b-',linewidth=5,label='HydroFirn',color=blue,mfc='none',markersize=10)
plt.plot(np.log10(t_num/t_num[0]), np.log10((t_num[0]/t_num)**(1 - 2*beta_)),'k--',linewidth=3,color=red,alpha=0.75)
plt.ylabel(r'log $( h_{max}/h_{max,0})$')
plt.xlabel(r'log (t/t$_0)$')
#plt.ylim([-0.6,0.05])
plt.legend(loc='lower left',framealpha=0.0)

plt.subplot(1,3,2)
#plt.loglog(t,x_max_analy,'r-',linewidth=3)
plt.plot(np.log10(t_num/t_num[0]),np.log10(x_num/x_num[0]),'b-',linewidth=5,label='Simulation',color=blue,mfc='none',markersize=10)
plt.plot(np.log10(t_num/t_num[0]), np.log10((t_num[0]/t_num)**(-beta_)),'k--',linewidth=3,color=red,label=r'Ana')
plt.xlabel(r'$t$')
plt.ylabel(r'log $(x_{max}/x_{max,0})$')
plt.xlabel(r'log (t/t$_0$)')

plt.subplot(1,3,3)
plt.plot(np.log10(t_num/t_num[0]),np.log10(Q_num/Q_num[0]),'b-',linewidth=5,label='Simulation',color=blue,mfc='none',markersize=10)
plt.plot(np.log10(t_num/t_num[0]), np.log10((t_num/t_num[0])**(3*beta_-1)),'k--',linewidth=3,color=red,label=r'Analytic')
plt.xlabel(r'log (t/t$_0$)')
plt.ylabel(r'log $(Q/Q_{0})$')


#plt.axis('scaled')
#plt.ylim([-0.1,0.7])
#plt.legend(loc='best',framealpha=0.0)
plt.tight_layout()
#plt.xlim([Grid.xmin, 10])
plt.subplots_adjust(wspace=0.5, hspace=0)
plt.savefig(f'../Figures/volume_height_mound_cold_T{T}C_nonDim_diff.pdf',bbox_inches='tight', dpi = 600)



############################################################
#Include Vertically Integrated model results as well
############################################################
npz_path = "vertically-integrated-model-cold-firn_250by2_T-30C.npz"
from types import SimpleNamespace


with np.load(npz_path, allow_pickle=False) as data:
    VIM = SimpleNamespace(**{k: data[k] for k in data.files})

VIM.x = np.linspace(0,2.5*r_init,250)
#np.transpose(h_sol[:,i].reshape(Grid.Nx,Grid.Ny))[0,:]
#Multiple_time_plots
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,figsize=(12,8))
plt.subplot(2,2,1)
ii = np.argwhere(t==1*yr2s)[0,0]
h_analy = h_init*(t_init/t[ii])**alpha
x_analy = np.linspace(0,r_init,len(h_init))*(t[ii]/t_init)**beta
plt.plot(x_analy, h_analy,'k--',linewidth=6,color=red,label='Ana')
cf = plt.contourf(
    Xc, Grid.ymax - Yc,
    np.transpose((phi_w_sol[:,ii]/(1-phi_i_sol[:,ii])).reshape(Grid.Nx,Grid.Ny)),
    levels=200,              # ← increase this
    cmap="Blues",
    vmin=0, vmax=1
)

iiii = np.argwhere(VIM.t==1*yr2s)[0,0]
plt.plot(VIM.Grid_xc, np.transpose(VIM.h_sol[:,iiii].reshape(250,2))[0,:],'k-.',linewidth=4,color=green,alpha=0.8,label='CFA3D')
plt.legend(loc='best')
plt.ylabel(r'$z$ [m]')
#plt.xlim([Grid.xmin, 10])
plt.ylim([Grid.ymin,Grid.ymax])
plt.yticks([0, 50, 100,150,200])
#plt.axis('scaled')

plt.subplot(2,2,2)
ii = np.argwhere(t<3*yr2s)[-1]+1#np.argwhere(t<3*yr2s)[-1]+1#np.argwhere(t==2*yr2s)[0,0]
plot = [plt.contourf(Xc, Grid.ymax-Yc, np.transpose((phi_w_sol[:,ii]/(1-phi_i_sol[:,ii])).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1)]
h_analy = h_init*(t_init/t[ii])**alpha
x_analy = np.linspace(0,r_init,len(h_init))*(t[ii]/t_init)**beta
plt.plot(x_analy, h_analy,'k--',linewidth=6,color=red)

iiii = np.argwhere(VIM.t<3*yr2s)[-1]+1
plt.plot(VIM.Grid_xc, np.transpose(VIM.h_sol[:,iiii].reshape(250,2))[0,:],'k-.',linewidth=4,color=green,alpha=0.8)


plt.ylim([Grid.ymin,Grid.ymax])

plt.subplot(2,2,3)
ii = np.argwhere(t<5*yr2s)[-1]+1#np.argwhere(t==4*yr2s)[0,0]
plot = [plt.contourf(Xc, Grid.ymax-Yc, np.transpose((phi_w_sol[:,ii]/(1-phi_i_sol[:,ii])).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1)]
h_analy = h_init*(t_init/t[ii])**alpha
x_analy = np.linspace(0,r_init,len(h_init))*(t[ii]/t_init)**beta
plt.plot(x_analy, h_analy,'k--',linewidth=6,color=red)

iiii = np.argwhere(VIM.t<5*yr2s)[-1]+1
plt.plot(VIM.Grid_xc, np.transpose(VIM.h_sol[:,iiii].reshape(250,2))[0,:],'k-.',linewidth=4,color=green,alpha=0.8)

plt.ylim([Grid.ymin,Grid.ymax])
plt.yticks([0, 50, 100,150,200])
plt.ylabel(r'$z$ [m]')
plt.xlabel(r'$x$ [m]')
plt.subplot(2,2,4)
ii = np.argwhere(t<10*yr2s)[-1]+1#np.argwhere(t==7*yr2s)[0,0]
plot = [plt.contourf(Xc, Grid.ymax-Yc, np.transpose((phi_w_sol[:,ii]/(1-phi_i_sol[:,ii])).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1)]
h_analy = h_init*(t_init/t[ii])**alpha
x_analy = np.linspace(0,r_init,len(h_init))*(t[ii]/t_init)**beta
plt.plot(x_analy, h_analy,'k--',linewidth=6,color=red)

iiii = np.argwhere(VIM.t<10*yr2s)[-1]+1
plt.plot(VIM.Grid_xc, np.transpose(VIM.h_sol[:,iiii].reshape(250,2))[0,:],'k-.',linewidth=4,color=green,alpha=0.8)

plt.ylim([Grid.ymin,Grid.ymax])
plt.xlabel(r'$x$ [m]')

plt.tight_layout(rect=[0, 0, 0.92, 1])
plt.subplots_adjust(wspace=0, hspace=0)
cbar = fig.colorbar(
    cf,
    ax=axes,
    orientation='vertical',
    fraction=0.04,
    pad=0.02
)
cbar.set_ticks(np.linspace(0, 1, 6))
cbar.ax.set_title(r'$s_w$', fontsize=24, pad=3)
plt.savefig(f'../Figures/{simulation_name}_Marcs_way_Saturation_all.pdf',bbox_inches='tight', dpi = 600)
plt.savefig(f'../Figures/{simulation_name}_Marcs_way_Saturation_all.png',bbox_inches='tight', dpi = 600)


############################################################
#new code (cold firn aquifer)
############################################################
phi_orig = 0.7
T  = -30 # Temperature of the aquifer [C]
Delta_phi_ = rho_i*cp_i*(phi_orig-1)/(rho_w*L_fusion)*(T_constant)  #calculating refrozen water (From Clark et al.,2017)

phi0 = phi_orig - Delta_phi_
Delta_rho = rho_w - rho_nw

S  = phi0**(m-1)*Delta_rho*grav*k0/mu_w  #Constant in constant porosity model (Huppert and Woods, 1994)
S1 = phi0**(m-1)*Delta_rho*grav*k0/mu_w  #Constant in constant porosity model (Huppert and Woods, 1994)
S2 = phi0**(m)*Delta_rho*grav*k0/(mu_w*(phi0+Delta_phi_))  #Constant in constant porosity model (Huppert and Woods, 1994)
kappabykappa1 = S2/S1 #the kappa ratio


tlog = np.log10(VIM.t[1:]/VIM.t[0])
th_dimless = np.log10(VIM.h_max[1:]/VIM.h_max[0])
tr_dimless = np.log10(VIM.r_max[1:]/VIM.r_max[0])

##Analytic solution initialize
g_solution_, beta_ = Barenblatt_cartesian(kappabykappa1)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
plt.subplot(1,3,1)
#plt.loglog(t,h_max_analy,'r-',linewidth=3)
plt.plot(np.log10(t_num[1:]/t_num[0]), np.log10((t_num[0]/t_num[:-1])**(1 - 2*beta_)),'k--',color=red,linewidth=6,label=r'Analytic',alpha=0.75)
plt.plot(tlog,th_dimless,'b-.',label='CFA3D',linewidth=4,color=green,alpha=0.8,mfc='none',markersize=10)
plt.plot(np.log10(t_num[1:]/t_num[0]),np.log10(h_num[:-1]/h_num[0]),'b-',linewidth=5,label='HydroFirn',color=blue,mfc='none',markersize=10)

plt.ylabel(r'log $( h_{max}/h_{max,0})$')
plt.xlabel(r'log (t/t$_0)$')
#plt.ylim([-0.6,0.05])
plt.legend(loc='lower left',framealpha=0.0)

plt.subplot(1,3,2)
#plt.loglog(t,x_max_analy,'r-',linewidth=3)
plt.plot(np.log10(t_num[1:]/t_num[0]),np.log10(x_num[:-1]/x_num[0]),'b-',linewidth=5,label='Simulation',color=blue,mfc='none',markersize=10)
plt.plot(np.log10(t_num[1:]/t_num[0]), np.log10((t_num[0]/t_num[:-1])**(-beta_)),'k--',color=red,linewidth=6,label=r'Ana',alpha=0.75)
plt.plot(tlog,tr_dimless,'b-.',label='CFA3D',linewidth=4,color=green,alpha=0.8,mfc='none',markersize=10)
plt.xlabel(r'$t$')
plt.ylabel(r'log $(x_{max}/x_{max,0})$')
plt.xlabel(r'log (t/t$_0$)')

plt.subplot(1,3,3)
plt.plot(np.log10(t_num[1:]/t_num[0]),np.log10(Q_num[:-1]/Q_num[0]),'b-',linewidth=5,label='Simulation',color=blue,mfc='none',markersize=10)
plt.plot(np.log10(t_num[1:]/t_num[0]), np.log10((t_num[:-1]/t_num[0])**(3*beta_-1)),'k--',color=red,linewidth=6,label=r'Analytic',alpha=0.75)

Vol = np.sum(VIM.h_sol,0)*(VIM.Grid_xc[1]-VIM.Grid_xc[0])*phi0
plt.plot(tlog,np.log10(Vol[:-1]/Vol[0]),'b-.',label='CFA3D',linewidth=4,color=green,alpha=0.8,mfc='none',markersize=10)
plt.xlabel(r'log (t/t$_0$)')
plt.ylabel(r'log $(V/V_{0})$')


#plt.axis('scaled')
#plt.ylim([-0.1,0.7])
#plt.legend(loc='best',framealpha=0.0)
plt.tight_layout()
#plt.xlim([Grid.xmin, 10])
plt.subplots_adjust(wspace=0.5, hspace=0)
plt.savefig(f'../Figures/volume_height_mound_cold_T{T}C_nonDim_all_cond.pdf',bbox_inches='tight', dpi = 600)


####################################################################################





#New Contour plot
fps = 10000 # frame per sec
#frn = endstop # frame number of the animation
[N,frn] = np.shape(phi_w_sol) # frame number of the animation from the saved file

def update_plot(frame_number, zarray, plot,t):
    plt.clf()
    tyear_frame = t[frame_number]/yr2s
    plot[0] = plt.contourf(Xc, Grid.ymax - Yc, np.transpose(zarray[:,frame_number].reshape(Grid.Nx,Grid.Ny)), cmap="Blues",vmin = 0, vmax = 1)
    plt.title("t= %0.2f years" % tyear_frame,loc = 'center', fontsize=18)
    #plt.axis('scaled')
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    #clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
    plt.clim(0,1)
    #plt.plot(x(t[frame_number],xi),Grid.ymax - h_func(t[frame_number],xi,phi_0),'r--')
    h_analy = h_init*(t_init/t[frame_number])**alpha
    x_analy = np.linspace(0,r_init,len(h_init))*(t[frame_number]/t_init)**beta
    plt.plot(x_analy, h_analy,'k--',linewidth=6,color=red)

    
    plt.xlim([Grid.xmin, Grid.xmax])
    plt.ylim([Grid.ymin,Grid.ymax])
    mm = plt.cm.ScalarMappable(cmap=cm.Blues)
    mm.set_array(phi_w_sol[:,:]/(1-phi_i_sol[:,:]))
    mm.set_clim(0., 1.)
    clb = plt.colorbar()#mm, pad=0.1)
    clb.set_label(r'$s_w$', labelpad=-3,x=-3, y=1.13, rotation=0)
    plt.ylabel(r'$z$ [m]')
    plt.xlabel(r'$x$ [m]')
    

fig = plt.figure(figsize=(10,10) , dpi=100)
plot = [plt.contourf(Xc, Yc, np.transpose((phi_w_sol[:,0]/(1-phi_i_sol[:,0])).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1)]
h_analy = h_init
x_analy = np.linspace(0,r_init,len(h_init))
plt.plot(x_analy, h_analy,'k--',linewidth=6,color=red)
plt.ylabel(r'$z$ [m]')
plt.xlabel(r'$x$ [m]')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymin,Grid.ymax])
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_clim(0., 1.)
clb = plt.colorbar()#mm, pad=0.1)
clb.set_label(r'$s_w$', labelpad=-3,x=-3, y=1.13, rotation=0)
plt.title("t= %0.4f days" % t[0],loc = 'center', fontsize=18)
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
ii=100
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(phi_w_sol[:,::ii]/(1-phi_i_sol[:,::ii]), plot[::ii],t[::ii]), interval=1/fps)

ani.save(f"../Figures/{simulation_name}_{C_L/phi_L}_{Grid.Nx}by{Grid.Ny}_tf{t[frn-1]}_S_wsingle.mov", writer='ffmpeg', fps=30)




#New Contour plot
fps = 10000 # frame per sec
#frn = endstop # frame number of the animation
[N,frn] = np.shape(phi_w_sol) # frame number of the animation from the saved file

def update_plot(frame_number, zarray, plot,t):
    plt.clf()
    plot[0] = plt.contourf(Xc, Yc, np.transpose(zarray[:,frame_number].reshape(Grid.Nx,Grid.Ny)), cmap="Blues",vmin = 0, vmax = 1)
    plt.title("t= %0.4f days" % t[frame_number],loc = 'center', fontsize=18)
    #plt.axis('scaled')
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    #clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
    plt.clim(0,1)
    #plt.plot(x(t[frame_number],xi),Grid.ymax - h_func(t[frame_number],xi,phi_0),'r--')
    
    
    plt.xlim([Grid.xmin, Grid.xmax])
    plt.ylim([Grid.ymax,Grid.ymin])
    mm = plt.cm.ScalarMappable(cmap=cm.Blues)
    mm.set_array(phi_w_sol[:,:-1]/(1-phi_i_sol[:,:]))
    mm.set_clim(0., 1.)
    clb = plt.colorbar()#mm, pad=0.1)
    clb.set_label(r'$s_w$', labelpad=-3,x=-3, y=1.13, rotation=0)
    

fig = plt.figure(figsize=(10,10) , dpi=100)
plot = [plt.contourf(Xc, Yc, np.transpose((phi_w_sol[:,0]/(1-phi_i_sol[:,0])).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1)]
#mm.set_array(phi_w_sol[:,:]/(1-phi_i_sol[:,:]))
#plt.plot(x(t[0],xi),Grid.ymax - h_func(t[0],xi,phi_0),'r--')
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
#plt.axis('scaled')

mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_clim(0., 1.)
clb = plt.colorbar()#mm, pad=0.1)
clb.set_label(r'$s_w$', labelpad=-3,x=-3, y=1.13, rotation=0)
plt.title("t= %0.4f days" % t[0],loc = 'center', fontsize=18)
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
ii=100
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(phi_w_sol[:,::ii]/(1-phi_i_sol[:,::ii]), plot[::ii],t[::ii]), interval=1/fps)

ani.save(f"../Figures/{simulation_name}_{C_L/phi_L}_{Grid.Nx}by{Grid.Ny}_tf{t[frn-1]}_S_wsingle.mov", writer='ffmpeg', fps=30)






#Multiple_time_plots
fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=True,figsize=(10,20))
plt.subplot(4,1,1)
plot = [plt.contourf(Xc, Yc, np.transpose((T_sol[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Reds")]

ii = 0
h_analy = h_init*(t_init/t[ii])**alpha
x_analy = np.linspace(0,r_init,len(h_init))*(t[ii]/t_init)**beta
plt.plot(x_analy, Grid.ymax- h_analy,'k--',linewidth=5)

plt.ylabel(r'$z$')
#plt.xlim([Grid.xmin, 10])
#plt.yticks([0.25,0.5,0.75])
plt.ylim([Grid.ymax,Grid.ymin])
#plt.axis('scaled')

plt.subplot(4,1,2)
plot = [plt.contourf(Xc, Yc, np.transpose((T_sol[:,10000]).reshape(Grid.Nx,Grid.Ny)),cmap="Reds")]

ii = 10000
h_analy = h_init*(t_init/t[ii])**alpha
x_analy = np.linspace(0,r_init,len(h_init))*(t[ii]/t_init)**beta
plt.plot(x_analy, Grid.ymax-h_analy,'k--',linewidth=5)
plt.ylim([Grid.ymax,Grid.ymin])


plt.subplot(4,1,3)
plot = [plt.contourf(Xc, Yc, np.transpose((T_sol[:,20000]).reshape(Grid.Nx,Grid.Ny)),cmap="Reds")]
ii = 20000
h_analy = h_init*(t_init/t[ii])**alpha
x_analy = np.linspace(0,r_init,len(h_init))*(t[ii]/t_init)**beta
plt.plot(x_analy, Grid.ymax-h_analy,'k--',linewidth=5)
plt.ylabel(r'$z$')
plt.ylim([Grid.ymax,Grid.ymin])

plt.subplot(4,1,4)
plot = [plt.contourf(Xc, Yc, np.transpose((T_sol[:,40000]).reshape(Grid.Nx,Grid.Ny)),cmap="Reds")]
ii = 40000
h_analy = h_init*(t_init/t[ii])**alpha
x_analy = np.linspace(0,r_init,len(h_init))*(t[ii]/t_init)**beta
plt.plot(x_analy, Grid.ymax-h_analy,'k--',linewidth=5)
plt.xlabel(r'$x$')
plt.ylabel(r'$z$')
plt.ylim([Grid.ymax,Grid.ymin])
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(f'../Figures/{simulation_name}_Temp_combined_11yrs.pdf',bbox_inches='tight', dpi = 600)


'''
#New mid 
#Multiple_time_plots
fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=True,figsize=(10,20))
plt.subplot(4,1,1)
plot = [plt.contourf(Xc, Yc, np.transpose((T_sol[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Reds")]

ii = 0
h_analy = h_init*(t_init/t[ii])**alpha
x_analy = np.linspace(0,r_init,len(h_init))*(t[ii]/t_init)**beta
plt.plot(Grid.xmax/2+x_analy, Grid.ymax- h_analy,'k--',linewidth=5)

plt.ylabel(r'$z$')
#plt.xlim([Grid.xmin, 10])
#plt.yticks([0.25,0.5,0.75])
plt.ylim([Grid.ymax,Grid.ymin])
#plt.axis('scaled')

plt.subplot(4,1,2)
plot = [plt.contourf(Xc, Yc, np.transpose((T_sol[:,10000]).reshape(Grid.Nx,Grid.Ny)),cmap="Reds")]

ii = 10000
h_analy = h_init*(t_init/t[ii])**alpha
x_analy = np.linspace(0,r_init,len(h_init))*(t[ii]/t_init)**beta
plt.plot(Grid.xmax/2+x_analy, Grid.ymax-h_analy,'k--',linewidth=5)
plt.ylim([Grid.ymax,Grid.ymin])


plt.subplot(4,1,3)
plot = [plt.contourf(Xc, Yc, np.transpose((T_sol[:,20000]).reshape(Grid.Nx,Grid.Ny)),cmap="Reds")]
ii = 20000
h_analy = h_init*(t_init/t[ii])**alpha
x_analy = np.linspace(0,r_init,len(h_init))*(t[ii]/t_init)**beta
plt.plot(Grid.xmax/2+x_analy, Grid.ymax-h_analy,'k--',linewidth=5)
plt.ylabel(r'$z$')
plt.ylim([Grid.ymax,Grid.ymin])

plt.subplot(4,1,4)
plot = [plt.contourf(Xc, Yc, np.transpose((T_sol[:,40000]).reshape(Grid.Nx,Grid.Ny)),cmap="Reds")]
ii = 40000
h_analy = h_init*(t_init/t[ii])**alpha
x_analy = np.linspace(0,r_init,len(h_init))*(t[ii]/t_init)**beta
plt.plot(Grid.xmax/2+x_analy, Grid.ymax-h_analy,'k--',linewidth=5)
plt.xlabel(r'$x$')
plt.ylabel(r'$z$')
plt.ylim([Grid.ymax,Grid.ymin])
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(f'../Figures/{simulation_name}_Temp_combined_11yrs.pdf',bbox_inches='tight', dpi = 600)

'''




#New Contour plot
fps = 10000 # frame per sec
#frn = endstop # frame number of the animation
[N,frn] = np.shape(phi_w_sol) # frame number of the animation from the saved file

def update_plot(frame_number, zarray, plot,t):
    plt.clf()
    plot[0] = plt.contourf(Xc, Yc, np.transpose(zarray[:,frame_number].reshape(Grid.Nx,Grid.Ny)), cmap="Blues",vmin = 0, vmax = 1)
    plt.title("t= %0.4f days" % t[frame_number],loc = 'center', fontsize=18)
    #plt.axis('scaled')
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    #clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
    plt.clim(0,1)
    #plt.plot(x(t[frame_number],xi),Grid.ymax - h_func(t[frame_number],xi,phi_0),'r--')
    
    
    plt.xlim([Grid.xmin, Grid.xmax])
    plt.ylim([Grid.ymax,Grid.ymin])
    mm = plt.cm.ScalarMappable(cmap=cm.Blues)
    mm.set_array(phi_w_sol[:,:]/(1-phi_i_sol[:,:]))
    mm.set_clim(0., 1.)
    clb = plt.colorbar()#mm, pad=0.1)
    clb.set_label(r'$s_w$', labelpad=-3,x=-3, y=1.13, rotation=0)
    

fig = plt.figure(figsize=(10,10) , dpi=100)
plot = [plt.contourf(Xc, Yc, np.transpose((phi_w_sol[:,0]/(1-phi_i_sol[:,0])).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1)]
#mm.set_array(phi_w_sol[:,:]/(1-phi_i_sol[:,:]))
#plt.plot(x(t[0],xi),Grid.ymax - h_func(t[0],xi,phi_0),'r--')
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
#plt.axis('scaled')

mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_clim(0., 1.)
clb = plt.colorbar()#mm, pad=0.1)
clb.set_label(r'$s_w$', labelpad=-3,x=-3, y=1.13, rotation=0)
plt.title("t= %0.4f days" % t[0],loc = 'center', fontsize=18)
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
ii=100
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(phi_w_sol[:,::ii]/(1-phi_i_sol[:,::ii]), plot[::ii],t[::ii]), interval=1/fps)

ani.save(f"../Figures/{simulation_name}_{C_L/phi_L}_{Grid.Nx}by{Grid.Ny}_tf{t[frn-1]}_S_wsingle.mov", writer='ffmpeg', fps=30)






#Horizontal flux

[Xf_flux,Yc_flux] = np.meshgrid(Grid.xf,Grid.yc)     #building the (x,y) matrix

fig = plt.figure(figsize=(15,7.5) , dpi=100)
plot = [plt.contourf(Xf_flux, Yc_flux, np.transpose(flux_vert[0:Grid.Nfx,-1].reshape(Grid.Nx+1,Grid.Ny)),cmap="Blues",vmin = -1, vmax = 1)]
plt.ylabel(r'$z$ [m]')
plt.xlabel(r'$x$ [m]')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.colorbar()
#plt.axis('scaled')
plt.plot(Xc_col[np.argwhere(phi_w_sol[:,-1]<0)[:,0]],Yc_col[np.argwhere(phi_w_sol[:,-1]<0)[:,0]],'ro')
plt.savefig(f'../Figures/{simulation_name}_Marcs_way_horizontalflux.pdf',bbox_inches='tight', dpi = 600)



#Horizontal flux

[Xc_flux,Yf_flux] = np.meshgrid(Grid.xc,Grid.yf)     #building the (x,y) matrix

fig = plt.figure(figsize=(15,7.5) , dpi=100)
plot = [plt.contourf(Xc_flux, Yf_flux, np.transpose(flux_vert[Grid.Nfx:,-1].reshape(Grid.Nx,Grid.Ny+1)),cmap="Blues",vmin = -1, vmax = 1)]
plt.ylabel(r'$z$ [m]')
plt.xlabel(r'$x$ [m]')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.colorbar()
#plt.axis('scaled')
plt.plot(Xc_col[np.argwhere(phi_w_sol[:,-1]<0)[:,0]],Yc_col[np.argwhere(phi_w_sol[:,-1]<0)[:,0]],'ro')
plt.savefig(f'../Figures/{simulation_name}_Marcs_way_verticalflux.pdf',bbox_inches='tight', dpi = 600)


#Plotting contour plots
plt.figure()
plt.title(f't={tday[0]:.0f} days and t={tday[-1]:.0f} days')
plt.contourf(Xc, Yc, np.transpose((phi_w_sol[:,0]/(1-phi_i_sol[:,0])).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1)
plt.contourf(Xc, Yc, np.transpose((phi_w_sol[:,-1]/(1-phi_i_sol[:,-1])).reshape(Grid.Nx,Grid.Ny)),cmap="Reds",vmin = 0, vmax = 1,alpha=0.5,label=f'{tday[-1]}')
plt.ylabel(r'$z$ [m]')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.savefig(f'../Figures/initial_final_soil_sat.pdf',bbox_inches='tight', dpi = 600)


#Plotting contour plots
plt.figure()
plt.title(f't={tday[0]:.0f} days and t={tday[-1]:.0f} days')
plt.contourf(Xc, Yc, np.transpose((phi_w_sol[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1)
plt.contourf(Xc, Yc, np.transpose((phi_w_sol[:,-1]).reshape(Grid.Nx,Grid.Ny)),cmap="Reds",vmin = 0, vmax = 1,alpha=0.5,label=f'{tday[-1]}')
plt.ylabel(r'$z$ [m]')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.savefig(f'../Figures/initial_final_soil_moisture_content.pdf',bbox_inches='tight', dpi = 600)


#Plotting temperature contour plots
plt.figure()
plt.title(f't={tday[0]:.0f} days and t={tday[-1]:.0f} days')
plt.contourf(Xc, Yc, np.transpose((T_sol[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues")
plt.contourf(Xc, Yc, np.transpose((T_sol[:,-1]).reshape(Grid.Nx,Grid.Ny)),cmap="Reds",alpha=0.5)
plt.ylabel(r'$z$ [m]')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
#plt.colorbar()
plt.savefig(f'../Figures/initial_final_firn_Temp.pdf',bbox_inches='tight', dpi = 600)


'''
#New Contour plot
fps = 100000 # frame per sec
#frn = endstop # frame number of the animation
[N,frn] = np.shape(s_w_sol) # frame number of the animation from the saved file

def update_plot(frame_number, zarray, plot,tday): 
    plot[0] = plt.contourf(Xc, Yc, np.transpose(zarray[:,frame_number].reshape(Grid.Nx,Grid.Ny)), cmap="Blues",vmin = -0.05, vmax = 1.05)
    plt.title("t= %0.2f days" % tday[frame_number],loc = 'center')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    #plt.clim(0,1)
 
fig = plt.figure(figsize=(10,7.5) , dpi=100)
s_w_sol_new = phi_w_sol/(1-phi_i_sol)
plot = [plt.contourf(Xc, Yc, np.transpose(s_w_sol_new[:,0].reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = -0.05, vmax = 1.05)]
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
#mm = plt.cm.ScalarMappable(cmap=cm.Blues)
#mm.set_array(Ind)
#mm.set_clim(0., 1.)
#clb = plt.colorbar(mm, pad=0.0)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=0.0)
#clb.set_label(r'$s_w$')
plt.title("t= %0.2f days" % tday[0],loc = 'center', fontsize=18)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
skip =1000
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(s_w_sol_new[:,::skip], plot[::skip],tday[::skip]), interval=1/fps)
ani.save(f"../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_tf{t[frn-1]}_S_w.mov", writer='ffmpeg', fps=30)

#ani.save(fn+'.gif',writer='imagemagick',fps=fps)
#cmd = 'magick convert %s.gif -fuzz 5%% -layers Optimize %s_r.gif'%(fn,fn)
#subprocess.check_output(cmd)
'''

'''
t_iter = t[-100]
s_w_sol_new = phi_w_sol/(1-phi_i_sol)
s_w_sol_new_iter = s_w_sol_new[:,np.argwhere(t==t_iter)[0,0]]
plt.contourf(Xc, Yc, np.transpose((s_w_sol_new_iter).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",levels=100)
plt.colorbar()
'''

#Calculate maximum height and location of the front
phi_0 = phi_orig
x_max_analy = np.empty_like(t);h_max_analy = np.empty_like(t) 
x_max_sim   = np.empty_like(t);h_max_sim   = np.empty_like(t) 
s_w_sol_new = phi_w_sol/(1-phi_i_sol)
Xc_colnew = Xc_col.transpose(); Yc_colnew = Yc_col.transpose()
#Maximum height
for i,t_iter in enumerate(t[:-1]):
    #x_max_analy[i] = np.max(x(t_iter,xi))
    #h_max_analy[i] = np.max(h_func(t_iter,xi,phi_0))
    
    # s_w_sol_new_iter = s_w_sol_new[:,np.argwhere(t==t_iter)[0,0]]
    # Ind_iter = np.transpose(1*np.array([s_w_sol_new_iter>0.99]))
    # #plt.figure()
    # #plt.contourf(Xc, Yc, np.transpose((Ind_iter).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",levels=100)
    # x_max_sim[i] = np.max(Xc_col*Ind_iter)
    # h_max_sim[i] = Grid.ymax-np.min(Yc_col[Yc_col>10]*Ind_iter[Yc_col>10])
    # #print(i,t[i],x_max_sim[i],h_max_sim[i])
    
    s_w_iter = phi_w_sol[:,np.argwhere(t==t_iter)[0,0]]
    Ind_iter = 1*[s_w_iter>0.9*phi_0]
    x_max_sim[i] = np.max(Xc_colnew[Ind_iter])
    h_max_sim[i] = Grid.ymax-np.min(Yc_colnew[Ind_iter])  
    
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8,8))
plt.subplot(2,1,1)
#plt.loglog(t,h_max_analy,'r-',linewidth=3)
plt.loglog(t[:-1]/yr2s,h_max_sim[:-1],'b-',linewidth=3,label='Simulations, Present')
#plt.loglog(t[0:1],h_max_analy[0:1],'r-',linewidth=3,label='Analytic')
plt.ylabel(r'$h_{max} [m]$')
#plt.legend(loc='best',framealpha=0.0)
#plt.axis('scaled')

plt.subplot(2,1,2)
#plt.loglog(t,x_max_analy,'r-',linewidth=3)
plt.loglog(t[:-1]/yr2s,x_max_sim[:-1],'b-',linewidth=3)
#plt.loglog(t[0:1],x_max_analy[0:1],'r-',linewidth=3,label='Analytic')
plt.xlabel(r'$t$')
plt.ylabel(r'$x_{max} [m]$')
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.xlabel(r'$t$ [days]')
#plt.axis('scaled')
plt.tight_layout()
#plt.xlim([Grid.xmin, 10])
plt.savefig(f'../Figures/max_length_height_mound_cold.pdf',bbox_inches='tight', dpi = 600)





#Analyze data for Kochina
tday = t/day2s;
tyear = t/yr2s
step = -10000
Delta_phi = 0.06350981451791826#-rho_i*cp_i*(phi_0-1)/(rho_w*L_fusion)*(-T_constant) #drop in porosity using energy balance
phi_prime = phi_0 - Delta_phi                                    #new porosity

kappabykappa1 = (1-s_gr-s_wr)/(1-s_gr+rho_i/rho_w*Delta_phi/phi_prime)

tlog = np.log10(tday[step:-1])
th_dimless = np.log10(h_max_sim[step:-1]/h_max_sim[0])
tr_dimless = np.log10(x_max_sim[step:-1]/x_max_sim[0])
# Perform linear fit
coefficients = np.polyfit(tlog, tr_dimless,1)
m_r = coefficients[0]
b_r = coefficients[1]

print("Slope (m):", m_r)
print("Y-intercept (b):", b_r)

tr_dimless = np.log10(x_max_sim[:-1]/x_max_sim[0])
# Perform linear fit
coefficients = np.polyfit(tlog, th_dimless, 1)
m_h = coefficients[0]
b_h = coefficients[1]

print("Slope (m):", m_h)
print("Y-intercept (b):", b_h)


tlog = np.log10(tday[:-1])
th_dimless = np.log10(h_max_sim[:-1]/h_max_sim[0])
tr_dimless = np.log10(x_max_sim[:-1]/x_max_sim[0])


fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8,8))

red    = [190/255 ,30/255 ,45/255 ]
blue   = [ 30/255 ,144/255 , 255/255 ]

plt.subplot(2,1,1)
#plt.loglog(t,h_max_analy,'r-',linewidth=3)
plt.plot(np.log10(tday[:-1]),np.log10(h_max_sim[:-1]/h_max_sim[0]),'b-',linewidth=3,label='Simulations, Present',color=blue)
plt.plot(tlog, tlog*m_h + b_h,'r-',linewidth=3,label=f'Fit, α= %0.3f'%m_h,color=red,alpha=0.75)
plt.ylabel(r'$log_{10}[h_{max}/h_{max}(t=0)]$')
plt.ylim([-0.6,0.05])
plt.legend(loc='best',framealpha=0.0)
#plt.axis('scaled')

plt.subplot(2,1,2)
#plt.loglog(t,x_max_analy,'r-',linewidth=3)
plt.plot(np.log10(tday[:-1]),np.log10(x_max_sim[:-1]/x_max_sim[0]),'b-',linewidth=3,label='Simulations, Present',color=blue)
plt.plot(tlog, tlog*m_r + b_r,'r-',linewidth=3,label=f'Fit, β= %0.3f'%m_r,color=red,alpha=0.75)
plt.xlabel(r'$t$')
plt.ylabel(r'$log_{10}[x_{max}/x_{max}(t=0)]$')
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.xlabel(r'log t [log days]')
#plt.axis('scaled')
plt.ylim([-0.1,0.7])
plt.tight_layout()
#plt.xlim([Grid.xmin, 10])
plt.legend(loc='best',framealpha=0.0)
plt.savefig(f'../Figures/max_length_height_mound_cold.pdf',bbox_inches='tight', dpi = 600)




plt.figure()
plt.plot(Xc_col[T_sol[:,0]==np.max(T_sol)], Yc_col[T_sol[:,0]==np.max(T_sol)],'rx')


plt.figure()
plt.contourf(Xc, Yc, np.transpose((1-phi_i_sol[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Greys",vmin = 0, vmax = 1)
plt.colorbar()
plt.plot(Xc_col[phi_i_sol[:,0]>1-phi0-0.001], Yc_col[phi_i_sol[:,0]>1-phi0-0.001],'rx')


plt.figure()
plt.contourf(Xc, Yc, np.transpose((1-phi_i_sol[:,-1]).reshape(Grid.Nx,Grid.Ny)),cmap="Greys",vmin = 0, vmax = 1)
plt.colorbar()
plt.plot(Xc_col[phi_i_sol[:,-1]>1-phi0-0.001], Yc_col[phi_i_sol[:,-1]>1-phi0-0.001],'rx')






#Analysis
plt.figure()
plot = [plt.contourf(Xc, Yc, np.transpose((phi_orig-1+phi_i_sol[:,80000]).reshape(Grid.Nx,Grid.Ny)),cmap="Greys_r")]
plt.colorbar()
plt.ylabel(r'$z$')


#Analysis
plt.figure()
plot = [plt.contourf(Xc, Yc, np.transpose((T_sol[:,60000]).reshape(Grid.Nx,Grid.Ny)),cmap="Greys_r")]
plt.colorbar()
plt.ylabel(r'$z$')


Extra_porosity_generation_due_to_water_loss = (np.float64(0.06350981451791826)-0.05211105771248689)/(rho_i*cp_i*T_constant/(rho_w*L_fusion))

#phi_numeric = phi_orig - np.min((phi_sol[:,10000]))
















'''



from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
# ---------------------------
# Precompute main field
# ---------------------------
ii = 100  # stride
Sw = phi_w_sol[:, ::ii] / (1 - phi_i_sol[:, ::ii])
tt = t[::ii]
tyear = tt / yr2s

Nx, Ny = Grid.Nx, Grid.Ny
Zplot = Grid.ymax - Yc

# ---------------------------
# Scaling curves (use the FULL arrays, same indexing as your current version)
# ---------------------------
t_num = np.asarray(t_num)

# x-axis = log10(t/t0)
xlog = np.log10(t_num / t_num[0])

# simulation curves (log normalized)
y_h = np.log10(h_num / h_num[0])
y_x = np.log10(x_num / x_num[0])
y_Q = np.log10(Q_num / Q_num[0])

# analytic curves (log)
ana_h = np.log10((t_num[0] / t_num) ** (1 - 2 * beta_))
ana_x = np.log10((t_num[0] / t_num) ** (-beta_))
ana_Q = np.log10((t_num / t_num[0]) ** (3 * beta_ - 1))

# ------------------------------------------------------------
# Figure layout: big left + 3 stacked right
# ------------------------------------------------------------
fig = plt.figure(figsize=(14, 12), dpi=100)
fig.set_size_inches(14, 12)
gs = GridSpec(
    nrows=3, ncols=2,
    width_ratios=[3.2, 1.4],
    height_ratios=[1, 1, 1],
    wspace=0.55,
    hspace=0.18   # vertical spacing between right panels (increase to add more)
)

ax_big = fig.add_subplot(gs[:, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 1])
ax3 = fig.add_subplot(gs[2, 1])
fig.align_ylabels([ax1, ax2, ax3])

# ------------------------------------------------------------
# BIG panel: initial contour + analytic line
# ------------------------------------------------------------
field0 = Sw[:, 0].reshape(Nx, Ny).T
cf = ax_big.contourf(Xc, Zplot, field0, levels=100, cmap="Blues", vmin=0, vmax=1)

# analytic line on big panel
h_analy0 = h_init * (t_init / tt[0]) ** alpha
x_analy0 = np.linspace(0, r_init, len(h_init)) * (tt[0] / t_init) ** beta
line_an, = ax_big.plot(x_analy0, h_analy0, "--", linewidth=4, color=red)

ax_big.set_xlim([Grid.xmin, Grid.xmax])
ax_big.set_ylim([Grid.ymin, Grid.ymax])
ax_big.set_xlabel(r"$x$ [m]")
ax_big.set_ylabel(r"$z$ [m]")
title = ax_big.set_title(f"t = {tyear[0]:0.2f} years", fontsize=24)

# Colorbar with label on top
cbar = fig.colorbar(cf, ax=ax_big, fraction=0.046, pad=0.04)
cbar.ax.set_title(r"$s_w$", pad=10)
cbar.set_ticks(np.linspace(0, 1, 11))

# markers on big panel at (0,hmax) and (rmax,0)
star_hmax, = ax_big.plot([], [], marker="s", markersize=18,
                         markerfacecolor="none", markeredgecolor="k", markeredgewidth=2)
star_rmax, = ax_big.plot([], [], marker="o", markersize=18,
                         markerfacecolor="none", markeredgecolor="k", markeredgewidth=2)

# ------------------------------------------------------------
# Small panels: DYNAMIC curves (plot only up to marker)
# ------------------------------------------------------------
ax1.set_ylabel(r"log $\!\left(\frac{h_{max}}{h_{max,0}}\right)$")
ax2.set_ylabel(r"log $\!\left(\frac{x_{max}}{x_{max,0}}\right)$")
ax3.set_ylabel(r"log $\!(V/V_0)$")
ax3.set_xlabel(r"log  (t/t$_0$)")

# line artists
line1_full, = ax1.plot([], [], "-",  linewidth=4, color=blue)
line2_full, = ax2.plot([], [], "-",  linewidth=4, color=blue)
line3_full, = ax3.plot([], [], "-",  linewidth=4, color=blue)

line1_ana,  = ax1.plot([], [], "--", linewidth=3, color=red)
line2_ana,  = ax2.plot([], [], "--", linewidth=3, color=red)
line3_ana,  = ax3.plot([], [], "--", linewidth=3, color=red)

# markers on side panels
dot1, = ax1.plot([], [], marker="s", markersize=14,
                 markerfacecolor="none", markeredgecolor="k", markeredgewidth=2)
dot2, = ax2.plot([], [], marker="o", markersize=14,
                 markerfacecolor="none", markeredgecolor="k", markeredgewidth=2)
dot3, = ax3.plot([], [], marker="*", markersize=14,
                 markerfacecolor="none", markeredgecolor="k", markeredgewidth=2)

# axis limits for side panels
ax1.set_xlim(xlog[0]-0.05, xlog[-1]+0.05)
ax2.set_xlim(xlog[0]-0.05, xlog[-1]+0.05)
ax3.set_xlim(xlog[0]-0.05, xlog[-1]+0.05)

ax1.set_ylim(np.nanmin([y_h, ana_h])-0.03, np.nanmax([y_h, ana_h])+0.03)
ax2.set_ylim(np.nanmin([y_x, ana_x])-0.03, np.nanmax([y_x, ana_x])+0.03)
ax3.set_ylim(np.nanmin([y_Q, ana_Q])-0.001, np.nanmax([y_Q, ana_Q])+0.001)

# y ticks (3 each)
ax1.yaxis.set_major_locator(MaxNLocator(3))
ax2.yaxis.set_major_locator(MaxNLocator(3))
ax3.yaxis.set_major_locator(MaxNLocator(3))

# x ticks: keep ticks on all, but hide tick labels on first two
for ax in [ax1, ax2, ax3]:
    ax.set_xticks([0.0, 0.5, 1.0])

ax1.tick_params(labelbottom=False)   # ticks but no labels
ax2.tick_params(labelbottom=False)   # ticks but no labels
# ax3 keeps tick labels

# ------------------------------------------------------------
# Frame mapping
# ------------------------------------------------------------
def nearest_index(arr, val):
    return int(np.argmin(np.abs(arr - val)))

# ------------------------------------------------------------
# Update function
# ------------------------------------------------------------
def update(frame):
    # update big contour
    for coll in ax_big.collections:
        coll.remove()

    field = Sw[:, frame].reshape(Nx, Ny).T
    ax_big.contourf(Xc, Zplot, field, levels=100, cmap="Blues", vmin=0, vmax=1)

    # current time
    tnow = tt[frame]

    # update analytic line on big panel
    h_analy = h_init * (t_init / tnow) ** alpha
    x_analy = np.linspace(0, r_init, len(h_init)) * (tnow / t_init) ** beta
    line_an.set_data(x_analy, h_analy)

    # update title
    title.set_text(f"t = {tnow/yr2s:0.2f} years")

    # map to closest scaling index
    j = nearest_index(t_num, tnow)
    j = max(0, min(j, len(xlog) - 1))
    jj = j + 1

    # update side curves up to marker
    line1_full.set_data(xlog[:jj], y_h[:jj])
    line2_full.set_data(xlog[:jj], y_x[:jj])
    line3_full.set_data(xlog[:jj], y_Q[:jj])

    line1_ana.set_data(xlog[:jj], ana_h[:jj])
    line2_ana.set_data(xlog[:jj], ana_x[:jj])
    line3_ana.set_data(xlog[:jj], ana_Q[:jj])

    # move side markers
    dot1.set_data([xlog[j]], [y_h[j]])
    dot2.set_data([xlog[j]], [y_x[j]])
    dot3.set_data([xlog[j]], [y_Q[j]])

    # move big-panel markers at (0,hmax) and (rmax,0)
    star_hmax.set_data([Grid.dx / 2.0], [h_num[j]])
    star_rmax.set_data([x_num[j]], [Grid.dy / 2.0])

    return (
        [line_an, title,
         line1_full, line2_full, line3_full,
         line1_ana, line2_ana, line3_ana,
         dot1, dot2, dot3,
         star_hmax, star_rmax]
        + ax_big.collections
    )

# ------------------------------------------------------------
# Animate and save
# ------------------------------------------------------------
fps_out = 30
ani = animation.FuncAnimation(
    fig, update,
    frames=range(Sw.shape[1]),
    interval=1000 / fps_out,
    blit=False
)

# Save video (uncomment)
ani.save("big_plus_3sidepanels_dynamic_conduction.mp4", writer="ffmpeg", fps=fps_out, dpi=200)

plt.show()



#Temperature contour

from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
# ---------------------------
# Precompute main field
# ---------------------------
ii = 100  # stride
T_arr = T_sol[:, ::ii]
tt = t[::ii]
tyear = tt / yr2s

Nx, Ny = Grid.Nx, Grid.Ny
Zplot = Grid.ymax - Yc

# ---------------------------
# Scaling curves (use the FULL arrays, same indexing as your current version)
# ---------------------------
t_num = np.asarray(t_num)

# x-axis = log10(t/t0)
xlog = np.log10(t_num / t_num[0])

# simulation curves (log normalized)
y_h = np.log10(h_num / h_num[0])
y_x = np.log10(x_num / x_num[0])
y_Q = np.log10(Q_num / Q_num[0])

# analytic curves (log)
ana_h = np.log10((t_num[0] / t_num) ** (1 - 2 * beta_))
ana_x = np.log10((t_num[0] / t_num) ** (-beta_))
ana_Q = np.log10((t_num / t_num[0]) ** (3 * beta_ - 1))

# ------------------------------------------------------------
# Figure layout: big left + 3 stacked right
# ------------------------------------------------------------
fig = plt.figure(figsize=(14, 12), dpi=100)
gs = GridSpec(
    nrows=3, ncols=2,
    width_ratios=[3.2, 1.4],
    height_ratios=[1, 1, 1],
    wspace=0.55,
    hspace=0.18   # vertical spacing between right panels (increase to add more)
)

ax_big = fig.add_subplot(gs[:, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 1])
ax3 = fig.add_subplot(gs[2, 1])
fig.align_ylabels([ax1, ax2, ax3])

# ------------------------------------------------------------
# BIG panel: initial contour + analytic line
# ------------------------------------------------------------
field0 = T_arr[:, 0].reshape(Nx, Ny).T
cf = ax_big.contourf(Xc, Zplot, field0-Tm, levels=100, cmap="Reds", vmin=-30, vmax=0)

# analytic line on big panel
h_analy0 = h_init * (t_init / tt[0]) ** alpha
x_analy0 = np.linspace(0, r_init, len(h_init)) * (tt[0] / t_init) ** beta
line_an, = ax_big.plot(x_analy0, h_analy0, "--", linewidth=4, color=red)

ax_big.set_xlim([Grid.xmin, Grid.xmax])
ax_big.set_ylim([Grid.ymin, Grid.ymax])
ax_big.set_xlabel(r"$x$ [m]")
ax_big.set_ylabel(r"$z$ [m]")
title = ax_big.set_title(f"t = {tyear[0]:0.2f} years", fontsize=24)

# Colorbar with label on top
cbar = fig.colorbar(cf, ax=ax_big, fraction=0.046, pad=0.04)
cbar.ax.set_title(r"T[$^\circ C$]", pad=10)
cbar.set_ticks(np.linspace(-30, 0, 6))

# markers on big panel at (0,hmax) and (rmax,0)
star_hmax, = ax_big.plot([], [], marker="s", markersize=18,
                         markerfacecolor="none", markeredgecolor="k", markeredgewidth=2)
star_rmax, = ax_big.plot([], [], marker="o", markersize=18,
                         markerfacecolor="none", markeredgecolor="k", markeredgewidth=2)

# ------------------------------------------------------------
# Small panels: DYNAMIC curves (plot only up to marker)
# ------------------------------------------------------------
ax1.set_ylabel(r"log $\!\left(\frac{h_{max}}{h_{max,0}}\right)$")
ax2.set_ylabel(r"log $\!\left(\frac{x_{max}}{x_{max,0}}\right)$")
ax3.set_ylabel(r"log $\!(V/V_0)$")
ax3.set_xlabel(r"log  (t/t$_0$)")

# line artists
line1_full, = ax1.plot([], [], "-",  linewidth=4, color=blue)
line2_full, = ax2.plot([], [], "-",  linewidth=4, color=blue)
line3_full, = ax3.plot([], [], "-",  linewidth=4, color=blue)

line1_ana,  = ax1.plot([], [], "--", linewidth=3, color=red)
line2_ana,  = ax2.plot([], [], "--", linewidth=3, color=red)
line3_ana,  = ax3.plot([], [], "--", linewidth=3, color=red)

# markers on side panels
dot1, = ax1.plot([], [], marker="s", markersize=14,
                 markerfacecolor="none", markeredgecolor="k", markeredgewidth=2)
dot2, = ax2.plot([], [], marker="o", markersize=14,
                 markerfacecolor="none", markeredgecolor="k", markeredgewidth=2)
dot3, = ax3.plot([], [], marker="*", markersize=14,
                 markerfacecolor="none", markeredgecolor="k", markeredgewidth=2)

# axis limits for side panels
ax1.set_xlim(xlog[0]-0.05, xlog[-1]+0.05)
ax2.set_xlim(xlog[0]-0.05, xlog[-1]+0.05)
ax3.set_xlim(xlog[0]-0.05, xlog[-1]+0.05)

ax1.set_ylim(np.nanmin([y_h, ana_h])-0.03, np.nanmax([y_h, ana_h])+0.03)
ax2.set_ylim(np.nanmin([y_x, ana_x])-0.03, np.nanmax([y_x, ana_x])+0.03)
ax3.set_ylim(np.nanmin([y_Q, ana_Q])-0.001, np.nanmax([y_Q, ana_Q])+0.001)

# y ticks (3 each)
ax1.yaxis.set_major_locator(MaxNLocator(3))
ax2.yaxis.set_major_locator(MaxNLocator(3))
ax3.yaxis.set_major_locator(MaxNLocator(3))

# x ticks: keep ticks on all, but hide tick labels on first two
for ax in [ax1, ax2, ax3]:
    ax.set_xticks([0.0, 0.5, 1.0])

ax1.tick_params(labelbottom=False)   # ticks but no labels
ax2.tick_params(labelbottom=False)   # ticks but no labels
# ax3 keeps tick labels

# ------------------------------------------------------------
# Frame mapping
# ------------------------------------------------------------
def nearest_index(arr, val):
    return int(np.argmin(np.abs(arr - val)))

# ------------------------------------------------------------
# Update function
# ------------------------------------------------------------
def update(frame):
    # update big contour
    for coll in ax_big.collections:
        coll.remove()

    field = T_arr[:, frame].reshape(Nx, Ny).T
    ax_big.contourf(Xc, Zplot, field-Tm, levels=100, cmap="Reds", vmin=-30, vmax=0)

    # current time
    tnow = tt[frame]

    # update analytic line on big panel
    h_analy = h_init * (t_init / tnow) ** alpha
    x_analy = np.linspace(0, r_init, len(h_init)) * (tnow / t_init) ** beta
    line_an.set_data(x_analy, h_analy)

    # update title
    title.set_text(f"t = {tnow/yr2s:0.2f} years")

    # map to closest scaling index
    j = nearest_index(t_num, tnow)
    j = max(0, min(j, len(xlog) - 1))
    jj = j + 1

    # update side curves up to marker
    line1_full.set_data(xlog[:jj], y_h[:jj])
    line2_full.set_data(xlog[:jj], y_x[:jj])
    line3_full.set_data(xlog[:jj], y_Q[:jj])

    line1_ana.set_data(xlog[:jj], ana_h[:jj])
    line2_ana.set_data(xlog[:jj], ana_x[:jj])
    line3_ana.set_data(xlog[:jj], ana_Q[:jj])

    # move side markers
    dot1.set_data([xlog[j]], [y_h[j]])
    dot2.set_data([xlog[j]], [y_x[j]])
    dot3.set_data([xlog[j]], [y_Q[j]])

    # move big-panel markers at (0,hmax) and (rmax,0)
    star_hmax.set_data([Grid.dx / 2.0], [h_num[j]])
    star_rmax.set_data([x_num[j]], [Grid.dy / 2.0])

    return (
        [line_an, title,
         line1_full, line2_full, line3_full,
         line1_ana, line2_ana, line3_ana,
         dot1, dot2, dot3,
         star_hmax, star_rmax]
        + ax_big.collections
    )

# ------------------------------------------------------------
# Animate and save
# ------------------------------------------------------------
fps_out = 30
ani = animation.FuncAnimation(
    fig, update,
    frames=range(Sw.shape[1]),
    interval=1000 / fps_out,
    blit=False
)

# Save video (uncomment)
ani.save("big_plus_3sidepanels_dynamic_conduction_temp.mp4", writer="ffmpeg", fps=fps_out, dpi=200)

plt.show()




#combined all - porosity, temperature and saturation
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator

# ---------------------------
# Inputs you already have
# ---------------------------
# phi_w_sol, phi_i_sol, T_sol, t, yr2s
# Xc, Yc
# Grid.xmin, Grid.xmax, Grid.ymin, Grid.ymax, Grid.Nx, Grid.Ny
# h_init, t_init, r_init, alpha, beta
# Tm

# ---------------------------
# Precompute fields
# ---------------------------
ii = 100  # stride
Sw_arr   = phi_w_sol[:, ::ii] / (1 - phi_i_sol[:, ::ii])   # saturation
poro_arr = 1 - phi_i_sol[:, ::ii]                          # porosity proxy
T_arr    = T_sol[:, ::ii]                                  # temperature

tt = t[::ii]
tyear = tt / yr2s

Nx, Ny = Grid.Nx, Grid.Ny
Zplot = Grid.ymax - Yc

# ---------------------------
# Figure layout: 3 rows × 1 column
# ---------------------------
fig, axes = plt.subplots(
    3, 1,
    figsize=(7.5, 12),
    dpi=150,
    sharex=True,
    sharey=True
)
plt.subplots_adjust(hspace=0.12)

ax_sw, ax_poro, ax_T = axes

# ---------------------------
# Common axis formatting
# ---------------------------
for ax in axes:
    ax.set_xlim(Grid.xmin, Grid.xmax)
    ax.set_ylim(Grid.ymin, Grid.ymax)
    ax.tick_params(direction="in", top=True, right=True)

ax_sw.set_ylabel(r"$z$ [m]")
ax_poro.set_ylabel(r"$z$ [m]")
ax_T.set_ylabel(r"$z$ [m]")
ax_T.set_xlabel(r"$x$ [m]")

# ---------------------------
# Contour levels (smooth)
# ---------------------------
levels_sw = np.linspace(0, 1, 201)

poro_min, poro_max = np.nanmin(poro_arr), np.nanmax(poro_arr)
levels_poro = np.linspace(poro_min, poro_max, 201)

levels_T = np.linspace(-30, 0, 201)

# ---------------------------
# Initial contours
# ---------------------------
cf_sw = ax_sw.contourf(
    Xc, Zplot,
    Sw_arr[:, 0].reshape(Nx, Ny).T,
    levels=levels_sw, cmap="Blues", vmin=0, vmax=1
)

cf_poro = ax_poro.contourf(
    Xc, Zplot,
    poro_arr[:, 0].reshape(Nx, Ny).T,
    levels=levels_poro, cmap="Greys_r",  # <-- greys
    vmin=poro_min, vmax=poro_max
)

cf_T = ax_T.contourf(
    Xc, Zplot,
    (T_arr[:, 0].reshape(Nx, Ny).T - Tm),
    levels=levels_T, cmap="Reds", vmin=-30, vmax=0
)

# ---------------------------
# Analytic interface (red dashed) on ALL panels
# ---------------------------
tnow0 = tt[0]
h_ana0 = h_init * (t_init / tnow0) ** alpha
x_ana0 = np.linspace(0, r_init, len(h_init)) * (tnow0 / t_init) ** beta

line_sw,   = ax_sw.plot(x_ana0, h_ana0, "r--", lw=3)
line_poro, = ax_poro.plot(x_ana0, h_ana0, "r--", lw=3)
line_T,    = ax_T.plot(x_ana0, h_ana0, "r--", lw=3)

# ---------------------------
# Colorbars (thick, 3 ticks max, label on right)
#   - Sw has custom ticks: 0.25, 0.5, 0.75, 1.0
#   - Porosity label is φ
# ---------------------------

# Sw colorbar (custom ticks)
cbar_sw = fig.colorbar(cf_sw, ax=ax_sw, fraction=0.08, pad=0.03)
cbar_sw.set_label(r"$s_w$", rotation=270, labelpad=20)
cbar_sw.set_ticks([0.25, 0.5, 0.75, 1.0])

# Porosity colorbar (auto <=3 ticks, label φ)
cbar_poro = fig.colorbar(cf_poro, ax=ax_poro, fraction=0.08, pad=0.03)
cbar_poro.set_label(r"$\phi$", rotation=270, labelpad=20)
cbar_poro.locator = MaxNLocator(nbins=3, prune=None)
cbar_poro.update_ticks()

# Temperature colorbar (auto <=3 ticks)
cbar_T = fig.colorbar(cf_T, ax=ax_T, fraction=0.08, pad=0.03)
cbar_T.set_label(r"T[$^\circ C$]", rotation=270, labelpad=20)
cbar_T.locator = MaxNLocator(nbins=3, prune=None)
cbar_T.update_ticks()

# ---------------------------
# Shared time title
# ---------------------------
supt = fig.suptitle(f"t = {tyear[0]:0.2f} years", fontsize=22, y=0.995)

# ---------------------------
# Update function
# ---------------------------
def update(frame):
    # remove old contour collections
    for ax in axes:
        for coll in ax.collections:
            coll.remove()

    # redraw contours
    ax_sw.contourf(
        Xc, Zplot,
        Sw_arr[:, frame].reshape(Nx, Ny).T,
        levels=levels_sw, cmap="Blues", vmin=0, vmax=1
    )

    ax_poro.contourf(
        Xc, Zplot,
        poro_arr[:, frame].reshape(Nx, Ny).T,
        levels=levels_poro, cmap="Greys_r",
        vmin=poro_min, vmax=poro_max
    )

    ax_T.contourf(
        Xc, Zplot,
        (T_arr[:, frame].reshape(Nx, Ny).T - Tm),
        levels=levels_T, cmap="Reds", vmin=-30, vmax=0
    )

    # analytic interface
    tnow = tt[frame]
    h_ana = h_init * (t_init / tnow) ** alpha
    x_ana = np.linspace(0, r_init, len(h_init)) * (tnow / t_init) ** beta

    line_sw.set_data(x_ana, h_ana)
    line_poro.set_data(x_ana, h_ana)
    line_T.set_data(x_ana, h_ana)

    # update time title
    supt.set_text(f"t = {tyear[frame]:0.2f} years")
    return []

# ---------------------------
# Animate and save
# ---------------------------
fps_out = 30
ani = animation.FuncAnimation(
    fig, update,
    frames=range(Sw_arr.shape[1]),
    interval=1000 / fps_out,
    blit=False
)

ani.save(
    "Sw_phi_T_stacked_thick_colorbars.mov",
    writer="ffmpeg",
    fps=fps_out,
    dpi=200
)

plt.show()


'''




# ---------------------------
# Helper functions
# ---------------------------
def idx_nearest_time(arr_t, target):
    return int(np.argmin(np.abs(arr_t - target)))

def get_vim_profile_at_time(VIM, target_time):
    it = idx_nearest_time(VIM.t, target_time)

    hsol = VIM.h_sol
    if hsol.ndim == 2:
        Nx_vim = len(VIM.Grid_xc)
        Ny_vim = hsol.shape[0] // Nx_vim
        h2 = hsol[:, it].reshape(Nx_vim, Ny_vim)
        hprof = h2[:, 0]
    elif hsol.ndim == 3:
        hprof = hsol[:, 0, it]
    else:
        raise ValueError("Unsupported VIM.h_sol shape")

    return VIM.Grid_xc, hprof

# ---------------------------
# Times to plot
# ---------------------------
times_years = [1, 5, 10]
idxs = [idx_nearest_time(t, yy * yr2s) for yy in times_years]

Nx, Ny = Grid.Nx, Grid.Ny
Zplot = Grid.ymax - Yc

# ---------------------------
# Fields
# ---------------------------
Sw_all  = phi_w_sol / (1 - phi_i_sol)
phi_all = 1 - phi_i_sol
T_anom  = T_sol - Tm

# ---------------------------
# Masking rule
# ---------------------------
Sw_thr = 1e-2

# ---------------------------
# Color limits
# ---------------------------
Sw_vmin, Sw_vmax = 0.0, 1.0
T_vmin,  T_vmax  = -30.0, 0.0
phi_min = np.nanmin(phi_all)
phi_max = np.nanmax(phi_all)

# ---------------------------
# Figure: 3 × 2, no spacing
# ---------------------------
fig, axes = plt.subplots(
    3, 2,
    figsize=(10, 12),
    dpi=150,
    sharex=True,
    sharey=True
)
plt.subplots_adjust(wspace=0.0, hspace=0.0, bottom=0.32)

cf_T = cf_Sw = cf_phi = None

# ---------------------------
# Plot panels
# ---------------------------
for r, it in enumerate(idxs):
    tnow = t[it]

    # Analytic solution (orange dashed)
    h_ana = h_init * (t_init / tnow) ** alpha
    x_ana = np.linspace(0, r_init, len(h_init)) * (tnow / t_init) ** beta

    # CFA3D (green, semi-transparent)
    x_vim, h_vim = get_vim_profile_at_time(VIM, tnow)

    # -----------------------
    # LEFT: Temperature + Sw
    # -----------------------
    axL = axes[r, 0]

    Sw = Sw_all[:, it].reshape(Nx, Ny).T
    TmA = T_anom[:, it].reshape(Nx, Ny).T

    Sw_plot = Sw.copy()
    Sw_plot[Sw_plot < Sw_thr] = np.nan

    T_plot = TmA.copy()
    T_plot[Sw >= Sw_thr] = np.nan

    cf_T = axL.contourf(
        Xc, Zplot, T_plot,
        levels=200, cmap="Reds",
        vmin=T_vmin, vmax=T_vmax
    )
    cf_Sw = axL.contourf(
        Xc, Zplot, Sw_plot,
        levels=200, cmap="Blues",
        vmin=Sw_vmin, vmax=Sw_vmax
    )

    axL.plot(x_ana, h_ana,'k--',color=red,linewidth=6,alpha=0.75)
    axL.plot(x_vim, h_vim, 'b-.',linewidth=4,color=green,alpha=0.8,mfc='none',markersize=10)

    if r == 1:
        axL.set_ylabel(r"$z$ [m]")
    if r == 2:
        axL.set_xlabel(r"$x$ [m]")

    axL.tick_params(direction="in", top=True, right=True)

    # Legend INSIDE (only once)
    if r == 0:
        axL.legend(
            handles=[
                plt.Line2D([], [], linestyle="--", color=red,linewidth=6,alpha=0.75, label="Analytic"),
                plt.Line2D([], [], linestyle='-.',linewidth=4,color=green,alpha=0.8,mfc='none',markersize=10,label='CFA3D'),
            ],
            loc="upper right",
            frameon=False
        )

    # -----------------------
    # RIGHT: Porosity
    # -----------------------
    axR = axes[r, 1]
    phi = phi_all[:, it].reshape(Nx, Ny).T

    cf_phi = axR.contourf(
        Xc, Zplot, phi,
        levels=200, cmap="Greys_r",
        vmin=phi_min, vmax=phi_max
    )

    axR.plot(x_ana, h_ana, "--", color=red,linewidth=6,alpha=0.75)
    axR.plot(x_vim, h_vim, 'b-.',label='CFA3D',linewidth=4,color=green,alpha=0.8,mfc='none',markersize=10)

    if r == 2:
        axR.set_xlabel(r"$x$ [m]")

    axR.tick_params(direction="in", top=True, right=True)

# ---------------------------
# Limits & ticks cleanup
# ---------------------------
for ax in axes.ravel():
    ax.set_xlim(Grid.xmin, Grid.xmax)
    ax.set_ylim(Grid.ymin, Grid.ymax)

for r in range(3):
    axes[r, 1].tick_params(labelleft=False)
    if r < 2:
        axes[r, 0].tick_params(labelbottom=False)
        axes[r, 1].tick_params(labelbottom=False)

# ---------------------------
# Bottom colorbars
# ---------------------------
cax_T   = fig.add_axes([0.10, 0.12, 0.25, 0.025])
cax_Sw  = fig.add_axes([0.40, 0.12, 0.25, 0.025])
cax_phi = fig.add_axes([0.70, 0.12, 0.25, 0.025])

cb_T = fig.colorbar(cf_T, cax=cax_T, orientation="horizontal")
cb_T.set_label(r"T$[^\circ C]$")
cb_T.set_ticks(np.linspace(T_vmin, T_vmax, 4))

cb_Sw = fig.colorbar(cf_Sw, cax=cax_Sw, orientation="horizontal")
cb_Sw.set_label(r"$s_w$")
cb_Sw.set_ticks([0.25, 0.5,0.75, 1.0])

cb_phi = fig.colorbar(cf_phi, cax=cax_phi, orientation="horizontal")
cb_phi.set_label(r"$\phi$")
cb_phi.set_ticks([0.6, 0.65, 0.7])

plt.show()

# Optional save:
fig.savefig("T_Sw_combined_left_phi_right_with_CFA3D.pdf",
            dpi=300, bbox_inches="tight")

fig.savefig("T_Sw_combined_left_phi_right_with_CFA3D.png",
            dpi=300, bbox_inches="tight")







# ---------------------------
# Helper functions
# ---------------------------
def idx_nearest_time(arr_t, target):
    return int(np.argmin(np.abs(arr_t - target)))

def get_vim_profile_at_time(VIM, target_time):
    it = idx_nearest_time(VIM.t, target_time)

    hsol = VIM.h_sol
    if hsol.ndim == 2:
        Nx_vim = len(VIM.Grid_xc)
        Ny_vim = hsol.shape[0] // Nx_vim
        h2 = hsol[:, it].reshape(Nx_vim, Ny_vim)
        hprof = h2[:, 0]
    elif hsol.ndim == 3:
        hprof = hsol[:, 0, it]
    else:
        raise ValueError("Unsupported VIM.h_sol shape")

    return VIM.Grid_xc, hprof


# ---------------------------
# Times to plot
# ---------------------------
times_years = [1, 5, 10]
idxs = [idx_nearest_time(t, yy * yr2s) for yy in times_years]

Nx, Ny = Grid.Nx, Grid.Ny
Zplot = Grid.ymax - Yc

# ---------------------------
# Fields (NO masking)
# ---------------------------
Sw_all  = phi_w_sol / (1 - phi_i_sol)
phi_all = 1 - phi_i_sol
T_anom  = T_sol - Tm

# ---------------------------
# Color limits
# ---------------------------
Sw_vmin, Sw_vmax = 0.0, 1.0
T_vmin,  T_vmax  = -30.0, 0.0
phi_min = np.nanmin(phi_all)
phi_max = np.nanmax(phi_all)

# ---------------------------
# Figure: 3 × 3 (no spacing)
# ---------------------------
fig, axes = plt.subplots(
    3, 3,
    figsize=(12, 12),
    dpi=150,
    sharex=True,
    sharey=True
)

plt.subplots_adjust(wspace=0.0, hspace=0.0, bottom=0.26)

cf_Sw = cf_T = cf_phi = None

# ---------------------------
# Plot panels
# ---------------------------
for r, it in enumerate(idxs):
    tnow = t[it]

    # Analytic solution
    h_ana = h_init * (t_init / tnow) ** alpha
    x_ana = np.linspace(0, r_init, len(h_init)) * (tnow / t_init) ** beta

    # CFA3D solution
    x_vim, h_vim = get_vim_profile_at_time(VIM, tnow)

    Sw  = Sw_all[:, it].reshape(Nx, Ny).T
    TmA = T_anom[:, it].reshape(Nx, Ny).T
    phi = phi_all[:, it].reshape(Nx, Ny).T

    # -------- Column 1: Saturation --------
    ax = axes[r, 0]
    cf_Sw = ax.contourf(
        Xc, Zplot, Sw,
        levels=200, cmap="Blues",
        vmin=Sw_vmin, vmax=Sw_vmax
    )
    ax.plot(x_ana, h_ana, "--", color=red, linewidth=5,alpha=0.75)
    ax.plot(x_vim, h_vim, "-.", color=green, linewidth=4, alpha=0.8)

    # -------- Column 2: Temperature --------
    ax = axes[r, 1]
    cf_T = ax.contourf(
        Xc, Zplot, TmA,
        levels=200, cmap="Reds",
        vmin=T_vmin, vmax=T_vmax
    )
    #ax.plot(x_ana, h_ana, "--", color=red, linewidth=5,alpha=0.75)
    #ax.plot(x_vim, h_vim, "-.", color=green, linewidth=4, alpha=0.58)

    # -------- Column 3: Porosity --------
    ax = axes[r, 2]
    cf_phi = ax.contourf(
        Xc, Zplot, phi,
        levels=200, cmap="Greys_r",
        vmin=phi_min, vmax=phi_max
    )
    #ax.plot(x_ana, h_ana, "--", color=red, linewidth=5,alpha=0.75)
    #ax.plot(x_vim, h_vim, "-.", color=green, linewidth=4, alpha=0.8)

    # Labels
    if r == 2:
        for c in range(3):
            axes[r, c].set_xlabel(r"$x$ [m]")
    if r == 1:
        axes[r, 0].set_ylabel(r"$z$ [m]")

    for c in range(3):
        axes[r, c].tick_params(direction="in", top=True, right=True)

# ---------------------------
# Limits & ticks
# ---------------------------
for ax in axes.ravel():
    ax.set_xlim(Grid.xmin, Grid.xmax)
    ax.set_ylim(Grid.ymin, Grid.ymax)

for r in range(3):
    for c in range(1, 3):
        axes[r, c].tick_params(labelleft=False)
    if r < 2:
        for c in range(3):
            axes[r, c].tick_params(labelbottom=False)

# ---------------------------
# Legend (once, inside)
# ---------------------------
axes[0, 0].legend(
    handles=[
        plt.Line2D([], [], linestyle="--", color=red, linewidth=5,alpha=0.75, label="Analytic"),
        plt.Line2D([], [], linestyle="-.", color=green, linewidth=4, alpha=0.8, label="CFA3D"),
    ],
    loc="upper right",
    frameon=False
)

# ---------------------------
# Bottom colorbars (one per column)
# ---------------------------
cax_Sw  = fig.add_axes([0.10, 0.08, 0.22, 0.03])
cax_T   = fig.add_axes([0.39, 0.08, 0.22, 0.03])
cax_phi = fig.add_axes([0.68, 0.08, 0.22, 0.03])

cb_Sw = fig.colorbar(cf_Sw, cax=cax_Sw, orientation="horizontal")
cb_Sw.set_label(r"$s_w$")
cb_Sw.set_ticks([0.0, 0.5, 1.0])

cb_T = fig.colorbar(cf_T, cax=cax_T, orientation="horizontal")
cb_T.set_label(r"$T\;[^\circ\mathrm{C}]$")
cb_T.set_ticks(np.linspace(T_vmin, T_vmax, 4))

cb_phi = fig.colorbar(cf_phi, cax=cax_phi, orientation="horizontal")
cb_phi.set_label(r"$\phi$")
cb_phi.set_ticks(np.linspace(phi_min, phi_max, 3))
cb_phi.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.show()

# Optional save
fig.savefig("Sw_T_phi_3x3_CFA3D_vs_analytic_cond.png", dpi=300, bbox_inches="tight")
fig.savefig("Sw_T_phi_3x3_CFA3D_vs_analytic_cond.pdf", dpi=300, bbox_inches="tight")







# ---------------------------
# Helper functions
# ---------------------------
def idx_nearest_time(arr_t, target):
    return int(np.argmin(np.abs(arr_t - target)))

def get_vim_profile_at_time(VIM, target_time):
    it = idx_nearest_time(VIM.t, target_time)

    hsol = VIM.h_sol
    if hsol.ndim == 2:
        Nx_vim = len(VIM.Grid_xc)
        Ny_vim = hsol.shape[0] // Nx_vim
        h2 = hsol[:, it].reshape(Nx_vim, Ny_vim)
        hprof = h2[:, 0]
    elif hsol.ndim == 3:
        hprof = hsol[:, 0, it]
    else:
        raise ValueError("Unsupported VIM.h_sol shape")

    return VIM.Grid_xc, hprof


# ---------------------------
# Times to plot
# ---------------------------
times_years = [1, 5, 10]
idxs = [idx_nearest_time(t, yy * yr2s) for yy in times_years]

Nx, Ny = Grid.Nx, Grid.Ny
Zplot = Grid.ymax - Yc

# ---------------------------
# Fields (NO masking)
# ---------------------------
Sw_all  = phi_w_sol / (1 - phi_i_sol)
phi_all = 1 - phi_i_sol
T_anom  = T_sol - Tm

# ---------------------------
# Color limits
# ---------------------------
Sw_vmin, Sw_vmax = 0.0, 1.0
T_vmin,  T_vmax  = -30.0, 0.0
phi_min = np.nanmin(phi_all)
phi_max = np.nanmax(phi_all)

# ---------------------------
# Figure: 3 × 3 (no spacing)
# ---------------------------
fig, axes = plt.subplots(
    3, 3,
    figsize=(12, 12),
    dpi=150,
    sharex=True,
    sharey=True
)

# ---------------------------
# Set y-ticks
# ---------------------------
for ax in axes.ravel():
    ax.set_yticks([0, 200])

plt.subplots_adjust(wspace=0.0, hspace=0.0, bottom=0.26)

cf_Sw = cf_T = cf_phi = None

# ---------------------------
# Plot panels
# ---------------------------
for r, it in enumerate(idxs):
    tnow = t[it]

    # Analytic solution
    h_ana = h_init * (t_init / tnow) ** alpha
    x_ana = np.linspace(0, r_init, len(h_init)) * (tnow / t_init) ** beta

    # CFA3D solution
    x_vim, h_vim = get_vim_profile_at_time(VIM, tnow)

    Sw  = Sw_all[:, it].reshape(Nx, Ny).T
    TmA = T_anom[:, it].reshape(Nx, Ny).T
    phi = phi_all[:, it].reshape(Nx, Ny).T

    # -------- Column 1: Saturation --------
    ax = axes[r, 0]
    cf_Sw = ax.contourf(
        Xc, Zplot, Sw,
        levels=200, cmap="Blues",
        vmin=Sw_vmin, vmax=Sw_vmax
    )
    ax.plot(x_ana, h_ana, "--", color=red, linewidth=5,alpha=0.75)

    # -------- Column 2: Temperature --------
    ax = axes[r, 1]
    cf_T = ax.contourf(
        Xc, Zplot, TmA,
        levels=200, cmap="Reds",
        vmin=T_vmin, vmax=T_vmax
    )
    #ax.plot(x_ana, h_ana, "--", color=red, linewidth=5,alpha=0.75)
    #ax.plot(x_vim, h_vim, "-.", color=green, linewidth=4, alpha=0.58)

    # -------- Column 3: Porosity --------
    ax = axes[r, 2]
    cf_phi = ax.contourf(
        Xc, Zplot, phi,
        levels=200, cmap="Greys_r",
        vmin=phi_min, vmax=phi_max
    )
    #ax.plot(x_ana, h_ana, "--", color=red, linewidth=5,alpha=0.75)
    #ax.plot(x_vim, h_vim, "-.", color=green, linewidth=4, alpha=0.8)

    # Labels
    if r == 2:
        for c in range(3):
            axes[r, c].set_xlabel(r"$x$ [m]")
    if r == 1:
        axes[r, 0].set_ylabel(r"$z$ [m]")

    for c in range(3):
        axes[r, c].tick_params(direction="in", top=True, right=True)

# ---------------------------
# Limits & ticks
# ---------------------------
for ax in axes.ravel():
    ax.set_xlim(Grid.xmin, Grid.xmax)
    ax.set_ylim(Grid.ymin, Grid.ymax)

for r in range(3):
    for c in range(1, 3):
        axes[r, c].tick_params(labelleft=False)
    if r < 2:
        for c in range(3):
            axes[r, c].tick_params(labelbottom=False)

# ---------------------------
# Legend (once, inside)
# ---------------------------
axes[0, 0].legend(
    handles=[
        plt.Line2D([], [], linestyle="--", color=red, linewidth=5,alpha=0.75, label="Analytic"),
    ],
    loc="upper right",
    frameon=False
)

# ---------------------------
# Bottom colorbars (one per column)
# ---------------------------
cax_Sw  = fig.add_axes([0.10, 0.08, 0.22, 0.03])
cax_T   = fig.add_axes([0.39, 0.08, 0.22, 0.03])
cax_phi = fig.add_axes([0.68, 0.08, 0.22, 0.03])

cb_Sw = fig.colorbar(cf_Sw, cax=cax_Sw, orientation="horizontal")
cb_Sw.set_label(r"$s_w$")
cb_Sw.set_ticks([0.0, 0.5, 1.0])

cb_T = fig.colorbar(cf_T, cax=cax_T, orientation="horizontal")
cb_T.set_label(r"$T\;[^\circ\mathrm{C}]$")
cb_T.set_ticks(np.linspace(T_vmin, T_vmax, 4))

cb_phi = fig.colorbar(cf_phi, cax=cax_phi, orientation="horizontal")
cb_phi.set_label(r"$\phi$")
cb_phi.set_ticks(np.linspace(phi_min, phi_max, 3))
cb_phi.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.show()

# Optional save
fig.savefig("Sw_T_phi_3x3_CFA3D_vs_analytic_cond_new.png", dpi=300, bbox_inches="tight")
fig.savefig("Sw_T_phi_3x3_CFA3D_vs_analytic_cond_new.pdf", dpi=300, bbox_inches="tight")














''' 
#Final video
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FFMpegWriter
import matplotlib as mpl

mpl.rcParams["font.size"] = 18

# ------------------------------------------------------------
# Optional color defaults
# ------------------------------------------------------------
try:
    blue
except NameError:
    blue = "tab:blue"
try:
    red
except NameError:
    red = "tab:red"
try:
    green
except NameError:
    green = "tab:green"

# ------------------------------------------------------------
# Precompute fields
# ------------------------------------------------------------
ii = 100  # stride
Sw_arr   = phi_w_sol[:, ::ii] / (1 - phi_i_sol[:, ::ii])
poro_arr = 1 - phi_i_sol[:, ::ii]
T_arr    = T_sol[:, ::ii]

tt = t[::ii]
tyear = tt / yr2s

Nx, Ny = Grid.Nx, Grid.Ny
Zplot = Grid.ymax - Yc

# ------------------------------------------------------------
# Scaling curves on the right
# ------------------------------------------------------------
t_num = np.asarray(t_num)
xlog  = np.log10(t_num / t_num[0])

y_h = np.log10(h_num / h_num[0])
y_x = np.log10(x_num / x_num[0])
y_Q = np.log10(Q_num / Q_num[0])

ana_h = np.log10((t_num[0] / t_num) ** (1 - 2 * beta_))
ana_x = np.log10((t_num[0] / t_num) ** (-beta_))
ana_Q = np.log10((t_num / t_num[0]) ** (3 * beta_ - 1))

# ------------------------------------------------------------
# Color limits
# ------------------------------------------------------------
Sw_vmin, Sw_vmax = 0.0, 1.0
poro_min, poro_max = np.nanmin(poro_arr), np.nanmax(poro_arr)
T_vmin, T_vmax = -30.0, 0.0

levels_sw   = np.linspace(Sw_vmin, Sw_vmax, 201)
levels_poro = np.linspace(poro_min, poro_max, 201)
levels_T    = np.linspace(T_vmin, T_vmax, 201)

# ------------------------------------------------------------
# Figure layout
# Left side = 3 contour panels
# Right side = 3 scaling panels
# ------------------------------------------------------------
fig = plt.figure(figsize=(14, 12), dpi=100)
fig.set_size_inches(14, 12, forward=True)
fig.patch.set_facecolor("white")

gs = GridSpec(
    nrows=3,
    ncols=2,
    width_ratios=[3.4, 1.5],
    height_ratios=[1, 1, 1],
    wspace=0.50,
    hspace=0.18
)

# Left: contours
ax_sw   = fig.add_subplot(gs[0, 0])
ax_poro = fig.add_subplot(gs[1, 0], sharex=ax_sw, sharey=ax_sw)
ax_T    = fig.add_subplot(gs[2, 0], sharex=ax_sw, sharey=ax_sw)

# Right: scaling plots
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)

for ax in [ax_sw, ax_poro, ax_T, ax1, ax2, ax3]:
    ax.tick_params(axis="both", labelsize=20)

fig.align_ylabels([ax1, ax2, ax3])

for ax in fig.axes:
    ax.set_facecolor("white")

# ------------------------------------------------------------
# Initial contour panels
# ------------------------------------------------------------
cf_sw = ax_sw.contourf(
    Xc, Zplot,
    Sw_arr[:, 0].reshape(Nx, Ny).T,
    levels=levels_sw, cmap="Blues", vmin=Sw_vmin, vmax=Sw_vmax
)

cf_poro = ax_poro.contourf(
    Xc, Zplot,
    poro_arr[:, 0].reshape(Nx, Ny).T,
    levels=levels_poro, cmap="Greys_r", vmin=poro_min, vmax=poro_max
)

cf_T = ax_T.contourf(
    Xc, Zplot,
    (T_arr[:, 0].reshape(Nx, Ny).T - Tm),
    levels=levels_T, cmap="Reds", vmin=T_vmin, vmax=T_vmax
)

# ------------------------------------------------------------
# Analytic line on all contour panels
# ------------------------------------------------------------
tnow0 = tt[0]
h_ana0 = h_init * (t_init / tnow0) ** alpha
x_ana0 = np.linspace(0, r_init, len(h_init)) * (tnow0 / t_init) ** beta

line_sw_ana,   = ax_sw.plot(x_ana0, h_ana0, "--", linewidth=4, color=red)
#line_poro_ana, = ax_poro.plot(x_ana0, h_ana0, "--", linewidth=4, color=red)
#line_T_ana,    = ax_T.plot(x_ana0, h_ana0, "--", linewidth=4, color=red)

# Markers only on saturation panel
star_hmax, = ax_sw.plot(
    [], [], marker="s", markersize=18,
    markerfacecolor="none", markeredgecolor="k", markeredgewidth=2
)
star_rmax, = ax_sw.plot(
    [], [], marker="o", markersize=18,
    markerfacecolor="none", markeredgecolor="k", markeredgewidth=2
)

# ------------------------------------------------------------
# Axis formatting for contour panels
# ------------------------------------------------------------
for ax in [ax_sw, ax_poro, ax_T]:
    ax.set_xlim(Grid.xmin, Grid.xmax)
    ax.set_ylim(Grid.ymin, Grid.ymax)
    ax.tick_params(direction="in", top=True, right=True)

ax_sw.set_ylabel(r"$z$ [m]")
ax_poro.set_ylabel(r"$z$ [m]")
ax_T.set_ylabel(r"$z$ [m]")
ax_T.set_xlabel(r"$x$ [m]")

# Small internal labels
ax_sw.text(0.02, 0.92, r"Saturation, $s_w$", transform=ax_sw.transAxes, fontsize=20, va="top")
ax_poro.text(0.02, 0.92, r"Porosity, $\phi$", transform=ax_poro.transAxes, fontsize=20, va="top")
ax_T.text(0.02, 0.92, r"Temperature, $T\,[^\circ\mathrm{C}]$", transform=ax_T.transAxes, fontsize=20, va="top")


# ------------------------------------------------------------
# Colorbars for contour panels
# ------------------------------------------------------------
cbar_sw = fig.colorbar(cf_sw, ax=ax_sw, fraction=0.08, pad=0.02)
#cbar_sw.set_label(r"$s_w$", rotation=0, labelpad=8)
cbar_sw.set_ticks([0.25, 0.5, 0.75, 1.0])
cbar_sw.ax.tick_params(labelsize=20)

cbar_poro = fig.colorbar(cf_poro, ax=ax_poro, fraction=0.08, pad=0.02)
#cbar_poro.set_label(r"$\phi$", rotation=0, labelpad=8)
cbar_poro.locator = MaxNLocator(nbins=3, prune=None)
cbar_poro.update_ticks()
cbar_poro.ax.tick_params(labelsize=20)

cbar_T = fig.colorbar(cf_T, ax=ax_T, fraction=0.08, pad=0.02)
#cbar_T.set_label(r"$T\,[^\circ\mathrm{C}]$", rotation=0, labelpad=8)
cbar_T.locator = MaxNLocator(nbins=3, prune=None)
cbar_T.update_ticks()
cbar_T.ax.tick_params(labelsize=20)

# ------------------------------------------------------------
# Right-side scaling panels
# ------------------------------------------------------------
ax1.set_ylabel(r"$\log\!\left(\frac{h_{max}}{h_{max,0}}\right)$", fontsize=24)
ax2.set_ylabel(r"$\log\!\left(\frac{x_{max}}{x_{max,0}}\right)$", fontsize=24)
ax3.set_ylabel(r"$\log\!\left(\frac{V}{V_0}\right)$", fontsize=24)
ax3.set_xlabel(r"$\log(t/t_0)$", fontsize=24)

line1_full, = ax1.plot([], [], "-",  linewidth=4, color=blue,label='HydroFirn')
line2_full, = ax2.plot([], [], "-",  linewidth=4, color=blue)
line3_full, = ax3.plot([], [], "-",  linewidth=4, color=blue)

line1_ana, = ax1.plot([], [], "--", linewidth=3, color=red,label='Analytic')
line2_ana, = ax2.plot([], [], "--", linewidth=3, color=red)
line3_ana, = ax3.plot([], [], "--", linewidth=3, color=red)
ax1.legend(loc="lower left", frameon=False)

dot1, = ax1.plot([], [], marker="s", markersize=18,
                 markerfacecolor="none", markeredgecolor="k", markeredgewidth=2)
dot2, = ax2.plot([], [], marker="o", markersize=18,
                 markerfacecolor="none", markeredgecolor="k", markeredgewidth=2)
dot3, = ax3.plot([], [], marker="*", markersize=18,
                 markerfacecolor="none", markeredgecolor="k", markeredgewidth=2)

ax1.set_xlim(xlog[0] - 0.05, xlog[-1] + 0.05)
ax2.set_xlim(xlog[0] - 0.05, xlog[-1] + 0.05)
ax3.set_xlim(xlog[0] - 0.05, xlog[-1] + 0.05)

ax1.set_ylim(np.nanmin([y_h, ana_h]) - 0.03, np.nanmax([y_h, ana_h]) + 0.03)
ax2.set_ylim(np.nanmin([y_x, ana_x]) - 0.03, np.nanmax([y_x, ana_x]) + 0.03)
ax3.set_ylim(np.nanmin([y_Q, ana_Q]) - 0.001, np.nanmax([y_Q, ana_Q]) + 0.001)

# y ticks (3 each)
ax1.yaxis.set_major_locator(MaxNLocator(4))
ax2.yaxis.set_major_locator(MaxNLocator(4))
ax3.yaxis.set_major_locator(MaxNLocator(4))

for ax in [ax1, ax2, ax3]:
    ax.set_xticks([0.0, 0.5, 1.0])

ax1.tick_params(labelbottom=False)
ax2.tick_params(labelbottom=False)

# ------------------------------------------------------------
# Time title
# ------------------------------------------------------------
supt = fig.suptitle(f"t = {tyear[0]:0.2f} years", fontsize=24, y=0.995)

# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------
def nearest_index(arr, val):
    return int(np.argmin(np.abs(arr - val)))

# ------------------------------------------------------------
# Update function
# ------------------------------------------------------------
def update(frame):
    for ax in [ax_sw, ax_poro, ax_T]:
        for coll in ax.collections:
            coll.remove()

    ax_sw.contourf(
        Xc, Zplot,
        Sw_arr[:, frame].reshape(Nx, Ny).T,
        levels=levels_sw, cmap="Blues", vmin=Sw_vmin, vmax=Sw_vmax
    )

    ax_poro.contourf(
        Xc, Zplot,
        poro_arr[:, frame].reshape(Nx, Ny).T,
        levels=levels_poro, cmap="Greys_r", vmin=poro_min, vmax=poro_max
    )

    ax_T.contourf(
        Xc, Zplot,
        (T_arr[:, frame].reshape(Nx, Ny).T - Tm),
        levels=levels_T, cmap="Reds", vmin=T_vmin, vmax=T_vmax
    )

    tnow = tt[frame]

    h_ana = h_init * (t_init / tnow) ** alpha
    x_ana = np.linspace(0, r_init, len(h_init)) * (tnow / t_init) ** beta

    line_sw_ana.set_data(x_ana, h_ana)
    #line_poro_ana.set_data(x_ana, h_ana)
    #line_T_ana.set_data(x_ana, h_ana)

    j = nearest_index(t_num, tnow)
    j = max(0, min(j, len(xlog) - 1))
    jj = j + 1

    star_hmax.set_data([Grid.dx / 2.0], [h_num[j]])
    star_rmax.set_data([x_num[j]], [Grid.dy / 2.0])

    line1_full.set_data(xlog[:jj], y_h[:jj])
    line2_full.set_data(xlog[:jj], y_x[:jj])
    line3_full.set_data(xlog[:jj], y_Q[:jj])

    line1_ana.set_data(xlog[:jj], ana_h[:jj])
    line2_ana.set_data(xlog[:jj], ana_x[:jj])
    line3_ana.set_data(xlog[:jj], ana_Q[:jj])

    dot1.set_data([xlog[j]], [y_h[j]])
    dot2.set_data([xlog[j]], [y_x[j]])
    dot3.set_data([xlog[j]], [y_Q[j]])

    supt.set_text(f"t = {tyear[frame]:0.2f} years")

    return (
        [line_sw_ana, #line_poro_ana, line_T_ana,
         star_hmax, star_rmax,
         line1_full, line2_full, line3_full,
         line1_ana, line2_ana, line3_ana,
         dot1, dot2, dot3, supt]
        + ax_sw.collections + ax_poro.collections + ax_T.collections
    )

# ------------------------------------------------------------
# Animate and save
# ------------------------------------------------------------
fps_out = 30
ani = animation.FuncAnimation(
    fig, update,
    frames=range(Sw_arr.shape[1]),
    interval=1000 / fps_out,
    blit=False
)

writer = FFMpegWriter(
    fps=fps_out,
    codec="libx264",
    extra_args=["-pix_fmt", "yuv420p"]
)

ani.save("three_contours_plus_scaling_panels.mp4", writer=writer, dpi=300)

plt.show()
'''


