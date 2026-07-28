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
from complex_domain import find_crater_dofs

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
simulation_name = f'schematic_figure'
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
phi_L  = 0.7
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
C_L = 0.03
enable_outflow = True #enable outflow
rho_i_closeoff = 917#830 #closeoff porosity [kg/m3] 830 for later
phi_i_threshold = rho_i_closeoff/rho_i #cutoff porosity
Delta_phi = rho_i*cp_i*(phi_orig-1)/(rho_w*L_fusion)*(T_constant)  #calculating refrozen water (From Clark et al.,2017)
#Delta_phi = 0.06350981451791826  #from the numeric simulation


#temporal
tf     = 1*day2s
tmax = tf   #time scaling with respect to fc
t_interest = np.linspace(0,tmax,int(tf/day2s*24)+1)   #swr,sgr=0

Nt   = 100000
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
Grid.xmin =  0*z0; Grid.xmax =200; Grid.Nx=20;    #Horizontal direction
Grid.ymin =  0*z0; Grid.ymax =200; Grid.Ny=20;    #Vertically downward direction
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


def find_crater_dofs(x0,y0,r,Grid,Xc_col,Yc_col):
    dof_in = [];
    dist_center = (np.sqrt((x0-Xc_col)**2+(y0-Yc_col)**2))[:,0]
    dof_in =Grid.dof[dist_center <= r]
    dof_out = np.setdiff1d(Grid.dof,dof_in)
    return dof_in, dof_out


###############################

#phi_nw[Xc_col<1e3]= 0.0
#phi_nw  = fit_porosity(Yc_col) #volume fraction of gas 

phi_w   = np.zeros_like(phi_nw)  #volume fraction of water phase #No water
C       = rho_w * phi_w + (1 - phi_w - phi_nw) * rho_i
T =       T_constant*np.ones_like(Yc_col) + Tm


H       = enthalpyfromT(T_constant*np.ones_like(Yc_col)+Tm,Tm,rho_i,rho_w,0,cp_i,cp_w,0,phi_w,phi_nw,L_fusion)



##################################################################
#################Adding blobs
[dof_blob1,_] = find_crater_dofs(40,80,30,Grid,Xc_col,Yc_col)
[dof_blob2,_] = find_crater_dofs(160,80,30,Grid,Xc_col,Yc_col)
[dof_blob3,_] = find_crater_dofs(100,200,80,Grid,Xc_col,Yc_col)
dof_blob = np.concatenate([dof_blob1,dof_blob2,dof_blob3],axis=0)

phi_nw[dof_blob-1,0] = 0
phi_w[dof_blob-1,0] = phi_L

C[dof_blob-1,0] = rho_w * phi_w[dof_blob-1,0] + (1 - phi_w[dof_blob-1,0] - phi_nw[dof_blob-1,0]) * rho_i
T[dof_blob-1,0] = Tm 
H[dof_blob-1,0] = rho_w * phi_w[dof_blob-1,0] * L_fusion


##################################################################
#########Adding ice layer (ice layer has same temperature as the surroundings)
dof_ice_layer1 = np.intersect1d(np.intersect1d(Grid.dof[Xc_col[:,0]>=20],Grid.dof[Xc_col[:,0]<=70]),np.intersect1d(Grid.dof[Yc_col[:,0]>=110],Grid.dof[Yc_col[:,0]<=130]))
dof_ice_layer2 = np.intersect1d(np.intersect1d(Grid.dof[Xc_col[:,0]>=130],Grid.dof[Xc_col[:,0]<=180]),np.intersect1d(Grid.dof[Yc_col[:,0]>=110],Grid.dof[Yc_col[:,0]<=130]))
dof_ice_layer  = np.concatenate([dof_ice_layer1,dof_ice_layer2],axis=0)

phi_nw[dof_ice_layer-1,0] = 0
phi_w[dof_ice_layer-1,0]  = 0

C[dof_ice_layer-1,0] = rho_w * phi_w[dof_ice_layer-1,0] + (1 - phi_w[dof_ice_layer-1,0] - phi_nw[dof_ice_layer-1,0]) * rho_i
T[dof_ice_layer-1,0] = T_constant 
H[dof_ice_layer-1,0] = enthalpyfromT(np.array([[T_constant+Tm]]),Tm,rho_i,rho_w,0,cp_i,cp_w,0,np.array([[0]]),np.array([[0]]),L_fusion)[0,0]

##################################################################

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

t    =[0]
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
    #Param.H.qb= Qnet_func(time/day2s)*np.ones((len(dof_inj),1)) #1000*np.ones((len(dof_inj),1)) #
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
        dt2   = np.min(dt2[dt2>1e-2])     #changed dt2 from 1e-3 to 1e-2
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
data = np.load('schematic_figure_nodiff_-30C_new_C0.03_20by20_t86400.npz')
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






# -------------------------------------------------
# Build grid
# -------------------------------------------------
Grid = build_grid(Grid)
[D, G, I] = build_ops(Grid)

# -------------------------------------------------
# Derived fields
# -------------------------------------------------
phi_sol = 1.0 - phi_i_sol
times = [0, 1, 24]

# Saturation scale
sw_vmin, sw_vmax = 0.0, 1.0

# Porosity scale (global)
phi_vmin = float(np.min(phi_sol))
phi_vmax = float(np.max(phi_sol))

# Temperature scale (plot T - Tm)
T_anom = T_sol - Tm
T_vmin = float(np.min(T_anom))
T_vmax = float(np.max(T_anom))

# Fewer ticks on colorbars
ticks_sw = [0.0, 0.5, 1.0]
ticks_phi = [
    phi_vmin,
    0.5 * (phi_vmin + phi_vmax),
    phi_vmax
]
ticks_T = [
    T_vmin,
    0.5 * (T_vmin + T_vmax),
    T_vmax
]

# -------------------------------------------------
# Figure
# -------------------------------------------------
fig, axes = plt.subplots(
    nrows=4,
    ncols=3,
    sharex=True,
    sharey=True,
    figsize=(7, 12),
    dpi=120
)

# Zero spacing
fig.subplots_adjust(
    left=0.08,
    right=0.90,
    bottom=0.07,
    top=0.98,
    wspace=0.00,
    hspace=0.00
)

# Use the first saturation contour for its cmap/norm
mappable_sw = None

# -------------------------------------------------
# Plot panels
# -------------------------------------------------
for j, tt in enumerate(times):

    # -------------------------------------------------
    # Determine saturated and active cells
    # -------------------------------------------------
    dof_inact = Grid.dof[
        phi_w_sol[:, tt]
        / (phi_sol[:, tt] * (1.0 - s_gr))
        >= sat_threshold
    ]

    dof_act = np.setdiff1d(
        Grid.dof,
        dof_inact
    )

    # Saturated-region faces
    dof_f_saturated = find_faces(
        dof_inact,
        D,
        Grid
    )

    dof_sat_faces = find_all_faces(
        dof_inact,
        D,
        Grid
    )

    X_sat, Y_sat = comp_face_coords(
        dof_sat_faces,
        Grid
    )

    X_bnd, Y_bnd = comp_face_coords(
        dof_f_saturated,
        Grid
    )

    X_all, Y_all = comp_face_coords(
        Grid.dof_f,
        Grid
    )

    # Saturated boundary located on the upper domain boundary
    dof_f_saturated_domain_bnd = np.intersect1d(
        Grid.dof_f_ymax,
        dof_f_saturated
    )

    X_sat_dom_bnd, Y_sat_dom_bnd = comp_face_coords(
        dof_f_saturated_domain_bnd,
        Grid
    )

    # -------------------------------------------------
    # Ice-layer cells and faces
    # -------------------------------------------------
    dof_ice_layer = Grid.dof[
        phi_i_sol[:, tt] >= phi_i_threshold
    ]

    dof_ice_layer_faces = find_all_faces(
        dof_ice_layer,
        D,
        Grid
    )

    X_ice, Y_ice = comp_face_coords(
        dof_ice_layer_faces,
        Grid
    )

    dof_f_ice = find_faces(
        dof_ice_layer,
        D,
        Grid
    )

    X_ice_bnd, Y_ice_bnd = comp_face_coords(
        dof_f_ice,
        Grid
    )

    # =================================================
    # Row 1: Saturation contour and boundaries
    # =================================================
    ax = axes[0, j]

    sw_grid = (
        phi_w_sol[:, tt] / phi_sol[:, tt]
    ).reshape(
        Grid.Nx,
        Grid.Ny
    )

    cf_sw = ax.contourf(
        Xc,
        Yc,
        np.transpose(sw_grid),
        cmap="Blues",
        vmin=sw_vmin,
        vmax=sw_vmax,
        levels=100,
        zorder=0
    )

    if mappable_sw is None:
        mappable_sw = cf_sw

    # Thin black grid
    ax.plot(
        X_all,
        Y_all,
        color="black",
        linestyle="-",
        linewidth=0.4,
        zorder=2
    )

    # Thin ice-region faces
    ax.plot(
        X_ice,
        Y_ice,
        color="blue",
        linestyle="-",
        linewidth=0.4,
        zorder=3
    )

    # Thick saturated-region boundary
    ax.plot(
        X_bnd,
        Y_bnd,
        color="red",
        linestyle="-",
        linewidth=2,
        zorder=5
    )

    # Thick ice-layer boundary
    ax.plot(
        X_ice_bnd,
        Y_ice_bnd,
        color="blue",
        linestyle="-",
        linewidth=2,
        zorder=6
    )

    # Plot the green domain boundary LAST so it stays
    # above the black grid and axes spine.
    ax.plot(
        X_sat_dom_bnd,
        Y_sat_dom_bnd,
        color="green",
        linestyle="-",
        linewidth=3,
        zorder=100,
        solid_capstyle="round",
        solid_joinstyle="round",
        clip_on=False
    )

    ax.set_xlim([
        Grid.xmin,
        Grid.xmax
    ])

    ax.set_ylim([
        Grid.ymax + 0.005,
        Grid.ymin
    ])

    # =================================================
    # Row 2: Temperature anomaly contour
    # =================================================
    ax = axes[1, j]

    T_grid = T_anom[:, tt].reshape(
        Grid.Nx,
        Grid.Ny
    )

    ax.contourf(
        Xc,
        Yc,
        np.transpose(T_grid),
        cmap="Reds",
        vmin=T_vmin,
        vmax=T_vmax,
        levels=100
    )

    ax.set_xlim([
        Grid.xmin,
        Grid.xmax
    ])

    ax.set_ylim([
        Grid.ymax + 0.005,
        Grid.ymin
    ])

    # =================================================
    # Row 3: Porosity contour
    # =================================================
    ax = axes[2, j]

    phi_grid = phi_sol[:, tt].reshape(
        Grid.Nx,
        Grid.Ny
    )

    ax.contourf(
        Xc,
        Yc,
        np.transpose(phi_grid),
        cmap="Greys_r",
        vmin=phi_vmin,
        vmax=phi_vmax,
        levels=100
    )

    ax.set_xlim([
        Grid.xmin,
        Grid.xmax
    ])

    ax.set_ylim([
        Grid.ymax + 0.005,
        Grid.ymin
    ])

    # =================================================
    # Row 4: Classification and fake saturation background
    # =================================================
    ax = axes[3, j]

    fake = np.zeros((
        Grid.Nx,
        Grid.Ny
    ))

    ax.contourf(
        Xc,
        Yc,
        np.transpose(fake),
        cmap="Blues",
        vmin=sw_vmin,
        vmax=sw_vmax,
        levels=100,
        zorder=0
    )

    # Thin black grid
    ax.plot(
        X_all,
        Y_all,
        color="black",
        linestyle="-",
        linewidth=0.4,
        zorder=2
    )

    # Thin saturated-region faces
    ax.plot(
        X_sat,
        Y_sat,
        color="red",
        linestyle="-",
        linewidth=0.4,
        zorder=3
    )

    # Thin ice-region faces
    ax.plot(
        X_ice,
        Y_ice,
        color="blue",
        linestyle="-",
        linewidth=0.4,
        zorder=3
    )

    # Thick saturated-region boundary
    ax.plot(
        X_bnd,
        Y_bnd,
        color="red",
        linestyle="-",
        linewidth=2,
        zorder=5
    )

    # Thick ice-layer boundary
    ax.plot(
        X_ice_bnd,
        Y_ice_bnd,
        color="blue",
        linestyle="-",
        linewidth=2,
        zorder=6
    )

    # Inactive/saturated cells
    ax.plot(
        Xc_col[dof_inact - 1, 0],
        Yc_col[dof_inact - 1, 0],
        "ro",
        markerfacecolor="none",
        markersize=2,
        zorder=10
    )

    # Active cells
    ax.plot(
        Xc_col[dof_act - 1, 0],
        Yc_col[dof_act - 1, 0],
        "go",
        markerfacecolor="none",
        markersize=2,
        zorder=10
    )

    # Ice-layer cells
    ax.plot(
        Xc_col[dof_ice_layer - 1, 0],
        Yc_col[dof_ice_layer - 1, 0],
        "bo",
        markerfacecolor="none",
        markersize=2,
        zorder=10
    )

    # Plot the green boundary LAST and above all other artists
    ax.plot(
        X_sat_dom_bnd,
        Y_sat_dom_bnd,
        color="green",
        linestyle="-",
        linewidth=3,
        zorder=100,
        solid_capstyle="round",
        solid_joinstyle="round",
        clip_on=False
    )

    ax.set_xlim([
        Grid.xmin,
        Grid.xmax
    ])

    ax.set_ylim([
        Grid.ymax + 0.005,
        Grid.ymin
    ])

# -------------------------------------------------
# Axis formatting
#   - show ticks everywhere
#   - labels only on bottom row and first column
#   - ticks inside and outside
# -------------------------------------------------
xticks = [0, 100, 200]
yticks = [0, 100, 200]

for r in range(4):
    for c in range(3):

        ax = axes[r, c]

        ax.set_aspect(
            "equal",
            adjustable="box"
        )

        # Set tick locations everywhere
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        # Ticks inside and outside
        ax.tick_params(
            axis="both",
            direction="inout",
            length=6,
            width=1
        )

        # Keep axes spines below the green boundary
        for spine in ax.spines.values():
            spine.set_zorder(1)

        # ---------- X axis ----------
        if r != 3:
            ax.tick_params(
                labelbottom=False
            )
        else:
            ax.set_xlabel(r"$x$ [m]")

            for lbl in ax.get_xticklabels():
                lbl.set_rotation(30)
                lbl.set_ha("right")

        # ---------- Y axis ----------
        if c != 0:
            ax.tick_params(
                labelleft=False
            )
        else:
            ax.set_ylabel(r"$z$ [m]")

# -------------------------------------------------
# Colorbars
#   - sw: use actual contourf mappable
#   - phi and T: use ScalarMappable
# -------------------------------------------------
cbar_sw = fig.colorbar(
    mappable_sw,
    ax=[
        axes[0, 0],
        axes[0, 1],
        axes[0, 2],
        axes[1, 0],
        axes[1, 1],
        axes[1, 2]
    ],
    fraction=0.025,
    pad=0.02
)

cbar_sw.set_ticks(ticks_sw)
cbar_sw.set_label(
    r"$s_w$",
    rotation=0,
    labelpad=10
)

mm_phi = cm.ScalarMappable(
    cmap="Greys_r",
    norm=plt.Normalize(
        vmin=phi_vmin,
        vmax=phi_vmax
    )
)

mm_phi.set_array([])

cbar_phi = fig.colorbar(
    mm_phi,
    ax=axes[2, :].ravel().tolist(),
    fraction=0.025,
    pad=0.02
)

cbar_phi.set_ticks(ticks_phi)
cbar_phi.set_label(
    r"$\phi$",
    rotation=0,
    labelpad=10
)

cbar_phi.ax.yaxis.set_major_formatter(
    FormatStrFormatter("%.2f")
)

mm_T = cm.ScalarMappable(
    cmap="Reds",
    norm=plt.Normalize(
        vmin=T_vmin,
        vmax=T_vmax
    )
)

mm_T.set_array([])

cbar_T = fig.colorbar(
    mm_T,
    ax=axes[3, :].ravel().tolist(),
    fraction=0.025,
    pad=0.02
)

cbar_T.set_ticks(ticks_T)
cbar_T.set_label(
    r"$T[^\circ C]$",
    rotation=0,
    labelpad=10
)

cbar_T.ax.yaxis.set_major_formatter(
    FormatStrFormatter("%.0f")
)

# -------------------------------------------------
# Save figure
# -------------------------------------------------
plt.savefig(
    f"../Figures/{simulation_name}_4x3_multiphysics.pdf",
    bbox_inches="tight",
    dpi=600
)

plt.savefig(
    f"../Figures/{simulation_name}_4x3_multiphysics.png",
    bbox_inches="tight",
    dpi=600
)

plt.show()





