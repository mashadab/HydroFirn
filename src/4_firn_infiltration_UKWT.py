#Coding the Hyperbolic Richard's solver with CH thermodynamics
#Multiple_layers
#Mohammad Afzal Shadab
#Date modified: 05/03/2022

import sys
sys.path.insert(1, '../Solver')

# import personal libraries and class
from classes import *    

from solve_lbvpfun_SPD import solve_lbvp_SPD
from complex_domain import find_faces, find_bnd_cells, find_all_faces, find_all_x_faces,find_all_y_faces
from comp_fluxfun import comp_flux
from scipy.integrate import solve_ivp
from comp_sat_unsat_bnd_flux_fun import comp_sat_unsat_bnd_flux, find_top_bot_cells
from comp_face_coords_fun import comp_face_coords
from comp_mean_matrix import comp_mean_matrix
from eval_phase_behavior import eval_phase_behaviorCwH, enthalpyfromT, eval_h
from matplotlib import ticker

#for 2D
from build_gridfun2D import build_grid

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

class Param:
    def __init__(self):
        self.dof_dir = np.array([])       # identify cells on Dirichlet bnd
        self.dof_f_dir = np.array([])     # identify faces on Dirichlet bnd
        self.dof_neu = np.array([])       # identify cells on Neumann bnd
        self.dof_f_neu = np.array([])     # identify faces on Neumann bnd
        self.g = np.array([])             # column vector of non-homogeneous Dirichlet BCs (Nc X 1)

class Param:
    class P:
        def __init__(self):
            self.dof_dir = np.array([])       # identify cells on Dirichlet bnd
            self.dof_f_dir = np.array([])     # identify faces on Dirichlet bnd
            self.dof_neu = np.array([])       # identify cells on Neumann bnd
            self.dof_f_neu = np.array([])     # identify faces on Neumann bnd
            self.g = np.array([])             # column vector of non-homogeneous Dirichlet BCs (Nc X 1)
            self.qb = np.array([])         

    class S:
        def __init__(self):
            self.dof_dir = np.array([])       # identify cells on Dirichlet bnd
            self.dof_f_dir = np.array([])     # identify faces on Dirichlet bnd
            self.dof_neu = np.array([])       # identify cells on Neumann bnd
            self.dof_f_neu = np.array([])     # identify faces on Neumann bnd
            self.g = np.array([])             # column vector of non-homogeneous Dirichlet BCs (Nc X 1)
            self.qb = np.array([])

    class H:
        def __init__(self):
            self.dof_dir = np.array([])       # identify cells on Dirichlet bnd
            self.dof_f_dir = np.array([])     # identify faces on Dirichlet bnd
            self.dof_neu = np.array([])       # identify cells on Neumann bnd
            self.dof_f_neu = np.array([])     # identify faces on Neumann bnd
            self.g = np.array([])             # column vector of non-homogeneous Dirichlet BCs (Nc X 1)
            self.qb = np.array([])


#parameters
##simulation
simulation_name = f'new_Discussion_figure_-20C'
diffusion = 'no'
CFL    = 0.1     #CFL number
tilt_angle = 0   #angle of the slope
day2s = 24*60*60

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

[C_L,phi_L] = [0.4,0.73551017] #water volume fraction, porosity at the surface
npp    = 4   #number of positive days [days]

##thermodynamics
T_firn = -20    # temperature of firn [C]
rho_w = 1000    # density of water [kg/m^3]
cp_w  = 4186    # specific heat of water at constant pressure [J/(kg K)]
k_w   = 0.606   # coefficient of thermal conductivity of water [W / (m K)]
rho_nw = 1.225  # density of gas [kg/m^3]
phi_nw_init = 0.7    # volumetric ratio of gas, decreasing exponentially
cp_nw  = 1003.5  # specific heat of gas at constant pressure [J/(kg K)]
rho_i = 1000     # average density of ice cap [kg/m^3]
cp_i  = 2106.1  # specific heat of ice at constant pressure [J/(kg K)]
k_i   = 2.25    # coefficient of thermal conductivity of ice [W / (m K)]
kk    = k_i/(rho_i*cp_i) # thermal diffusivity of ice
Tm    = 273.16  # melting point temperature of ice [K]
L_fusion= 333.55e3# latent heat of fusion of water [J / kg]

#domain details
z0 = 3.75 #characteristic height (m)

fc = k0*k_w0*rho_w*grav/mu_w*phi_L**3 #Infiltration capacity (m/s)

sat_threshold = 1-1e-4 #threshold for saturated region formation

#injection
Param.xleft_inj= 0e3;  Param.xright_inj= 1000e3

#temporal
tf     = 1*day2s
tmax = tf
t_interest = np.linspace(0,tmax,int(tf/day2s*24)+1)   #swr,sgr=0

#tmax = tmax / phi_L**m   #time scaling with respect to K_0 where K_0 = f_c/phi**m
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

#spatial
Grid.xmin =  0*z0; Grid.xmax =1000e3; Grid.Nx = 2; 
Grid.ymin =  0*z0; Grid.ymax =10;  Grid.Ny = 800;
Grid = build_grid(Grid)
[D,G,I] = build_ops(Grid)
D  = -np.transpose(G)
Avg     = comp_mean_matrix(Grid)

[Xc,Yc] = np.meshgrid(Grid.xc,Grid.yc)                 #building the (x,y) matrix
Xc_col  = np.reshape(np.transpose(Xc), (Grid.N,-1))    #building the single X vector
Yc_col  = np.reshape(np.transpose(Yc), (Grid.N,-1))    #building the single Y vector

s_wfunc    = lambda phi_w,phi_nw: phi_w  / (phi_w + phi_nw)
s_nwfunc   = lambda phi_w,phi_nw: phi_nw / (phi_w + phi_nw)
T_annual_func_sigmoid = lambda Tbot, Ttop, Yc_col, Y0: Tbot + (Ttop - Tbot)/Y0*(Yc_col)

#Initial conditions
phi_nw  = phi_nw_init*np.ones_like(Yc_col)
phi_nw[Yc_col>5] = 0.3
phi_w   = np.zeros_like(phi_nw) #No water
C       = rho_w * phi_w + (1 - phi_w - phi_nw) * rho_i
T_dummy = (Tm)*np.ones_like(Yc_col)
T_dummy[Yc_col>5] = Tm+T_firn
H       = enthalpyfromT(T_dummy,Tm,rho_i,rho_w,0,cp_i,cp_w,0,phi_w,phi_nw,L_fusion) 


phi_w,phi_i,T,dTdH = eval_phase_behaviorCwH(H,Tm,rho_i,rho_w,rho_nw,cp_i,cp_w,cp_nw,C,L_fusion)
s_w_init= s_wfunc(phi_w,phi_nw)
phi = (phi_w+ phi_nw)*np.ones((Grid.N,1)) #porosity in each cell
s_w = s_w_init.copy()
fs_theta = 0.0*np.ones((Grid.N,1))                     #RHS of heat equation

simulation_name = simulation_name+f'phi{phi_nw[0]}'+f'T{T_firn}'+f'npp{npp}'

#initializing arrays
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

##########

#boundary condition for saturation equation
BC.dof_dir   = dof_inj
BC.dof_f_dir = dof_f_inj
BC.dof_neu   = np.array([])
BC.dof_f_neu = np.array([])
BC.qb = np.array([])
BC.C_g    = C[dof_inj-1] + rho_w*C_L*np.ones((len(dof_inj),1))
[B,N,fn]  = build_bnd(BC, Grid, I)
# Enthalpy equation (total)

dof_fixedH = np.setdiff1d(Grid.dof_ymin,dof_inj)
dof_f_fixedH = np.setdiff1d(Grid.dof_f_ymin,dof_f_inj)

Param.H.dof_dir = np.concatenate([dof_inj, Grid.dof_ymax])
Param.H.dof_f_dir = np.concatenate([dof_f_inj,Grid.dof_f_ymax])
Param.H.g = np.vstack([rho_w*C_L*L_fusion*np.ones((Grid.Nx,1)),H[Grid.dof_ymax-1]])

Param.H.dof_neu = np.array([])
Param.H.dof_f_neu = np.array([])
Param.H.qb = np.array([])
[H_B,H_N,H_fn] = build_bnd(Param.H,Grid,I)

t    =[0.0]
time = 0
v = np.ones((Grid.Nf,1))

i = 0

while time<tmax:
    if time >npp*day2s:
        Param.H.g= np.vstack([np.zeros((Grid.Nx,1)),H[Grid.dof_ymax-1]])
        BC.C_g   = phi_i[dof_inj-1]*rho_i     #no water

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
        dt1   = CFL*Grid.dy/np.max(speed[speed>0]) #Calculating the time step from the filling of volume
    else:
        dt1   = 1e16 #Calculating the time step from the filling of volume

    res = D@flux_vert  #since the gradients are only zero and 1    
    res_vert = res.copy()
    ######
    #Taking out the domain to cut off single phase region
    dof_act  = Grid.dof[phi_w_old[:,0] / (phi[:,0]*(1-s_gr)) < sat_threshold]
    dof_inact= np.setdiff1d(Grid.dof,dof_act) #saturated cells
    if len(dof_act)< Grid.N: #when atleast one layer is present
        #############################################
        dof_f_saturated = find_faces(dof_inact,D,Grid)       
        #Step 1: Modify the gradient to import natural BC at the crater
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
        
    dt2   = np.abs((phi*sat_threshold - phi*s_gr - phi_w_old)/(res)) #Calculating the time step from the filling of volume
    dt2  =  CFL*np.min(dt2[dt2>1e-4*z0/fc])


    #Time step of diffusion
    dt3  = np.min(CFL*(Grid.dy**2)/(2*(phi_w*k_w + phi_i*k_i)/(rho_i*phi_i*cp_i+rho_w*phi_w*cp_w))) 
    if np.isnan(dt3): dt3=1e10
    dt = np.min([dt1,dt2,dt3])

    #if i<10: 
    #    dt = tmax/(Nt*10)
    if time+dt >= t_interest[np.max(np.argwhere(time+dt >= t_interest))] and time < t_interest[np.max(np.argwhere(time+dt >= t_interest))]:
        dt = t_interest[np.max(np.argwhere(time+dt >= t_interest))] - time   #To have the results at a specific time

    #Explicit Enthalpy update
    phi_w,phi_i,T,dTdH = eval_phase_behaviorCwH(H,Tm,rho_i,rho_w,0,cp_i,cp_w,0,C,L_fusion)

    #Enthalpy
    #calculating K_bar using K_bar = phi_w*k_w + phi_i*k_i
    #K_bar = phi_w*k_w + phi_i*k_i
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

    if np.isin(time,t_interest):
        t.append(time) 
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


#saving the tensors
np.savez(f'{simulation_name}_C{C_L}_{Grid.Nx}by{Grid.Ny}_t{tmax}.npz', t=t,q_w_new_sol=q_w_new_sol,H_sol=H_sol,T_sol=T_sol,s_w_sol=s_w_sol,phi_w_sol =phi_w_sol,phi_i_sol =phi_i_sol,phi=phi,Xc=Xc,Yc=Yc,Xc_col=Xc_col,Yc_col=Yc_col,Grid_Nx=Grid.Nx,Grid_Ny=Grid.Ny,Grid_xc=Grid.xc,Grid_yc=Grid.yc,Grid_xf=Grid.xf,Grid_yf=Grid.yf)


'''
######################################################################
#for loading data
######################################################################
data = np.load('new_Discussion_figure_-20Cphi[0.7]T-20npp4_C0.4_2by800_t86400.npz')
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


def f_C(H,C,m,n):
    fC = np.zeros_like(H)          #Region 1: ice + gas
    fC[H>0] = (1-C[H>0]+H[H>0])**m * (H[H>0]/(1-C[H>0]+H[H>0]))**n        #Region 2: ice + water
    fC[H>C] = C[H>C]**(n)#Region 3: water + gas

    fC[C==1] = H[C==1]**m            #Region 4: single component region
    fC[C>1]  = 0.0       #Region 4: outer region   
    return fC

def f_H(H,C,m,n):
    fH = np.zeros_like(H)         #Region 1: ice + gas
    fH[H>0] = (1-C[H>0]+H[H>0])**m * (H[H>0]/(1-C[H>0]+H[H>0]))**n  #Region 2: ice + water + gas
    fH[H>C] = H[H>C]*C[H>C]**(n-1)#Region 3: water + gas
    
    fH[C==1] = H[C==1]**m     #Region 4: single component region
    fH[C>1]  = 0.0       #Region 4: outer region             
    return fH


#Plotting
Grid.ymax = 10
zc=5
Kh = k0*k_w0*rho_w*grav/mu_w
tc = (zc/Kh)/day2s
s = (f_C(np.array([0.4]),np.array([0.7]),m,n)[0]-f_C(np.array([0]),np.array([0.3]),m,n)[0])/(0.7 - 0.3)#Shock speed

C_L,H_L,C_R,H_R =0.7,0.4,0.7,-0.088395

#Analytical solution
C_I1 = 1; C_I2 = 1
H_I1 = 1 - C_L + H_L 
H_I2 = 1 - C_R + H_R 

a = (1-C_L)/(1-C_R)
b =-(1-C_L)/(1-C_R) + (H_L/(1-C_L+H_L))**n - 1
c = 1 - ((1-C_L+H_L) / (1-C_R+H_R))**m * (H_L/(1-C_L+H_L))**n

Ratio = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)

q_s_dimless = (Ratio - 1)/(Ratio/(1-C_L+H_L)**m - 1/(1-C_R+H_R)**m)


s1 = (q_s_dimless - f_H(np.array([H_L]),np.array([C_L]),m,n)[0])/(H_I1 - H_L)#Backfilling shock speed      
s3 = (q_s_dimless-f_H(np.array([H_R]),np.array([C_R]),m,n)[0])/(H_I2 - H_R)#Shock speed

tday = t/day2s
T1_analy = np.linspace(0,1/s,1000)
T2_analy = np.linspace(1/s,tday[-1]/tc,1000)



#Plotting
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


fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharey=True, sharex=True,figsize=(9,14))
phi_w_sol_backup = phi_w_sol.copy()
tday = t/day2s
depth_array=np.kron(np.ones(len(tday)),np.transpose([Grid.yc]))
t_array=np.kron(tday,np.ones((Grid.Ny,1)))
phi_w_array = phi_w_sol_backup[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
phi_i_array = phi_i_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]

plt.subplot(3,1,3)
plot = [plt.contourf(t_array/tc, depth_array/zc, (1-phi_i_array),cmap="Greys_r",levels=100,antialiased=False,edgecolor='face',linewidth=0.4)]
mm = plt.cm.ScalarMappable(cmap=cm.Greys_r)
mm.set_array((1-phi_i_array))
plt.ylabel(r'z/5 m')
plt.xlim([tday[0]/tc,tday[-1]/tc])
plt.ylim([Grid.ymax/zc,Grid.ymin/zc])

clb = plt.colorbar(orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$\phi$', labelpad=0, y=1.0, rotation=0)

tick_locator = ticker.MaxNLocator(nbins=5)
clb.locator = tick_locator
clb.update_ticks()

plt.plot(T1_analy,s*T1_analy,'r--',lw=4,color=red)
plt.plot(T2_analy,1+s1*(T2_analy-1/s),'g--',lw=4,color=green)
plt.plot(T2_analy,1+s3*(T2_analy-1/s),'b--',lw=4,color=blue)
plt.xlabel(r't/2.53 hours')

plt.subplot(3,1,1)
plot = [plt.contourf(t_array/tc, depth_array/zc, phi_w_array/(1-phi_i_array),cmap="Blues",levels=100,antialiased=False,edgecolor='face',linewidth=0.4)]
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(np.linspace(0.0,1.0,1000))
plt.ylabel(r'z/5 m')
plt.xlim([tday[0]/tc,tday[-1]/tc])
plt.ylim([Grid.ymax/zc,Grid.ymin/zc])
clb = plt.colorbar(orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$s_w$', labelpad=0, y=1.0, rotation=0)

tick_locator = ticker.MaxNLocator(nbins=5)
clb.locator = tick_locator
clb.update_ticks()

plt.plot(T1_analy,s*T1_analy,'r--',lw=4,color=red,label='Wetting front')
plt.plot(T2_analy,1+s1*(T2_analy-1/s),'g--',lw=4,color=green,label='Rising perched water table')
plt.plot(T2_analy,1+s3*(T2_analy-1/s),'b--',lw=4,color=blue,label='Wetting front')
#plt.legend(frameon=False)

plt.subplot(3,1,2)
T_array = T_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
plot = [plt.contourf(t_array/tc, depth_array/zc, T_array-Tm,cmap="Reds",levels=100,antialiased=False,edgecolor='face',linewidth=0.4)]
mm = plt.cm.ScalarMappable(cmap=cm.Reds)
mm.set_array(np.linspace(-20,0,1000))
plt.ylabel(r'z/5 m')
plt.xlim([tday[0]/tc,tday[-1]/tc])
plt.ylim([Grid.ymax/zc,Grid.ymin/zc])
clb = plt.colorbar(orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'T[$^\circ$C]', labelpad=0, y=1.0, rotation=0)
plt.plot(T1_analy,s*T1_analy,'r--',color=red,lw=4)
plt.plot(T2_analy,1+s1*(T2_analy-1/s),'g--',color=green,lw=4)
plt.plot(T2_analy,1+s3*(T2_analy-1/s),'b--',color=blue,lw=4)
tick_locator = ticker.MaxNLocator(nbins=5)
clb.locator = tick_locator
clb.update_ticks()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(wspace=0.0, hspace=0.0)

plt.savefig(f'../Figures/combined.pdf',bbox_inches='tight', dpi = 600)
plt.savefig(f'../Figures/combined.png',bbox_inches='tight', dpi = 600)



#Making one huge subplot
light_gray = [0.85,0.85,0.85]
blue   = [ 30/255 ,144/255 , 255/255 ]


s_w_sol = np.concatenate((s_w_sol,s_w), axis=1) 
H_sol   = np.concatenate((H_sol,H), axis=1) 
T_sol   = np.concatenate((T_sol,T), axis=1) 
phi_w_sol    = np.concatenate((phi_w_sol,phi_w), axis=1) 
phi_i_sol   = np.concatenate((phi_i_sol,phi_i), axis=1) 
q_w_new_sol  = np.concatenate((q_w_new_sol,flux_vert), axis=1) 


'''
#New Contour plot video
print('Saving animation')
fps = 1# frame per sec
#frn = endstop # frame number of the animation
[N,frn] = np.shape(C_sol) # frame number of the animation from the saved file

def update_plot(frame_number, phi_i_sol, plot,t):  
    
    fig.clear() 
    plot[0] = plt.fill_betweenx(Grid.yc/5-1,1, facecolor='white',label=r'$\phi_{gas}$')
    plot[0] = plt.fill_betweenx(Grid.yc/5-1, phi_i_sol[0:Grid.Ny,frame_number]+ phi_w_sol[0:Grid.Ny,frame_number], facecolor=blue,label=r'$\phi_{water}$')
    plot[0] = plt.fill_betweenx(Grid.yc/5-1, phi_i_sol[0:Grid.Ny,frame_number], facecolor=light_gray,label=r'$\phi_{ice}$')
    
    #plt.axis('scaled')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    #plt.clim(250,270)
    plt.ylabel(r'Dimensionless depth $z/z_0$')
    plt.xlabel(r'Volume fraction $\phi$')
    plt.ylim([1,-1])
    plt.xlim([0,1])
    #plt.axis('scaled')
    #plt.clim(250,270)
    #plt.title(r"$t'$= %0.2f" % t[frame_number],loc = 'center', fontsize=18)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.legend(loc='lower left',borderaxespad=0.)


fig = plt.figure(figsize=(4,7.5) , dpi=100)

plot = [plt.fill_betweenx(Grid.yc/5-1,1, facecolor='white',label=r'$\phi_{gas}$')]
plot = [plt.fill_betweenx(Grid.yc/5-1, phi_i_sol[0:Grid.Ny,0]+ phi_w_sol[0:Grid.Ny,0], facecolor=blue,label=r'$\phi_{water}$')]
plot = [plt.fill_betweenx(Grid.yc/5-1, phi_i_sol[0:Grid.Ny,0], facecolor=light_gray,label=r'$\phi_{soil}$')]


plt.ylabel(r'Dimensionless depth $z/z_0$')
plt.xlabel(r'Volume fraction $\phi$')
plt.ylim([1,-1])
plt.xlim([0,1])
#plt.title(r"$\tau$= %0.2f" % t[0],loc = 'center', fontsize=18)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='lower left',borderaxespad=0.)

#ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(C_sol[:,::40], plot[::40],t[::40]), interval=1/fps)
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(phi_i_sol, plot[:],t[:]), interval=1/fps) #for now

ani.save(f"../Figures/fig2.mov", writer='ffmpeg', fps=30)

'''



'''

#New Contour plot video
print('Saving animation')
fps = 10000 # frame per sec
#frn = endstop # frame number of the animation
[N,frn] = np.shape(H_sol) # frame number of the animation from the saved file

def update_plot(frame_number, zarray,carray, plot,t):  
    
    zeta  = eta*t[frame_number]
    zeta1 = np.linspace(Grid.xmin,eta[0]*t[frame_number],1000)
    zeta2 = np.linspace(eta[-2]*t[frame_number],Grid.xmax,1000)

    H_analy1 = H_analy[0]*np.ones_like(zeta1)
    C_analy1 = C_analy[0]*np.ones_like(zeta1)
    H_analy2 = H_analy[-1]*np.ones_like(zeta2)
    C_analy2 = C_analy[-1]*np.ones_like(zeta2)
     
    zeta_comb  = np.concatenate([zeta1,zeta,zeta2])
    H_analy_comb  = np.concatenate([H_analy1,H_analy,H_analy2])
    C_analy_comb  = np.concatenate([C_analy1,C_analy,C_analy2])
    
    fig.clear() 
    plot[0] = plt.fill_betweenx(zeta_comb, phi_i(H_analy_comb,C_analy_comb)+phi_w(H_analy_comb,C_analy_comb)+phi_g(H_analy_comb,C_analy_comb), facecolor='white',label=r'$\phi_{gas}$')
    plot[0] = plt.fill_betweenx(zeta_comb, phi_w(H_analy_comb,C_analy_comb)+phi_i(H_analy_comb,C_analy_comb), facecolor=blue,label=r'$\phi_{water}$')
    plot[0] = plt.fill_betweenx(zeta_comb, phi_i(H_analy_comb,C_analy_comb), facecolor=light_gray,label=r'$\phi_{ice}$')

    
    #plt.axis('scaled')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    #plt.clim(250,270)
    plt.xlabel(r'$\zeta$')
    #plt.ylabel(r'$\mathcal{H}$')
    plt.ylim([Grid.xmax,Grid.xmin])
    plt.xlim([0,1])
    #plt.axis('scaled')
    #plt.clim(250,270)
    plt.title(r"$\tau$= %0.2f" % t[frame_number],loc = 'center', fontsize=18)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.legend(loc='lower right',borderaxespad=0.)


zeta  = eta*t[0]
zeta1 = np.linspace(Grid.xmin,eta[1]*t[0],1000)
zeta2 = np.linspace(eta[-1]*t[0],Grid.xmax,1000)

H_analy1 = H_analy[0]*np.ones_like(zeta1)
C_analy1 = C_analy[0]*np.ones_like(zeta1)
H_analy2 = H_analy[-1]*np.ones_like(zeta2)
C_analy2 = C_analy[-1]*np.ones_like(zeta2)
 
zeta_comb  = np.concatenate([zeta1,zeta,zeta2])
H_analy_comb  = np.concatenate([H_analy1,H_analy,H_analy2])
C_analy_comb  = np.concatenate([C_analy1,C_analy,C_analy2])

fig = plt.figure(figsize=(7,10) , dpi=100)
plot = [plt.fill_betweenx(zeta_comb, phi_i(H_analy_comb,C_analy_comb)+phi_w(H_analy_comb,C_analy_comb)+phi_g(H_analy_comb,C_analy_comb), facecolor='white',label=r'$\phi_{gas}$')]
plot = [plt.fill_betweenx(zeta_comb, phi_w(H_analy_comb,C_analy_comb)+phi_i(H_analy_comb,C_analy_comb), facecolor=blue,label=r'$\phi_{water}$')]
plot = [plt.fill_betweenx(zeta_comb, phi_i(H_analy_comb,C_analy_comb), facecolor=light_gray,label=r'$\phi_{ice}$')]


#plot = plt.plot(Grid.xc,T(H_sol[:,0],C_sol[:,0], Ste, Cpr),'r--',label=r'$\mathcal{T}$')
#plt.ylabel(r'$\mathcal{H}$')
plt.xlabel(r'$\zeta$')
plt.ylim([Grid.xmax,Grid.xmin])
#plt.ylim([np.min([np.min(H_sol),np.min(C_sol)])-0.05,np.max([np.max(H_sol),np.max(C_sol)])+0.05])
plt.xlim([0,1])
plt.title(r"$\tau$= %0.2f" % t[i],loc = 'center', fontsize=18)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='lower right',borderaxespad=0.)

ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(H_sol[:,::40],C_sol[:,::40], plot[::40],t[::40]), interval=1/fps)

ani.save(f"../Figures/{simulation_name}_video_Nx{Grid.Nx}_HL{H_L}HR{H_R}__tf{t[frn-1]}.mov", writer='ffmpeg', fps=30)

'''
