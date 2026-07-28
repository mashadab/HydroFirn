import scipy.sparse as sp
import numpy as np

def build_ops(Grid):
    # author: Mohammad Afzal Shadab
    # date: 1/27/2020
    # description:
    # This function computes the discrete divergence and gradient matrices on a
    # regular staggered grid using central difference approximations. The
    # discrete gradient assumes homogeneous boundary conditions.
    # Input:
    # Grid = structure containing all pertinent information about the grid.
    # Output:
    # D = discrete divergence matrix
    # G = discrete gradient matrix
    # I = identity matrix

    Nx = Grid.Nx
    Ny = Grid.Ny
    N  = Grid.N

    # Two dimensional divergence    
    #     Readable implementation
    #     # 2D divergence matrices
    
    if (Nx>1) and (Ny>1): #2D case
        #One diamentional divergence
        Dy = sp.spdiags(([-np.array(np.ones((Ny+1),'float64')) , np.array(np.ones((Ny+1),'float64'))])/np.asarray(Grid.dy),np.array([0,1]),Ny,Ny+1).toarray() # Dy^1
        
        #Two dimensional divergence
        Dy = sp.kron(sp.eye(Nx), Dy) #y component Dy^2
        
        e  = np.array(np.ones(Ny*(Nx+1),'float64'))
        Dx = sp.spdiags(([-e , e])/np.asarray(Grid.dx),np.array([0,Ny]),N,(Nx+1)*Ny) # 2D div-matrix in x-dir

        D  = sp.hstack([Dx , Dy])        
        dof_f_bnd = np.concatenate([Grid.dof_f_xmin-1, Grid.dof_f_xmax-1, Grid.dof_f_ymin-1, Grid.dof_f_ymax-1] )       # boundary faces
        dof_f_bnd = np.transpose(dof_f_bnd)
        
    elif (Nx > 1) and (Ny == 1): #one dimensional in x direction
        D = sp.spdiags(([-np.array(np.ones((Nx+1),'float64')),np.array(np.ones((Nx+1),'float64'))])/np.asarray(Grid.dx),np.array([0,1]),Nx,Nx+1) # 1D div-matrix in x-dir
        dof_f_bnd = [Grid.dof_f_xmin-1, Grid.dof_f_xmax-1] # boundary faces
        dof_f_bnd = np.transpose(dof_f_bnd)  

    elif (Nx == 1) and (Ny > 1): #one dimensional in y direction
        D = sp.spdiags(([-np.array(np.ones((Ny+1),'float64')),np.array(np.ones((Ny+1),'float64'))])/np.asarray(Grid.dy),np.array([0,1]),Ny,Ny+1) # 1D div-matrix in y-dir
        dof_f_bnd = [Grid.dof_f_ymin-1, Grid.dof_f_ymax-1] # boundary faces
        dof_f_bnd = np.transpose(dof_f_bnd)  

    # Gradient
    # Note this is only true in cartesian coordinates!
    # For more general coordinate systems it is worth
    # assembling G and D seperately.
    
    D =  sp.csr_matrix(D)
    G = -sp.csr_matrix.transpose(D)
    G = zero_rows(G,dof_f_bnd)

    #Identity
    I = (sp.eye(Grid.N)).tocsr()

    return D,G,I;


def build_ops_unstructured(Grid):
    # author: Mohammad Afzal Shadab
    # date: 04/03/2026
    # description:
    # This function computes the discrete divergence and gradient matrices on a
    # regular staggered grid using central difference approximations. The
    # discrete gradient assumes homogeneous boundary conditions.
    # Input:
    # Grid = structure containing all pertinent information about the grid.
    # Output:
    # D = discrete divergence matrix
    # G = discrete gradient matrix
    # I = identity matrix

    Nx = Grid.Nx
    Ny = Grid.Ny
    N  = Grid.N

    # Two dimensional divergence    
    #     Readable implementation
    #     # 2D divergence matrices
    
    if (Nx>1) and (Ny>1): #2D case
        #One diamentional divergence
        Dy = sp.spdiags(([-np.array(np.ones((Ny+1),'float64')) , np.array(np.ones((Ny+1),'float64'))]),np.array([0,1]),Ny,Ny+1).tocsr()#.toarray() # Dy^1
        print(np.shape(Dy),np.shape(sp.kron(Grid.dy_scaled,np.ones((1,Ny+1)))))
        Dy = Dy/sp.kron(Grid.dy_scaled,np.ones((1,Ny+1)))
        
        #Two dimensional divergence
        Dy = sp.kron(np.eye(Nx), Dy) #y component Dy^2
        
        
        #One diamentional divergence - x
        Dx = sp.spdiags(([-np.array(np.ones((Nx+1),'float64')) , np.array(np.ones((Nx+1),'float64'))]),np.array([0,1]),Nx,Nx+1).tocsr()#.toarray() # Dx^1
    
        Dx = Dx/sp.kron(Grid.dx_scaled,np.ones((1,Nx+1)))        
        
        #Two dimensional divergence
        Dx = sp.kron(Dx,np.eye(Ny)) #y component Dx^2        
        
        #e  = np.array(np.ones(Ny*(Nx+1),'float64'))        
        #Dx = sp.spdiags(([-e , e]),np.array([0,Ny]),N,(Nx+1)*Ny).tocsr()#.toarray() # 2D div-matrix in x-dir

        #D  = np.concatenate((Dx , Dy), axis=1)
        D  = sp.hstack([Dx , Dy]).tocsr()    
        dof_f_bnd = np.concatenate([Grid.dof_f_xmin-1, Grid.dof_f_xmax-1, Grid.dof_f_ymin-1, Grid.dof_f_ymax-1])       # boundary faces
        dof_f_bnd = np.transpose(dof_f_bnd)
        
    elif (Nx > 1) and (Ny == 1): #one dimensional in x direction
        D  = sp.spdiags(([-np.array(np.ones((Nx+1),'float64')) , np.array(np.ones((Nx+1),'float64'))]),np.array([0,1]),Nx,Nx+1).tocsr()#.toarray() # Dx^1
        D  = D/sp.kron(Grid.dx_scaled,np.ones((Nx+1,1)))        
        dof_f_bnd = [Grid.dof_f_xmin-1, Grid.dof_f_xmax-1] # boundary faces
        dof_f_bnd = np.transpose(dof_f_bnd)  

    elif (Nx == 1) and (Ny > 1): #one dimensional in y direction
        D  = sp.spdiags(([-np.array(np.ones((Ny+1),'float64')) , np.array(np.ones((Ny+1),'float64'))]),np.array([0,1]),Ny,Ny+1).tocsr()#.toarray() # Dy^1
        D  = D/sp.kron(Grid.dy_scaled,np.ones((Ny+1,1)))
        
        dof_f_bnd = [Grid.dof_f_ymin-1, Grid.dof_f_ymax-1] # boundary faces
        dof_f_bnd = np.transpose(dof_f_bnd)  

    # Gradient
    # Note this is only true in cartesian coordinates!
    # For more general coordinate systems it is worth
    # assembling G and D seperately.
    #print(D)
    
    if (Nx > 1) and (Ny > 1): # 2D case
            # One dimensional gradient in y
            Gy = sp.spdiags(
                ([-np.array(np.ones((Ny),'float64')),
                   np.array(np.ones((Ny),'float64'))]),
                np.array([0,1]), Ny-1, Ny
            ).tocsr()
            Gy = Gy / np.kron(Grid.dyf_scaled[1:Ny], np.ones((1,Ny)))
            Gy = sp.vstack([np.zeros((1,Ny)), Gy, np.zeros((1,Ny))]).tocsr()

            # Two dimensional gradient
            Gy = sp.kron(np.eye(Nx), Gy).tocsr()
    
            # One dimensional gradient in x
            Gx = sp.spdiags(
                ([-np.array(np.ones((Nx),'float64')),
                   np.array(np.ones((Nx),'float64'))]),
                np.array([0,1]), Nx-1, Nx
            ).tocsr()
            Gx = Gx / np.kron(Grid.dxf_scaled[1:Nx], np.ones((1,Nx)))
            Gx = sp.vstack([np.zeros((1,Nx)), Gx, np.zeros((1,Nx))]).tocsr()
    
            # Two dimensional gradient
            Gx = sp.kron(Gx, np.eye(Ny)).tocsr()
    
            G  = sp.vstack([Gx, Gy]).tocsr()
    
    elif (Nx > 1) and (Ny == 1): # 1D in x
        G = sp.spdiags(
            ([-np.array(np.ones((Nx),'float64')),
               np.array(np.ones((Nx),'float64'))]),
            np.array([0,1]), Nx-1, Nx
        ).tocsr()
        G = G / np.kron(Grid.dxf_scaled[1:Nx], np.ones((1,Nx)))
        G = sp.vstack([np.zeros((1,Nx)), G, np.zeros((1,Nx))]).tocsr()

    elif (Nx == 1) and (Ny > 1): # 1D in y
        G = sp.spdiags(
            ([-np.array(np.ones((Ny),'float64')),
               np.array(np.ones((Ny),'float64'))]),
            np.array([0,1]), Ny-1, Ny
        ).tocsr()
        G = G / np.kron(Grid.dyf_scaled[1:Ny], np.ones((1,Ny)))
        G = sp.vstack([np.zeros((1,Ny)), G, np.zeros((1,Ny))]).tocsr()

    else: # Nx == 1 and Ny == 1
        G = sp.csr_matrix((2,1))

    
    G =  zero_rows(G,dof_f_bnd)

    #Identity
    I = (sp.eye(Grid.N)).tocsr()
    
    #Curl matrix 
    C = []
    
    #Algebraic mean  
    if Grid.Nx>1 and Grid.Ny==1:
        #Averaging in x-direction considering the zero-flux at boundary
        Avg_x1 = sp.spdiags(([np.array(0.5*np.ones((Grid.Nx),'float64')),0.5*np.array(np.ones((Grid.Nx),'float64'))]),np.array([0,1]),Grid.Nx-1,Grid.Nx)
        Avg_x1 = sp.vstack([np.zeros((1,Grid.Nx)),Avg_x1, np.zeros((1,Grid.Nx))])         
        M      = Avg_x1.copy()
        
    elif Grid.Nx==1 and Grid.Ny>1:
        Avg_y1 = sp.spdiags(([np.array(0.5*np.ones((Grid.Ny),'float64')),0.5*np.array(np.ones((Grid.Ny),'float64'))]),np.array([0,1]),Grid.Ny-1,Grid.Ny)
        Avg_y1 = sp.vstack([np.zeros((1,Grid.Ny)),Avg_y1,np.zeros((1,Grid.Ny))]) 
        M      = Avg_y1.copy()     
    
    elif Grid.Nx>1 and Grid.Ny>1:
        #Averaging in y-direction considering the zero-flux at boundary
        Avg_y1 = sp.spdiags(([np.array(0.5*np.ones((Grid.Ny),'float64')),0.5*np.array(np.ones((Grid.Ny),'float64'))]),np.array([0,1]),Grid.Ny-1,Grid.Ny)
        Avg_y1 = sp.vstack([np.zeros((1,Grid.Ny)),Avg_y1,np.zeros((1,Grid.Ny))]) 
        Avg_y2 = sp.kron(sp.eye(Grid.Nx),Avg_y1)
    
        #Averaging in x-direction considering the zero-flux at boundary
        Avg_x1 = sp.spdiags(([np.array(0.5*np.ones((Grid.Nx),'float64')),0.5*np.array(np.ones((Grid.Nx),'float64'))]),np.array([0,1]),Grid.Nx-1,Grid.Nx)
        Avg_x1 = sp.vstack([np.zeros((1,Grid.Nx)),Avg_x1, np.zeros((1,Grid.Nx))]) 
        Avg_x2 = sp.kron(Avg_x1,sp.eye(Grid.Ny))         
        M      = sp.vstack([Avg_x2, Avg_y2])
    
    return D,G,C,I,M;



def zero_rows(M, rows_to_zero):

    ixs = np.ones(M.shape[0], int)
    ixs[rows_to_zero] = 0
    D = sp.diags(ixs)
    res = D * M
    return res