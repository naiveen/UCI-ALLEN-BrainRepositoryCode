import numpy as np
import scipy
import gc



def laplacianA1D(n, boundaryIndices):

    # Create X,Y,Z arrays that represent the x,y,z indices of each voxel
        
    X= np.arange(n)
    # Calculate each voxel's index when flattened
    ids_0  = X
    data  = np.ones(len(ids_0)) *2
    boundaryIndices = boundaryIndices.astype(int)

    """
    rids: row indices value
    cids: column indices values
    data: Diagonal entries of sparse matrix A. A[rid, rid] = 6 at all non boundary locations. 
          A[rid, rid] = 1 at Dirchlet boundary. A[rid, rid] = number of valid neighbours at volume boundary

    """
    print("Building data for Laplacian Sparse Matrix A")

    # Calculate the voxel index of (x-1,y,z) for each (x,y,z)
    cids_x1  = (X-1)
    invalid_cx1 = np.concatenate([np.where(X==0)[0], boundaryIndices])  # invalid column indices and coorespondences indices.
    rids_x1  = np.delete(ids_0, invalid_cx1) # remove invalid row indices that correspond to column
    cids_x1  = np.delete(cids_x1, invalid_cx1) # remove invalid column indices
    data[invalid_cx1] -=1   # decrease the value of A[rid] by 1 at invalid indices

    # Calculate the voxel index of (x+1,y,z) for each (x,y,z)
    cids_x2  = (X+1)
    invalid_cx2 = np.concatenate([np.where(X==n-1 )[0], boundaryIndices])
    rids_x2  = np.delete(ids_0, invalid_cx2)
    cids_x2  = np.delete(cids_x2, invalid_cx2)
    data[invalid_cx2] -=1


    # Diagonal entries corresponding to dirichlet boundaries should be 1
    data[boundaryIndices] +=1

    rowx = np.hstack([ids_0, rids_x1, rids_x2])
    rowy = np.hstack([ids_0, cids_x1, cids_x2])
    rowv = np.hstack([data , -1*np.ones(rowx.shape[0] - ids_0.shape[0]) ] )

    print("Creating Laplacian Sparse Matrix A")
    A= scipy.sparse.csr_matrix((rowv,(rowx,rowy)), shape =(X.shape[0],X.shape[0]))

    del rowx, rowy, rowv, X, data
    gc.collect()

    return A

def laplacianA2D(shape, boundaryIndices):
    """
    Creates the matrix A that coorespond to the linear system of ewuations used to perform laplacian Interpolation on 2D images

    Parameters
    --------------
    shape : tuple 2D shape
    boundaryIndices : indices of boundary points when flattened. If (x,y) is pixel x*shape[1]+y will be flattened index. 
    """

    #Create X,Y arrays that represent the x,y indices of each pixe
    
    k = len(shape)
    
    X,Y= np.meshgrid(range(shape[0]), range(shape[1]), indexing='ij')

    X=X.flatten().astype(int)
    Y=Y.flatten().astype(int)

    # Calculate each voxel's index when flattened
    ids_0  = X* shape[1] + Y
    data  = np.ones(len(ids_0)) *2*k
    boundaryIndices = boundaryIndices.astype(int)
    """
    rids: row indices value
    cids: column indices values
    data: Diagonal entries of sparse matrix A. A[rid, rid] = 6 at all non boundary locations. 
          A[rid, rid] = 1 at Dirchlet boundary. A[rid, rid] = number of valid neighbours at volume boundary
    """
    #print("Building data for Laplacian Sparse Matrix A")

    # Calculate the voxel index of (x-1,y) for each (x,y)
    cids_x1  = (X-1)* shape[1] + Y
    invalid_cx1 = np.concatenate([np.where(X==0)[0], boundaryIndices])  # invalid column indices and coorespondences indices.
    rids_x1  = np.delete(ids_0, invalid_cx1) # remove invalid row indices that correspond to column
    cids_x1  = np.delete(cids_x1, invalid_cx1) # remove invalid column indices
    data[invalid_cx1] -=1   # decrease the value of A[rid, rid] by 1 at invalid indices

    # Calculate the voxel index of (x+1,y) for each (x,y)
    cids_x2  = (X+1)* shape[1] + Y
    invalid_cx2 = np.concatenate([np.where(X==shape[0] -1 )[0], boundaryIndices])
    rids_x2  = np.delete(ids_0, invalid_cx2)
    cids_x2  = np.delete(cids_x2, invalid_cx2)
    data[invalid_cx2] -=1

    # Calculate the voxel index of (x,y-1) for each (x,y)
    cids_y1  = X* shape[1] + Y-1
    invalid_cy1 = np.concatenate([np.where(Y==0)[0], boundaryIndices])
    rids_y1  = np.delete(ids_0, invalid_cy1)
    cids_y1  = np.delete(cids_y1, invalid_cy1)
    data[invalid_cy1] -=1

    # Calculate the voxel index of (x,y+1) for each (x,y)
    cids_y2  = X* shape[1] + Y+1
    invalid_cy2 = np.concatenate([np.where(Y==shape[1]-1)[0], boundaryIndices])
    rids_y2  = np.delete(ids_0, invalid_cy2)
    cids_y2  = np.delete(cids_y2, invalid_cy2)
    data[invalid_cy2] -=1

    # Diagonal entries corresponding to dirichlet boundaries should be 1
    data[boundaryIndices] +=1

    rowx = np.hstack([ids_0, rids_x1, rids_x2,rids_y1,rids_y2])
    rowy = np.hstack([ids_0, cids_x1, cids_x2,cids_y1,cids_y2])
    rowv = np.hstack([data , -1*np.ones(rowx.shape[0] - ids_0.shape[0]) ] )

    #print("Creating Laplacian Sparse Matrix A")
    A= scipy.sparse.csr_matrix((rowv,(rowx,rowy)), shape =(X.shape[0],X.shape[0]))

    del rowx, rowy, rowv, X, Y, data
    gc.collect()

    return A

def laplacianA3D(shape, boundaryIndices):
    """
    Creates the matrix A that correspond to the linear system of ewuations used to perform laplacian Interpolation on 3D volume images

    Parameters
    --------------
    shape : tuple 3D shape
    boundaryIndices : indices of boundary points when flattened. If (x,y) is pixel x*shape[1]*shape[2]+y*shape[2]+z will be flattened index. 
    """
    k = len(shape)
    # Create X,Y,Z arrays that represent the x,y,z indices of each voxel
    X,Y,Z = np.meshgrid(range(shape[0]), range(shape[1]), range(shape[2]), indexing='ij')

    X=X.flatten().astype(int)
    Y=Y.flatten().astype(int)
    Z=Z.flatten().astype(int)

    # Calculate each voxel's index when flattened
    ids_0  = X* shape[1]*shape[2] + Y*shape[2]+ Z
    data  = np.ones(len(ids_0)) * 2*k
    boundaryIndices = boundaryIndices.astype(int)

    """
    rids: row indices value
    cids: column indices values
    data: Diagonal entries of sparse matrix A. A[rid, rid] = 6 at all non boundary locations. 
          A[rid, rid] = 1 at Dirchlet boundary. A[rid, rid] = number of valid neighbours at volume boundary


    """
    print("Building data for Laplacian Sparse Matrix A")

    # Calculate the voxel index of (x-1,y,z) for each (x,y,z)
    cids_x1  = (X-1)* shape[1]*shape[2] + Y*shape[2]+ Z
    invalid_cx1 = np.concatenate([np.where(X==0)[0], boundaryIndices])  # invalid column indices and coorespondences indices. X==0 is invalid because X-1 will be negative
    rids_x1  = np.delete(ids_0, invalid_cx1) # remove invalid row indices that correspond to column
    cids_x1  = np.delete(cids_x1, invalid_cx1) # remove invalid column indices
    data[invalid_cx1] -=1   # decrease the value of A[rid, rid] by 1 at invalid indices

    # Calculate the voxel index of (x+1,y,z) for each (x,y,z)
    cids_x2  = (X+1)* shape[1]*shape[2] + Y*shape[2]+ Z
    invalid_cx2 = np.concatenate([np.where(X==shape[0] -1 )[0], boundaryIndices])
    rids_x2  = np.delete(ids_0, invalid_cx2)
    cids_x2  = np.delete(cids_x2, invalid_cx2)
    data[invalid_cx2] -=1

    # Calculate the voxel index of (x,y-1,z) for each (x,y,z)
    cids_y1  = X* shape[1]*shape[2] + (Y-1)*shape[2]+ Z
    invalid_cy1 = np.concatenate([np.where(Y==0)[0], boundaryIndices])
    rids_y1  = np.delete(ids_0, invalid_cy1)
    cids_y1  = np.delete(cids_y1, invalid_cy1)
    data[invalid_cy1] -=1

    # Calculate the voxel index of (x,y+1,z) for each (x,y,z)
    cids_y2  = X* shape[1]*shape[2] + (Y+1)*shape[2]+ Z
    invalid_cy2 = np.concatenate([np.where(Y==shape[1]-1)[0], boundaryIndices])
    rids_y2  = np.delete(ids_0, invalid_cy2)
    cids_y2  = np.delete(cids_y2, invalid_cy2)
    data[invalid_cy2] -=1

    # Calculate the voxel index of (x,y,z-1) for each (x,y,z)
    cids_z1  = X* shape[1]*shape[2] + Y*shape[2]+ Z-1
    invalid_cz1 = np.concatenate([np.where(Z==0)[0], boundaryIndices])
    rids_z1  = np.delete(ids_0, invalid_cz1)
    cids_z1  = np.delete(cids_z1, invalid_cz1)
    data[invalid_cz1] -=1

    # Calculate the voxel index of (x,y,z+1) for each (x,y,z)
    cids_z2  = X* shape[1]*shape[2] +Y*shape[2]+ Z+1
    invalid_cz2 = np.concatenate([np.where(Z==shape[2] -1)[0], boundaryIndices])   
    rids_z2  = np.delete(ids_0, invalid_cz2)  
    cids_z2  = np.delete(cids_z2, invalid_cz2) 
    data[invalid_cz2] -=1 

    # Diagonal entries corresponding to dirichlet boundaries should be 1
    data[boundaryIndices] +=1

    rowx = np.hstack([ids_0, rids_x1, rids_x2,rids_y1,rids_y2, rids_z1, rids_z2])
    rowy = np.hstack([ids_0, cids_x1, cids_x2,cids_y1,cids_y2, cids_z1, cids_z2])
    rowv = np.hstack([data , -1*np.ones(rowx.shape[0] - ids_0.shape[0]) ] )

    print("Creating Laplacian Sparse Matrix A")
    A= scipy.sparse.csr_matrix((rowv,(rowx,rowy)), shape =(X.shape[0],X.shape[0]))

    del rowx, rowy, rowv, X, Y,Z, data
    gc.collect()

    return A