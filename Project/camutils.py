import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def makerotation(rx,ry,rz):
    """
    Generate a rotation matrix    

    Parameters
    ----------
    rx,ry,rz : floats
        Amount to rotate around x, y and z axes in degrees

    Returns
    -------
    R : 2D numpy.array (dtype=float)
        Rotation matrix of shape (3,3)
    """
    rx = np.pi*rx/180.0
    ry = np.pi*ry/180.0
    rz = np.pi*rz/180.0

    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,-np.sin(ry)],[0,1,0],[np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    R = (Rz @ Ry @ Rx)
    
    return R 

class Camera:
    """
    A simple data structure describing camera parameters 
    
    The parameters describing the camera
    cam.f : float   --- camera focal length (in units of pixels)
    cam.c : 2x1 vector  --- offset of principle point
    cam.R : 3x3 matrix --- camera rotation
    cam.t : 3x1 vector --- camera translation 

    
    """    
    def __init__(self,f,c,R,t):
        self.f = f
        self.c = c
        self.R = R
        self.t = t

    def __str__(self):
        return f'Camera : \n f={self.f} \n c={self.c.T} \n R={self.R} \n t = {self.t.T}'
    
    def project(self,pts3):
        """
        Project the given 3D points in world coordinates into the specified camera    

        Parameters
        ----------
        pts3 : 2D numpy.array (dtype=float)
            Coordinates of N points stored in a array of shape (3,N)

        Returns
        -------
        pts2 : 2D numpy.array (dtype=float)
            Image coordinates of N points stored in an array of shape (2,N)

        """
        assert(pts3.shape[0]==3)

        # get point location relative to camera
        pcam = self.R.transpose() @ (pts3 - self.t)
         
        # project
        p = self.f * (pcam / pcam[2,:])
        
        # offset principal point
        pts2 = p[0:2,:] + self.c
        
        assert(pts2.shape[1]==pts3.shape[1])
        assert(pts2.shape[0]==2)
    
        return pts2
 
    def update_extrinsics(self,params):
        """
        Given a vector of extrinsic parameters, update the camera
        to use the provided parameters.
  
        Parameters
        ----------
        params : 1D numpy.array (dtype=float)
            Camera parameters we are optimizing over stored in a vector
            params[0:2] are the rotation angles, params[2:5] are the translation

        """
        self.R = makerotation(params[0],params[1],params[2])
        self.t = np.array([[params[3]],[params[4]],[params[5]]])


def triangulate(pts2L,camL,pts2R,camR):
    """
    Triangulate the set of points seen at location pts2L / pts2R in the
    corresponding pair of cameras. Return the 3D coordinates relative
    to the global coordinate system


    Parameters
    ----------
    pts2L : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N) seen from camL camera

    pts2R : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N) seen from camR camera

    camL : Camera
        The first "left" camera view

    camR : Camera
        The second "right" camera view

    Returns
    -------
    pts3 : 2D numpy.array (dtype=float)
        (3,N) array containing 3D coordinates of the points in global coordinates

    """

    npts = pts2L.shape[1]

    qL = (pts2L - camL.c) / camL.f
    qL = np.vstack((qL,np.ones((1,npts))))

    qR = (pts2R - camR.c) / camR.f
    qR = np.vstack((qR,np.ones((1,npts))))
    
    R = camL.R.T @ camR.R
    t = camL.R.T @ (camR.t-camL.t)

    xL = np.zeros((3,npts))
    xR = np.zeros((3,npts))

    for i in range(npts):
        A = np.vstack((qL[:,i],-R @ qR[:,i])).T
        z,_,_,_ = np.linalg.lstsq(A,t,rcond=None)
        xL[:,i] = z[0]*qL[:,i]
        xR[:,i] = z[1]*qR[:,i]
 
    pts3L = camL.R @ xL + camL.t
    pts3R = camR.R @ xR + camR.t
    pts3 = 0.5*(pts3L+pts3R)

    return pts3


def residuals(pts3,pts2,cam,params):
    """
    Compute the difference between the projection of 3D points by the camera
    with the given parameters and the observed 2D locations

    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (3,N)

    pts2 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N)

    params : 1D numpy.array (dtype=float)
        Camera parameters we are optimizing over stored in a vector

    Returns
    -------
    residual : 1D numpy.array (dtype=float)
        Vector of residual 2D projection errors of size 2*N
        
    """

    cam.update_extrinsics(params)
    residual = pts2 - cam.project(pts3)
    
    return residual.flatten()

def calibratePose(pts3,pts2,cam_init,params_init):
    """
    Calibrate the provided camera by updating R,t so that pts3 projects
    as close as possible to pts2

    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (3,N)

    pts2 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N)

    cam : Camera
        Initial estimate of camera

    Returns
    -------
    cam_opt : Camera
        Refined estimate of camera with updated R,t parameters
        
    """

    # define our error function
    efun = lambda params: residuals(pts3,pts2,cam_init,params)        
    popt,_ = scipy.optimize.leastsq(efun,params_init)
    cam_init.update_extrinsics(popt)

    return cam_init



def decode(imprefix_color,imprefix,start,threshold_color,threshold):
    """
        Decode 10bit gray code pattern with the given difference
        threshold.  We assume the images come in consective pairs
        with filenames of the form <prefix><start>.png - <prefix><start+20>.png
        (e.g. a start offset of 20 would yield image20.png, image01.png... image39.png)
        
        Parameters
        ----------
        imprefix : str
        prefix of where to find the images (assumed to be .png)
        
        start : int
        image offset.
        
        threshold : float
        
        Returns
        -------
        code : 2D numpy.array (dtype=float)
        
        mask : 2D numpy.array (dtype=float)
        
        
        """
    nbits = 10
    
    imgs = list()
    imgs_inv = list()
    print('loading',end='')
    for i in range(start,start+2*nbits,2):
        fname0 = '%s%2.2d.png' % (imprefix,i)
        fname1 = '%s%2.2d.png' % (imprefix,i+1)
        print('(',i,i+1,')',end='')
        img = plt.imread(fname0)
        img_inv = plt.imread(fname1)
        if (img.dtype == np.uint8):
            img = img.astype(float) / 256
            img_inv = img_inv.astype(float) / 256
        if (len(img.shape)>2):
            img = np.mean(img,axis=2)
            img_inv = np.mean(img_inv,axis=2)
        imgs.append(img)
        imgs_inv.append(img_inv)
    
    (h,w) = imgs[0].shape
    print('\n')
    
    gcd = np.zeros((h,w,nbits))
    mask = np.ones((h,w))
    for i in range(nbits):
        gcd[:,:,i] = imgs[i]>imgs_inv[i]
        mask = mask * (np.abs(imgs[i]-imgs_inv[i])>threshold)
    
    bcd = np.zeros((h,w,nbits))
    bcd[:,:,0] = gcd[:,:,0]
    for i in range(1,nbits):
        bcd[:,:,i] = np.logical_xor(bcd[:,:,i-1],gcd[:,:,i])
    
    code = np.zeros((h,w))
    for i in range(nbits):
        code = code + np.power(2,(nbits-i-1))*bcd[:,:,i]
    
    #Note:!we need to make use of the color instead of convering to grayscale...
    #...since the object and the background could have the same gray level/brightness but be different colors!
    imc1 = plt.imread(imprefix_color +"%02d" % (0)+'.png')
    imc2= plt.imread(imprefix_color +"%02d" % (1)+'.png')
    color_mask = np.ones((h,w))
    color_mask = color_mask*((np.sum(np.square(imc1-imc2), axis=-1))>threshold_color)

    return code,mask,color_mask




def reconstruct(imprefixL1,imprefixL2,imprefixR1,imprefixR2,threshold1,threshold2,camL,camR):
    """
    Simple reconstruction based on triangulating matched pairs of points
    between to view which have been encoded with a 20bit gray code.

    Parameters
    ----------
    imprefix : str
      prefix for where the images are stored

    threshold : float
      decodability threshold

    camL,camR : Camera
      camera parameters

    Returns
    -------
    pts2L,pts2R : 2D numpy.array (dtype=float)

    pts3 : 2D numpy.array (dtype=float)

    """

    CLh,maskLh,cmaskL = decode(imprefixL1,imprefixL2,0,threshold1,threshold2)
    CLv,maskLv,_ = decode(imprefixL1,imprefixL2,20,threshold1,threshold2)
    CRh,maskRh,cmaskR = decode(imprefixR1,imprefixR2,0,threshold1,threshold2)
    CRv,maskRv,_ = decode(imprefixR1,imprefixR2,20,threshold1,threshold2)

    CL = CLh + 1024*CLv
    maskL = maskLh*maskLv*cmaskL
    CR = CRh + 1024*CRv
    maskR = maskRh*maskRv*cmaskR

    h = CR.shape[0]
    w = CR.shape[1]

    subR = np.nonzero(maskR.flatten())
    subL = np.nonzero(maskL.flatten())

    CRgood = CR.flatten()[subR]
    CLgood = CL.flatten()[subL]

    _,submatchR,submatchL = np.intersect1d(CRgood,CLgood,return_indices=True)

    matchR = subR[0][submatchR]
    matchL = subL[0][submatchL]

    xx,yy = np.meshgrid(range(w),range(h))
    xx = np.reshape(xx,(-1,1))
    yy = np.reshape(yy,(-1,1))

    pts2R = np.concatenate((xx[matchR].T,yy[matchR].T),axis=0)
    pts2L = np.concatenate((xx[matchL].T,yy[matchL].T),axis=0)
    
    #record the bvalues
    imageL= plt.imread(imprefixL1 +"%02d" % (1)+'.png')
    imageR = plt.imread(imprefixR1 +"%02d" % (1)+'.png')
    bvaluesL_list=[]
    bvaluesR_list=[]
    for i in range(pts2L.shape[1]):
        bvaluesL_list.append(imageL[pts2L[1][i]][pts2L[0][i]])
        bvaluesR_list.append(imageR[pts2R[1][i]][pts2R[0][i]])
    bvaluesL=np.array(bvaluesL_list).T
    bvaluesR=np.array(bvaluesR_list).T
    bvalues=(bvaluesL+bvaluesR)/2

    pts3 = triangulate(pts2L,camL,pts2R,camR)

    return pts2L,pts2R,pts3,bvalues

def meshgeneration\
(pts2L,pts2R,pts3,bvalues,imprefixL1,imprefixL2,imprefixR1,imprefixR2,blim,trithresh):
    from scipy.spatial import Delaunay
    
    # drop points which are outside the bounding box
    goodpts = np.nonzero((pts3[0,:]>blim[0])&(pts3[0,:]<blim[1]) & \
                         (pts3[1,:]>blim[2])&(pts3[1,:]<blim[3])& \
                         (pts3[2,:]>blim[4])&(pts3[2,:]<blim[5]))
                         
    pts3 = pts3[:,goodpts[0]]
    pts2L = pts2L[:,goodpts[0]]
    pts2R = pts2R[:,goodpts[0]]
    bvalues = bvalues[:,goodpts[0]]

    #
    # compute initial triangulation
    Triangles = Delaunay(pts2L.T)
    tri = Triangles.simplices

    #mesh smoothing
    def find_neighbors(pindex, triang):
        return triang.vertex_neighbor_vertices[1]\
    [triang.vertex_neighbor_vertices[0][pindex]:triang.vertex_neighbor_vertices[0][pindex+1]]

    for x in range (pts3.shape[1]):
        pts3[:,x] = np.mean(pts3[:,find_neighbors(x,Triangles)],axis=1)

    for x in range (pts3.shape[1]):
        pts3[:,x] = np.mean(pts3[:,find_neighbors(x,Triangles)],axis=1)

    for x in range (pts3.shape[1]):
        pts3[:,x] = np.mean(pts3[:,find_neighbors(x,Triangles)],axis=1)

    #
    # triangle pruning
    #
    d01 = np.sqrt(np.sum(np.power(pts3[:,tri[:,0]]-pts3[:,tri[:,1]],2),axis=0))
    d02 = np.sqrt(np.sum(np.power(pts3[:,tri[:,0]]-pts3[:,tri[:,2]],2),axis=0))
    d12 = np.sqrt(np.sum(np.power(pts3[:,tri[:,1]]-pts3[:,tri[:,2]],2),axis=0))

    goodtri = (d01<trithresh)&(d02<trithresh)&(d12<trithresh)
    tri = tri[goodtri,:]

    # remove any points which are not referenced in any triangle
    #
    #use np.unique applied to tri array to get a compact list of all vertices that are referenced:
    tokeep=np.unique(tri)
    map = np.zeros(pts3.shape[1])
    pts3=pts3[:,tokeep]
    bvalues = bvalues[:,tokeep]

    #update tri
    map[tokeep] = np.arange(0,tokeep.shape[0])
    tri=map[tri]

    return pts3,tri,bvalues
