import numpy as np
import scipy as sp
import scipy.ndimage
import imageio
from scipy.spatial.distance import cdist
import cv2
def box(img, r):
    """ O(1) box filter
        img - >= 2d image
        r   - radius of box filter
    """
    (rows, cols) = img.shape[:2]
    imDst = np.zeros_like(img)


    tile = [1] * img.ndim
    tile[0] = r
    imCum = np.cumsum(img, 0)
    imDst[0:r+1, :, ...] = imCum[r:2*r+1, :, ...]
    imDst[r+1:rows-r, :, ...] = imCum[2*r+1:rows, :, ...] - imCum[0:rows-2*r-1, :, ...]
    imDst[rows-r:rows, :, ...] = np.tile(imCum[rows-1:rows, :, ...], tile) - imCum[rows-2*r-1:rows-r-1, :, ...]

    tile = [1] * img.ndim
    tile[1] = r
    imCum = np.cumsum(imDst, 1)
    imDst[:, 0:r+1, ...] = imCum[:, r:2*r+1, ...]
    imDst[:, r+1:cols-r, ...] = imCum[:, 2*r+1 : cols, ...] - imCum[:, 0 : cols-2*r-1, ...]
    imDst[:, cols-r: cols, ...] = np.tile(imCum[:, cols-1:cols, ...], tile) - imCum[:, cols-2*r-1 : cols-r-1, ...]

    return imDst

def _gf_color(I, p, r, eps, s=None):
    """ Color guided filter
    I - guide image (rgb)
    p - filtering input (single channel)
    r - window radius
    eps - regularization (roughly, variance of non-edge noise)
    s - subsampling factor for fast guided filter
    """
    fullI = I
    fullP = p
    if s is not None:
        I = sp.ndimage.zoom(fullI, [1/s, 1/s, 1], order=1)
        p = sp.ndimage.zoom(fullP, [1/s, 1/s], order=1)
        r = round(r / s)

    h, w = p.shape[:2]
    N = box(np.ones((h, w)), r)

    mI_r = box(I[:,:,0], r) / N
    mI_g = box(I[:,:,1], r) / N
    mI_b = box(I[:,:,2], r) / N

    mP = box(p, r) / N

    # mean of I * p
    mIp_r = box(I[:,:,0]*p, r) / N
    mIp_g = box(I[:,:,1]*p, r) / N
    mIp_b = box(I[:,:,2]*p, r) / N

    # per-patch covariance of (I, p)
    covIp_r = mIp_r - mI_r * mP
    covIp_g = mIp_g - mI_g * mP
    covIp_b = mIp_b - mI_b * mP

    # symmetric covariance matrix of I in each patch:
    #       rr rg rb
    #       rg gg gb
    #       rb gb bb
    var_I_rr = box(I[:,:,0] * I[:,:,0], r) / N - mI_r * mI_r;
    var_I_rg = box(I[:,:,0] * I[:,:,1], r) / N - mI_r * mI_g;
    var_I_rb = box(I[:,:,0] * I[:,:,2], r) / N - mI_r * mI_b;

    var_I_gg = box(I[:,:,1] * I[:,:,1], r) / N - mI_g * mI_g;
    var_I_gb = box(I[:,:,1] * I[:,:,2], r) / N - mI_g * mI_b;

    var_I_bb = box(I[:,:,2] * I[:,:,2], r) / N - mI_b * mI_b;

    a = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            sig = np.array([
                [var_I_rr[i,j], var_I_rg[i,j], var_I_rb[i,j]],
                [var_I_rg[i,j], var_I_gg[i,j], var_I_gb[i,j]],
                [var_I_rb[i,j], var_I_gb[i,j], var_I_bb[i,j]]
            ])
            covIp = np.array([covIp_r[i,j], covIp_g[i,j], covIp_b[i,j]])
            a[i,j,:] = np.linalg.solve(sig + eps * np.eye(3), covIp)

    b = mP - a[:,:,0] * mI_r - a[:,:,1] * mI_g - a[:,:,2] * mI_b

    meanA = box(a, r) / N[...,np.newaxis]
    meanB = box(b, r) / N

    if s is not None:
        meanA = sp.ndimage.zoom(meanA, [s, s, 1], order=1)
        meanB = sp.ndimage.zoom(meanB, [s, s], order=1)
    if len(meanA)==len(fullI):
        q = np.sum(meanA * fullI, axis=2) + meanB
    else:
        q = np.sum(meanA[:min(len(meanA),len(fullI))] * fullI[:min(len(meanA),len(fullI))], axis=2) + meanB[:min(len(meanA),len(fullI))]

    return q


def _gf_gray(I, p, r, eps, s=None):
    """ grayscale (fast) guided filter
        I - guide image (1 channel)
        p - filter input (1 channel)
        r - window raidus
        eps - regularization (roughly, allowable variance of non-edge noise)
        s - subsampling factor for fast guided filter
    """
    if s is not None:
        Isub = sp.ndimage.zoom(I, 1/s, order=1)
        Psub = sp.ndimage.zoom(p, 1/s, order=1)
        r = round(r / s)
    else:
        Isub = I
        Psub = p


    (rows, cols) = Isub.shape

    N = box(np.ones([rows, cols]), r)

    meanI = box(Isub, r) / N
    meanP = box(Psub, r) / N
    corrI = box(Isub * Isub, r) / N
    corrIp = box(Isub * Psub, r) / N
    varI = corrI - meanI * meanI
    covIp = corrIp - meanI * meanP


    a = covIp / (varI + eps)
    b = meanP - a * meanI

    meanA = box(a, r) / N
    meanB = box(b, r) / N

    if s is not None:
        meanA = sp.ndimage.zoom(meanA, s, order=1)
        meanB = sp.ndimage.zoom(meanB, s, order=1)

    q = meanA * I + meanB
    return q


def _gf_colorgray(I, p, r, eps, s=None):
    """ automatically choose color or gray guided filter based on I's shape """
    if I.ndim == 2 or I.shape[2] == 1:
        return _gf_gray(I, p, r, eps, s)
    elif I.ndim == 3 and I.shape[2] == 3:
        return _gf_color(I, p, r, eps, s)
    else:
        print("Invalid guide dimensions:", I.shape)


def guided_filter(I, p, r, eps, s=None):
    """ run a guided filter per-channel on filtering input p
        I - guide image (1 or 3 channel)
        p - filter input (n channel)
        r - window raidus
        eps - regularization (roughly, allowable variance of non-edge noise)
        s - subsampling factor for fast guided filter
    """
    if p.ndim == 2:
        p3 = p[:,:,np.newaxis]
    else:
        p3 = p
    out = np.zeros_like(p3)
    for ch in range(p3.shape[2]):
        tmp = _gf_colorgray(I, p3[:,:,ch], r, eps, s)
        out[:len(tmp),:,ch] = tmp
    return np.squeeze(out) if p.ndim == 2 else out

##Spectral convolution##
def calc_degree_matrix_norm(a):
    return np.power(a.sum(axis=1)+1e-5,-0.5)

def create_graph_lapl_norm(img):
    print("img size:",img.shape)
    h,w,c = img.shape
    L_norm = np.zeros((w*h,w*h,c))
    for ch in range(c):
        #create adjancency matrix: a
        col,row = np.meshgrid(np.arange(w),np.arange(h))
        coord = np.stack((col,row),axis=2).reshape(-1,2)/(img.shape[0]*img.shape[1])
        dist = cdist(coord,coord)
        sigma = 0.005 * np.pi
        #kernel function for adjancency matrix
        a = np.exp(-dist/sigma**2)
        a[a<0.05] = 0
        for i in range(a.shape[0]):
             for j in range(a.shape[1]):
                if a[i,j]>0:
                    a[i,j] = img[i//w][i%w][ch] * img[j//w][j%w][ch] * a[i,j]
        a +=  np.eye(a.shape[-1])
        D_norm = calc_degree_matrix_norm(a)
        tmp = D_norm.reshape(-1,1)*a*D_norm.reshape(1,-1)
        L_norm[:,:,ch] = tmp
    return L_norm

def test_gf():
    test = imageio.imread('test_original.jpg').astype(np.float32) / 255
    test_ds = cv2.resize(test,(50,67))
    r = 2
    eps = 0.001
    test_L = np.zeros((test.shape[0],test.shape[1],test.shape[2]))
    test_smoothed_L = np.zeros((test.shape[0],test.shape[1],test.shape[2]))
    L = create_graph_lapl_norm(test_ds)

    for ch in range(test.shape[2]):
        tmp = np.matmul(L[:,:,ch],test_ds[:,:,ch].reshape(-1,1)).reshape(test_ds.shape[0],-1)
        #print(tmp.shape,test_L[:,:,ch].shape,cv2.resize(tmp,(test.shape[1],test.shape[0])).shape)
        test_L[:,:,ch] = cv2.resize(tmp,(test.shape[1],test.shape[0]))
    test_L_smoothed = guided_filter(test_L, test, r, eps)
    test_smoothed = cv2.resize(guided_filter(test_ds, test_ds, r, eps),(test.shape[1],test.shape[0]))
    test_smoothed_ds = cv2.resize(test_smoothed,(50,67))
    L_L = create_graph_lapl_norm(test_smoothed_ds)
    for ch in range(test.shape[2]):
        tmp = np.matmul(L_L[:,:,ch],test_smoothed_ds[:,:,ch].reshape(-1,1)).reshape(test_ds.shape[0],-1)
        test_smoothed_L[:,:,ch] = cv2.resize(tmp,(test.shape[1],test.shape[0]))
    imageio.imwrite('test_smoothed.jpg', test_smoothed)
    imageio.imwrite('test_L.jpg', test_L)
    imageio.imwrite('test_smoothed_L.jpg', test_smoothed_L)
    imageio.imwrite('test_L_smoothed.jpg', test_L_smoothed)
    #test_smoothed_s4 = guided_filter(test, test, r, eps, s=4)
    #imageio.imwrite('test_smoothed_s4.png', test_smoothed_s4)

def main():
    test_gf()
if __name__=='__main__':
    main()