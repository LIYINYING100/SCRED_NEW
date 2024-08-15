# -*- coding: utf-8 -*-
import torch
import numpy as np
from PIL import Image


def co_occurence_matrix(P):
    #P = P.view(160, 64, 64)
    #data =  P.cpu().detach().numpy()
    cooccurence_matrix = np.dot(P.transpose(),P)#data.transpose(),data)
    cooccurrence_matrix_diagonal = np.diagonal(cooccurence_matrix)
    with np.errstate(divide='ignore', invalid='ignore'):
        cooccurrence_matrix_percentage = np.nan_to_num(np.true_divide(cooccurence_matrix, cooccurrence_matrix_diagonal[:, None]))

    return cooccurrence_matrix_percentage

def compute_plus_minus(P, px_plus_y, px_minus_y):
    H = P.shape[0]
    W = P.shape[1]
    for i in range(H):
        for j in range(W):
            px_plus_y[i+j] += P[i,j]
    px_minus_y = [sum([P[i,j] for i in range(H) for j in range(W) if np.abs(i-j)==k]) for k in range(H)]
    return px_plus_y,px_minus_y

def entropy(P):
    p = P.ravel()
    p1 = p.copy()
    p1 += (p==0)
    return -np.dot(np.log2(p1), p)

def haralick_features(P):
    #print(P)
    feats = []
    cmat = co_occurence_matrix(P)
    #print(cmat)
    T = cmat.sum()
    #print('T')
    #print(T)
    maxv = len(cmat)
    k = np.arange(maxv)
    k2 = k**2
    tk = np.arange(2*maxv)
    tk2 = tk**2
    i,j = np.mgrid[:maxv,:maxv]
    ij = i*j
    i_j2_p1 = (i - j)**2
    i_j2_p1 += 1
    i_j2_p1 = 1. / i_j2_p1
    i_j2_p1 = i_j2_p1.ravel()
    px_plus_y = np.empty(2*maxv, np.double)
    px_minus_y = np.empty(maxv, np.double)

    if T == 0.0:
        p =cmat
    else:
        p = cmat / float(T)
    pravel = p.ravel()
    px = p.sum(0)
    py = p.sum(1)

    ux = np.dot(px, k)
    uy = np.dot(py, k)
    vx = np.dot(px, k2) - ux**2
    vy = np.dot(py, k2) - uy**2

    sx = np.sqrt(vx)
    sy = np.sqrt(vy)
    px_plus_y.fill(0)
    px_minus_y.fill(0)
    px_plus_y,px_minus_y = compute_plus_minus(p, px_plus_y, px_minus_y)
    px_plus_y = np.array(px_plus_y)
    px_minus_y = np.array(px_minus_y)

    f0 = np.dot(pravel, pravel)
    feats.append(f0)

    f1 = np.dot(k2, px_minus_y)
    feats.append(f1)

    if sx == 0. or sy == 0. :
        f2 = 1.
        feats.append(f2)
    else:
        f2 = (1. / sx / sy) * (np.dot(ij.ravel(), pravel) - ux * uy)
        feats.append(f2)

    f3 = vx
    feats.append(f3)

    f4 = np.dot(i_j2_p1, pravel)
    feats.append(f4)

    f5 = np.dot(tk, px_plus_y)
    feats.append(f5)

    f6 = np.dot(tk2, px_plus_y) - f5**2
    feats.append(f6)

    f7 = entropy(px_plus_y)
    feats.append(f7)

    f8 = entropy(pravel)
    feats.append(f8)

    f9 = np.var(px_minus_y)
    feats.append(f9)

    f10 = entropy(px_minus_y)
    feats.append(f10)

    HX = entropy(px)
    HY = entropy(py)
    crosspxpy = np.outer(px,py)
    crosspxpy += (crosspxpy == 0)
    crosspxpy = crosspxpy.ravel()
    HXY1 = -np.dot(pravel, np.log2(crosspxpy))
    HXY2 = entropy(crosspxpy)

    if max(HX, HY) == 0:
        f11 = (f8-HXY1)
        feats.append(f11)

    else:
        f11 = (f8-HXY1)/max(HX,HY)
        feats.append(f11)
    f12 = np.sqrt(max(0,1 - np.exp( -2. * (HXY2 - f8))))
    feats.append(f12)
    f13 = 0.
    feats.append(f13)
    features = np.array(feats)
    #features = torch.from_numpy(features)
    #f_abs = abs(features)
    #f_mean = f_abs.mean()
    return features

def hara_loss(x, y):
    h1 = haralick_features(x)
    h2 = haralick_features(y)
    h = h1-h2
    h = abs(h)
    l = h.mean()
    return l

def main():   
    #image1 = torch.rand(64,64)
    #image2 = torch.rand(64,64)
    image1 = np.load('')
    img1 = image1[150:200 ,150:200]
    image2 = np.load('')
    img2 = image2[150:200 ,150:200]
    #print('image')
    #h = haralick_features(img)
    #print(h)
    #print(image1)
    #print(image2)
    l = hara_loss(img1,img2)
    print(l)
    #print(m)
    
if __name__=='__main__':
    main()
