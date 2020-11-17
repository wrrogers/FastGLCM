# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import data


def fast_glcm(img, vmin=0, vmax=255, nbit=8, kernel_size=5):
    mi, ma = vmin, vmax
    ks = kernel_size
    h,w = img.shape

    # digitize
    bins = np.linspace(mi, ma+1, nbit+1)
    gl1 = np.digitize(img, bins) - 1
    gl2 = np.append(gl1[:,1:], gl1[:,-1:], axis=1)

    # make glcm
    glcm = np.zeros((nbit, nbit, h, w), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            mask = ((gl1==i) & (gl2==j))
            glcm[i,j, mask] = 1

    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            glcm[i,j] = cv2.filter2D(glcm[i,j], -1, kernel)

    glcm = glcm.astype(np.float32)
    return glcm

def get_feature_map(img, feature, vmin=0, vmax=255, nbit=8, ks=5):
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    
    if feature == 'mean':
        mean = np.zeros((h,w), dtype=np.float32)
        for i in range(nbit):
            for j in range(nbit):
                mean += glcm[i,j] * i / (nbit)**2
        out = mean

    elif feature == 'std':
        mean = np.zeros((h,w), dtype=np.float32)
        for i in range(nbit):
            for j in range(nbit):
                mean += glcm[i,j] * i / (nbit)**2
    
        std2 = np.zeros((h,w), dtype=np.float32)
        for i in range(nbit):
            for j in range(nbit):
                std2 += (glcm[i,j] * i - mean)**2
    
        std = np.sqrt(std2)
        out = std

    elif feature == 'contrast':
        cont = np.zeros((h,w), dtype=np.float32)
        for i in range(nbit):
            for j in range(nbit):
                cont += glcm[i,j] * (i-j)**2    
        out = cont

    elif feature == 'dissimilarity':
        diss = np.zeros((h,w), dtype=np.float32)
        for i in range(nbit):
            for j in range(nbit):
                diss += glcm[i,j] * np.abs(i-j)
        out = diss
    
    elif feature == 'homogeneity':
        homo = np.zeros((h,w), dtype=np.float32)
        for i in range(nbit):
            for j in range(nbit):
                homo += glcm[i,j] / (1.+(i-j)**2)
        out = homo

    elif feature == 'asm':
        asm = np.zeros((h,w), dtype=np.float32)
        for i in range(nbit):
            for j in range(nbit):
                asm  += glcm[i,j]**2    
        out = asm

    elif feature == 'energy':
        asm = np.zeros((h,w), dtype=np.float32)
        for i in range(nbit):
            for j in range(nbit):
                asm  += glcm[i,j]**2    
        ene = np.sqrt(asm)
        out = ene

    elif feature == 'max':
        max_  = np.max(glcm, axis=(0,1))
        out = max_

    elif feature == 'entropy':
        pnorm = glcm / np.sum(glcm, axis=(0,1)) + 1./ks**2
        ent  = np.sum(-pnorm * np.log(pnorm), axis=(0,1))
        out = ent

    return out

if __name__ == '__main__':
    nbit = 8
    ks = 5
    mi, ma = 0, 255

    img = data.camera()
    
    plt.imshow(img)
    plt.show()

    #img[:,:w//2] = img[:,:w//2]//2+127
    
    glcm_contrast = get_feature_map(img, 'contrast')
    
    plt.imshow(glcm_contrast)
    plt.show()
    
