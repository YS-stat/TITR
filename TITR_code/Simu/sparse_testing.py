# Sparse Image Learning
# We propose an alternating update algorithm combined with alternating direction method of multipliers (ADMM) for sparse scenario.
# This code includes the following functions:
##   ADMM(): ADMM algorithm to compute factor matrix
##   tensor_r_lasso(): tensor regression, return
#####          (eeB,eeC,ee1,ee2): estimated errors, (H,CH,theta): the estimated coefficients.
##   generate_data(): generate true coefficients when dimension=(50,50,2), return
#####          (B,C): true tensor coefficients, (theta_true): true vector coefficient.
##   generate_sample(): generate data sample for given coefficients, return
#####          (X,c,AX,y): tensor covariate, vector covariate, tensor covariate and scalar response,
##   pi_opt(),pi_hat(),pi_random(),pi_notensor(): 
#####           return average outcome value and corresponding MSEs under different treatment rules.
##   rankgrid(): grid search for different ranks, return corresponding estimation errors and estimated coefficients.
##   select_r(): select rank for tensor regression by loss in test set.
##   replication(): replication of the simulation
#####           n0: sample size, sigma: noise level
#####           return (mseB,mseC,mse1,mse2): estimated errors, 
#####                  (np.mean(V_opt),V_hat,V_random,V_notensor): average outcome values under different treatment rules, 
#####                  (MSE_hat,MSE_random,MSE_notensor): MSEs under different treatment rules,
#####                  (R_true,R_hat): indicator for correct rank estimation, estimated rank.


# Testing
# We use the wild bootstrap to test whether two treatment-related coefficients are equal to zero.
##   bootstrap_value():  calculate bootstrap statistics
##   generate_data2(), generate_sample2(): generate coefficients and data sample under dimension (10,10,10), rank (2,2,2) and sigma level 0.5.
##   replication(): calculate pvalues under different delta


import pdb,time
import pandas as pd
import math
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm,cholesky,solve,svd
import tensorly as tl
from tensorly import random
from tensorly import unfold
from scipy import linalg
import scipy
import warnings
warnings.filterwarnings('ignore')
import time
import random
import pydicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import multiprocessing
from tqdm import tqdm

# used functions
def ADMM(y, X, B0,lambd, gamma=1, tol=2e-4, iter_max=200):
    ###input response y and covariate X
    ###B0: initial
    ###lambd: lasso penalty
    ###gamma: ADMM-gamma
    # Initializations
    B = B0
    P = B0
    dim1, rank1 = B0.shape
    beta_b = B0.flatten().reshape((-1,1))
    beta = B0.flatten().reshape((-1,1))
    E = np.zeros((dim1, rank1))
    eta = np.zeros((dim1 * rank1, 1))
    u = np.zeros((len(y), 1))
    r = np.zeros((len(y), 1))
    tolerance1 = 1
    iter1 = 0
    while tolerance1 >= tol and iter1 <= iter_max:
        # Update beta
        beta = beta_b + eta / gamma
        beta = np.maximum(beta - lambd / gamma, 0) - np.maximum(-beta - lambd / gamma, 0)
        # Update r
        r = gamma*(y - np.dot(X, beta_b).reshape((-1, 1)) + u / gamma)/(gamma + 2)
        # Update beta_b
        tolerance2 = 1
        iter2 = 0
        beta_b0 = beta_b.copy()
        while tolerance2 >= 1e-3 and iter2 <= 50:
            beta_bb = np.dot(X.T, (u / gamma + y - r)) - eta / gamma + beta + P.flatten().reshape(-1,1) - E.flatten().reshape(-1,1) / gamma
            inv = np.linalg.inv(np.dot(X.T, X) + 2 * np.diag(np.ones(dim1 * rank1)))
            beta_bb = np.dot(inv, beta_bb)
            Bb = beta_bb.reshape((dim1, rank1))
            U, _, Vt = svd(Bb + E / gamma, full_matrices=False)
            P = np.dot(U, Vt)
            E = E + gamma * (Bb - P)
            tolerance2 = norm(beta_bb - beta_b0)
            beta_b0 = beta_bb
            iter2 = iter2 + 1
        beta_b = beta_bb
        # Update u and eta
        u = u + gamma * (y - np.dot(X, beta_b) - r)
        eta = eta + gamma * (beta_b - beta)
        B = beta_b.reshape((dim1, rank1))
        tolerance1 = norm(B - B0, ord='fro')
        B0 = B
        iter1 = iter1 + 1
    return B

#Tensor Regression 
def tensor_r_lasso(X,c,AX,y,B,C,theta_true,R1,R2,R3,lambda1,lambdaa,seed,tol,MAX_ITER1):
    ###X,AX: tensor covariates, c: vector covariate
    ###y: scalar response
    ###B,C: true tensor coefficients, theta_true: true vector coefficient
    ###R1,R2,R3: predetermined rank
    ###lambda1,lambdaa: tuning parameters(lasso penalty), seed: random seed
    ###tol: convergence threshold, MAX_ITER1: maximum number of iterations
    time_start=time.time()
    #initialization
    n, p1, p2, p3 = X.shape  
    n,l = c.shape 
    np.random.seed(seed)
    B1 = np.random.normal(0, 0.01, p1*R1).reshape((p1,R1))
    B2 = np.random.normal(0, 0.01, p2*R2).reshape((p2,R2))
    B3 = np.random.normal(0, 0.01, p3*R3).reshape((p3,R3))
    C1 = np.random.normal(0, 0.01, p1*R1).reshape((p1,R1))
    C2 = np.random.normal(0, 0.01, p2*R2).reshape((p2,R2))
    C3 = np.random.normal(0, 0.01, p3*R3).reshape((p3,R3))
    G = np.random.normal(0,0.01,size=(R1,R2,R3))
    AG = np.random.normal(0,0.01,size=(R1,R2,R3))
    theta = np.random.normal(0, 0.01, l).reshape((l,1))
    b1 = np.zeros((p1*R1,1)) 
    b2 = np.zeros((p2*R2,1))
    b3 = np.zeros((p3*R3,1))
    c1 = np.zeros((p1*R1,1))
    c2 = np.zeros((p2*R2,1))
    c3 = np.zeros((p3*R3,1))
    w1 = np.zeros((n,1))
    w2 = np.zeros((n,1))
    H = np.tensordot(np.tensordot(np.tensordot(G,B1,axes=(0,1)),B2,axes=(0,1)),B3,axes = (0,1))
    CH = np.tensordot(np.tensordot(np.tensordot(AG,C1,axes=(0,1)),C2,axes=(0,1)),C3,axes = (0,1))
    for i in range(n):
        w1[i,0] = np.sum(X[i,:,:,:]*H)
        w2[i,0] = np.sum(AX[i,:,:,:]*CH)
    
    y_0 = np.copy(y)
    loss = 0
    for t in range(0, MAX_ITER1):
        #B1
        yt = y_0 - c.dot(theta)
        y = yt - w2
        B1_old=np.copy(B1) 
        X1=np.zeros((n,R1*p1))
        G0 = unfold(G,0)
        for i in range(n):
            m1 = np.zeros((R1*p1,1))
            M0 = unfold(X[i,:,:,:],0) 
            m1 = np.ravel(M0.dot(np.kron(B2,B3)).dot(G0.T))
            X1[i,:] = m1.T  
        lambda2 = lambda1*norm(B2,ord=1)
        B1 = ADMM(y, X = X1, B0 = B1_old ,lambd = lambda2, gamma = 1)
        #B2
        B2_old=np.copy(B2)
        X2=np.zeros((n,R2*p2))
        G1 = unfold(G,1)
        for i in range(n):
            m2 = np.zeros((R2*p2,1))
            M1 = unfold(X[i,:,:,:],1) 
            m2 = np.ravel(M1.dot(np.kron(B1,B3)).dot(G1.T))
            X2[i,:] = m2.T       
        lambda2 = lambda1*norm(B1,ord=1)
        B2 = ADMM(y, X = X2, B0 = B2_old ,lambd = lambda2, gamma = 1)
        #B3
        B3_old=np.copy(B3)   
        X3=np.zeros((n,R3*p3))
        G2 = unfold(G,2)
        for i in range(n):
            m3=np.zeros((R3*p3,1))
            M2=unfold(X[i,:,:,:],2)
            m3=np.ravel(M2.dot(np.kron(B1,B2)).dot(G2.T))
            X3[i,:]=m3.T
        #lambda2 = lambda1*norm(B2,ord=1)*norm(B1,ord=1)
        #B3 = ADMM(y, X = X3, B0 = B3_old ,lambd = lambda2, gamma = 1)
        #b3=np.linalg.solve(X3.T.dot(X3), X3.T.dot(y))
        b3, residuals, rank, s = np.linalg.lstsq(X3.T.dot(X3), X3.T.dot(y), rcond=None)
        B3=b3.reshape((p3,R3))
        #G
        G_old = np.copy(G)
        XG = np.zeros((n,R1*R2*R3))
        for i in range(n):
            m = np.zeros((R1*R2*R3,1))
            Xvec = np.ravel(unfold(X[i,:,:,:],0))
            m = np.kron(np.kron(B1,B2),B3).T.dot(Xvec)
            XG[i,:]=m
        #bG = np.linalg.solve(XG.T.dot(XG), XG.T.dot(y))
        bG, residuals, rank, s = np.linalg.lstsq(XG.T.dot(XG), XG.T.dot(y), rcond=None)
        G = bG.reshape((R1,R2,R3))
        H_old=np.copy(H)
        H = np.zeros((p1,p2,p3))
        H = np.tensordot(np.tensordot(np.tensordot(G,B1,axes=(0,1)),B2,axes=(0,1)),B3,axes = (0,1))
        for i in range(n):
            w1[i,0]=np.sum(X[i,:,:,:]*H)
        y = yt - w1
        #C1
        C1_old=np.copy(C1) 
        X1=np.zeros((n,R1*p1))
        AG0 = unfold(AG,0)
        for i in range(n):
            m1=np.zeros((R1*p1,1))
            M0=unfold(AX[i,:,:,:],0)
            m1 = np.ravel(M0.dot(np.kron(C2,C3)).dot(AG0.T))
            X1[i,:]=m1.T        
        lambda2 = lambdaa*norm(C2,ord=1)
        C1 = ADMM(y, X = X1, B0 = C1_old ,lambd = lambda2, gamma = 1)
        #C2
        C2_old=np.copy(C2)
        X2=np.zeros((n,R2*p2))
        AG1 = unfold(AG,1)
        for i in range(n):
            m2=np.zeros((R2*p2,1))
            M1=unfold(AX[i,:,:,:],1)
            m2 = np.ravel(M1.dot(np.kron(C1,C3)).dot(AG1.T))
            X2[i,:]=m2.T       
        lambda2 = lambdaa*norm(C1,ord=1)
        C2 = ADMM(y, X = X2, B0 = C2_old ,lambd = lambda2, gamma = 1)
        #C3
        C3_old=np.copy(C3)   
        X3=np.zeros((n,R3*p3))
        AG2 = unfold(AG,2)
        for i in range(n):
            m3=np.zeros((R3*p3,1))
            M2=unfold(AX[i,:,:,:],2)
            m3 = np.ravel(M2.dot(np.kron(C1,C2)).dot(AG2.T))
            X3[i,:]=m3.T
        #lambda2 = lambdaa*norm(C2,ord=1)*norm(C1,ord=1)
        #C3 = ADMM(y, X = X3, B0 = C3_old ,lambd = lambda2, gamma = 1)
        #c3=np.linalg.solve(X3.T.dot(X3), X3.T.dot(y))
        c3, residuals, rank, s = np.linalg.lstsq(X3.T.dot(X3), X3.T.dot(y), rcond=None)
        C3=c3.reshape((p3,R3))
        #AG
        AG_old = np.copy(AG)
        XAG = np.zeros((n,R1*R2*R3))
        for i in range(n):
            m = np.zeros((R1*R2*R3,1))
            AXvec = np.ravel(unfold(AX[i,:,:,:],0))
            m = np.kron(np.kron(C1,C2),C3).T.dot(AXvec)
            XAG[i,:] = m
        #bAG = np.linalg.solve(XAG.T.dot(XAG),XAG.T.dot(y))
        bAG, residuals, rank, s = np.linalg.lstsq(XAG.T.dot(XAG), XAG.T.dot(y), rcond=None)
        AG = bAG.reshape(R1,R2,R3)
        #theta
        CH_old = np.copy(CH)
        CH = np.zeros((p1,p2,p3))
        CH = np.tensordot(np.tensordot(np.tensordot(AG,C1,axes=(0,1)),C2,axes=(0,1)),C3,axes = (0,1))
        for i in range(n):
            w2[i,0]=np.sum(AX[i,:,:,:]*CH)
        y = y_0 - w1 - w2
        theta_old = np.copy(theta)
        theta = np.linalg.solve(c.T.dot(c), c.T.dot(y))
        #estimation errors
        eeB = norm(H-B)
        eeC = norm(CH-C)
        ee1 = norm(theta[:(l//2)]-theta_true[:(l//2)])
        ee2 = norm(theta[(-l//2):]-theta_true[(-l//2):])
        print(t,eeB,eeC,ee1,ee2)
        loss_old = loss
        loss = eeC
        if t>5 and abs(loss-loss_old)<tol:
            break
    #print(t,eeB,eeC,ee1,ee2)
    time_end=time.time()
    time=time_end-time_start
    return time,eeB,eeC,ee1,ee2,H,CH,theta

#generate Ustar
def Ustar():
    a = np.zeros(5)
    a[0:3] = np.random.normal(0,1,3)
    b = np.zeros(5)
    b[3:5] = np.random.normal(0,1,2)
    U = np.vstack((a,b))
    U = U.T
    return U
def normalize(matrix):
    column_norms = np.linalg.norm(matrix, axis=0)
    normalized_matrix = matrix / column_norms
    return normalized_matrix

def generate_data(n,sigma,R = 2):
    B1 = normalize(np.vstack([Ustar() for _ in range(10)]))
    B2 = normalize(np.vstack([Ustar() for _ in range(10)]))
    random_matrix = np.random.standard_normal(size=(2,R))
    B3, R1 = np.linalg.qr(random_matrix)
    G = np.random.normal(0,1,size=(R,R,R))
    C3 = normalize(np.vstack((Ustar(),Ustar(),Ustar(),Ustar()))) #Typo (will be overwritten, but may influence random behavior).
    C1 = normalize(np.vstack([Ustar() for _ in range(10)]))
    C2 = normalize(np.vstack([Ustar() for _ in range(10)]))
    random_matrix1 = np.random.standard_normal(size=(2,R))
    C3, R2 = np.linalg.qr(random_matrix1)
    AG = np.random.normal(0,1,size=(R,R,R))
    B = np.tensordot(np.tensordot(np.tensordot(G,B1,axes=(0,1)),B2,axes=(0,1)),B3,axes = (0,1))
    C = np.tensordot(np.tensordot(np.tensordot(AG,C1,axes=(0,1)),C2,axes=(0,1)),C3,axes = (0,1))
    theta1=np.ones((5,1))
    theta2=np.ones((5,1))
    theta_true=np.ones((10,1))
    X = np.random.normal(0,1,size=(n,50,50,2))
    c1 = np.random.normal(0,1,size=(n,5))
    AX = np.zeros((n,50,50,2))
    c = np.zeros((n,10))
    y = np.zeros((n,1))
    for i in range(n):
        a = np.random.binomial(1., 0.7)
        c[i,0:5] = c1[i,:]
        c[i,5:10] = a*c1[i,:]
        AX[i,:,:,:] = a*X[i,:,:,:]
        y[i,0] = np.sum(X[i,:,:,:]*B)+c.dot(theta_true)[i,0]+np.sum(AX[i,:,:,:]*C)+np.random.normal(0,sigma,1)
    return X,c,AX,y,B,C,theta_true

def generate_sample(B,C,theta_true,n,sigma):
    X = np.random.normal(0,1,size=(n,50,50,2))
    c1 = np.random.normal(0,1,size=(n,5))
    AX = np.zeros((n,50,50,2))
    c = np.zeros((n,10))
    y = np.zeros((n,1))
    for i in range(n):
        a = np.random.binomial(1., 0.7)
        c[i,0:5] = c1[i,:]
        c[i,5:10] = a*c1[i,:]
        AX[i,:,:,:] = a*X[i,:,:,:]
        y[i,0] = np.sum(X[i,:,:,:]*B)+c.dot(theta_true)[i,0]+np.sum(AX[i,:,:,:]*C)+np.random.normal(0,sigma,1)
    return X,c,AX,y

# pi_opt
def pi_opt(n,X_test,c1_test,theta_true1,theta_true2,B,C): 
    pi_opt = np.zeros((n,1))
    V_opt = np.zeros((n,1))
    MSE1 = np.zeros((n,1))
    for i in range(n):
        if (np.sum(X_test[i,:,:,:]*C) + c1_test[i].dot(theta_true2)) > 0:
            pi_opt[i] = 1
            V_opt[i] = np.sum(X_test[i,:,:,:]*B) + c1_test[i].dot(theta_true1) + c1_test[i].dot(theta_true2) + np.sum(X_test[i,:,:,:]*C)
        else:
            V_opt[i] = np.sum(X_test[i,:,:,:]*B) + c1_test[i].dot(theta_true1)  
    return V_opt
# pi_hat
def pi_hat(n,X_test,c1_test,H,CH,theta,V_opt):
    pi_hat = np.zeros((n,1))
    V_hat = np.zeros((n,1))
    MSE2 = np.zeros((n,1))
    for i in range(n):
        if (np.sum(X_test[i,:,:,:]*CH) + c1_test[i].dot(theta[-5:])) > 0:
            pi_hat[i] = 1
            V_hat[i] = np.sum(X_test[i,:,:,:]*H) + c1_test[i].dot(theta[:5]) + c1_test[i].dot(theta[-5:]) + np.sum(X_test[i,:,:,:]*CH)
        else:
            V_hat[i] = np.sum(X_test[i,:,:,:]*H) + c1_test[i].dot(theta[:5])
        MSE2[i] = (V_hat[i]-V_opt[i])**2   
    MSE_hat = np.mean(MSE2)
    print("MSE_hat = {}".format(np.mean(MSE2)))
    return np.mean(V_hat),MSE_hat
# pi_random
def pi_random(n,X_test,c1_test,H,CH,theta,V_opt):
    pi_random = np.random.binomial(1., 0.5,size = n)
    V_random = np.zeros((n,1))
    MSE3 = np.zeros((n,1))
    for i in range(n):
        if pi_random[i] == 1:
            V_random[i] = np.sum(X_test[i,:,:,:]*H) + c1_test[i].dot(theta[:5]) + c1_test[i].dot(theta[-5:]) + np.sum(X_test[i,:,:,:]*CH)
        else:
            V_random[i] = np.sum(X_test[i,:,:,:]*H) + c1_test[i].dot(theta[:5])
        MSE3[i] = (V_random[i]-V_opt[i])**2
    MSE_random = np.mean(MSE3)
    print("MSE_random = {}".format(np.mean(MSE3)))
    return np.mean(V_random),MSE_random
# pi_notensor 
def pi_notensor(y,c,n,c1_test,V_opt):
    theta_notensor = np.linalg.solve(c.T.dot(c), c.T.dot(y))
    pi_notensor = np.zeros((n,1))
    V_notensor = np.zeros((n,1))
    MSE4 = np.zeros((n,1))
    for i in range(n):
        if (c1_test[i].dot(theta_notensor[-5:])) > 0:
            pi_notensor[i] = 1
            V_notensor[i] = c1_test[i].dot(theta_notensor[:5]) + c1_test[i].dot(theta_notensor[-5:])
        else:
            V_notensor[i] = c1_test[i].dot(theta_notensor[:5])
        MSE4[i] = (V_notensor[i]-V_opt[i])**2
    MSE_notensor = np.mean(MSE4)
    print("MSE_notensor = {}".format(np.mean(MSE4)))
    return np.mean(V_notensor),MSE_notensor

def rankgrid(X,c,AX,y,B,C,theta_true,lambda1,lambdaa,seed):
    Rgrid = np.array([[i, i, i] for i in range(1, 4)])
    time = np.zeros((Rgrid.shape[0],1))
    eeB = np.zeros((Rgrid.shape[0],1))
    eeC = np.zeros((Rgrid.shape[0],1))
    ee1 = np.zeros((Rgrid.shape[0],1))
    ee2 = np.zeros((Rgrid.shape[0],1))
    H = np.zeros((Rgrid.shape[0],50,50,2))
    CH = np.zeros((Rgrid.shape[0],50,50,2))
    theta = np.zeros((Rgrid.shape[0],10,1))
    for i in range(Rgrid.shape[0]):
        R = Rgrid[i,:]
        # try:
        #     [time[i],eeB[i],eeC[i],ee1[i],ee2[i],H[i,:,:,:],CH[i,:,:,:],theta[i,:,:]] = tensor_r_lasso(X,c,AX,y,B,C,theta_true,R[0],R[1],R[2],lambda1,lambdaa,seed,tol=0.001,MAX_ITER1=100)
        # except Exception as e:
        #     print(f"Error when rank = {R}: {e}")
        #     continue
        [time[i],eeB[i],eeC[i],ee1[i],ee2[i],H[i,:,:,:],CH[i,:,:,:],theta[i,:,:]] = tensor_r_lasso(X,c,AX,y,B,C,theta_true,R[0],R[1],R[2],lambda1,lambdaa,seed,tol=0.001,MAX_ITER1=90)
        print(i)
    return time,eeB,eeC,ee1,ee2,H,CH,theta

def select_r(time,eeB,eeC,ee1,ee2,H,CH,theta,n_test,y_test,X_test,AX_test,c_test):
    Rgrid = np.array([[i, i, i] for i in range(1, 4)])
    loss_value = np.full(Rgrid.shape[0],1000)
    for i in range(Rgrid.shape[0]):
        res = np.zeros((n_test,1))
        for j in range(n_test):
            res[j] = y_test[j,0] - np.sum(X_test[j,:,:,:]*H[i]) - c_test.dot(theta[i,:,:])[j,0] - np.sum(AX_test[j,:,:,:]*CH[i])
        loss_value[i] = (norm(res)**2)/n_test
    inx = np.argmin(loss_value) 
    R_hat = Rgrid[inx,:]
    return R_hat,time[inx][0],eeB[inx][0],eeC[inx][0],ee1[inx][0],ee2[inx][0],H[inx,:,:,:],CH[inx,:,:,:],theta[inx,:]

def replication(seed):
    lambda1=0.3
    lambdaa=0.7
    np.random.seed(1995)
    [X0,c0,AX0,y0,B,C,theta_true] = generate_data(n=600,sigma=1,R=2)
    np.random.seed(seed)
    n0=600
    sigma=1
    [X,c,AX,y] = generate_sample(B,C,theta_true,n0,sigma)
    #select Rank
    [time,eeB,eeC,ee1,ee2,H,CH,theta] = rankgrid(X,c,AX,y,B,C,theta_true,lambda1,lambdaa,seed)
    #testing sample1
    n_test=int(0.4*n0)
    [X_test,c_test,AX_test,y_test] = generate_sample(B,C,theta_true,n_test,sigma) 
    [R_hat,time0,mseB,mseC,mse1,mse2,H,CH,theta] = select_r(time,eeB,eeC,ee1,ee2,H,CH,theta,n_test,y_test,X_test,AX_test,c_test)
    print("R_hat = {}".format(R_hat))
    R_true = 0
    if R_hat[0]==2 & R_hat[1]==2 & R_hat[2]==2:
        R_true = 1  
    #testing sample2
    [X_test,c_test,AX_test,y_test] = generate_sample(B,C,theta_true,n_test,sigma) 
    c1_test = c_test[:,0:5]
    theta_true1 = np.ones((5,1))
    theta_true2 = np.ones((5,1))
    V_opt = pi_opt(n_test,X_test,c1_test,theta_true1,theta_true2,B,C)
    [V_hat,MSE_hat] = pi_hat(n_test,X_test,c1_test,H,CH,theta,V_opt)
    [V_random,MSE_random] = pi_random(n_test,X_test,c1_test,H,CH,theta,V_opt)
    [V_notensor,MSE_notensor] = pi_notensor(y,c,n_test,c1_test,V_opt)
    return time0,mseB,mseC,mse1,mse2,np.mean(V_opt),V_hat,V_random,V_notensor,MSE_hat,MSE_random,MSE_notensor,R_true,R_hat[1]


num_pro = 3
b = 99
results = []
if __name__ == '__main__':
    #seed = np.random.randint(1, 10000, size=b, dtype=int)
    a = time.time()
    pool = multiprocessing.Pool(num_pro)
    for i in range(b):
        #print(i)
        result = pool.apply_async(replication, (1996+i, ))
        results.append(result)
    final_results = [result.get() for result in results]
    
    pool.close()
    pool.join()
    
    b = time.time()
    time = b - a
    print(time)
    
    filename = '50_50_2_600.csv'
    np.savetxt(filename,np.array(final_results),fmt='%s',delimiter=',')


#testing
def bootstrap_value(V,n,y,X,AX,c,theta,H,CH,R0,lambda1,lambdaa,seed):
    res = np.zeros((n,1))
    y_b = np.zeros((n,1))
    for i in range(n):
        res[i] = y[i,0] - np.sum(X[i,:,:,:]*H) - c.dot(theta)[i,0] - np.sum(AX[i,:,:,:]*CH)
        y_b[i] = np.sum(X[i,:,:,:]*H) + c.dot(theta)[i,0] + np.sum(AX[i,:,:,:]*CH) + V[i]*res[i]
    [time,mseB,mseC,mse1,mse2,H_b,CH0_b,theta0_b] = tensor_r_lasso(X,c,AX,y_b,H,CH,theta,R0,R0,R0,lambda1,lambdaa,seed,tol=0.001,MAX_ITER1=50)
    a1 = np.dot(theta0_b[-5:].ravel()-theta[-5:].ravel(),theta0_b[-5:].ravel()-theta[-5:].ravel())
    a2 = np.dot(np.ravel(unfold(CH0_b,0))-np.ravel(unfold(CH,0)),np.ravel(unfold(CH0_b,0))-np.ravel(unfold(CH,0)))
    return a1,a2

def generate_data2(delta,R = 2):
    B1 = normalize(np.vstack((Ustar(),Ustar())))
    B2 = normalize(np.vstack((Ustar(),Ustar())))
    B3 = normalize(np.vstack((Ustar(),Ustar())))
    G = np.random.normal(0,1,size=(R,R,R))
    C1 = normalize(np.vstack((Ustar(),Ustar())))
    C2 = normalize(np.vstack((Ustar(),Ustar())))
    C3 = normalize(np.vstack((Ustar(),Ustar())))
    AG = np.random.normal(0,1,size=(R,R,R))
    B = np.tensordot(np.tensordot(np.tensordot(G,B1,axes=(0,1)),B2,axes=(0,1)),B3,axes = (0,1))
    C = np.tensordot(np.tensordot(np.tensordot(AG,C1,axes=(0,1)),C2,axes=(0,1)),C3,axes = (0,1))
    C = delta*C
    theta1=np.ones((5,1))
    theta2=np.ones((5,1))
    theta_true=np.ones((10,1))
    theta_true[-5:] = delta*theta_true[-5:]
    return B,C,theta_true
def generate_sample2(B,C,theta_true,n):
    X = np.random.normal(0,1,size=(n,10,10,10))
    c1 = np.random.normal(0,1,size=(n,5))
    AX = np.zeros((n,10,10,10))
    c = np.zeros((n,10))
    y = np.zeros((n,1))
    for i in range(n):
        a = np.random.binomial(1., 0.7)
        c[i,0:5] = c1[i,:]
        c[i,5:10] = a*c1[i,:]
        AX[i,:,:,:] = a*X[i,:,:,:]
        y[i,0] = np.sum(X[i,:,:,:]*B)+c.dot(theta_true)[i,0]+np.sum(AX[i,:,:,:]*C)+np.random.normal(0,0.5,1)
    return X,c,AX,y

def replication(seeds, result_queue):
    for seed in tqdm(seeds):
        lambda1=0.02
        lambdaa=0.02
        b = 100 #500
        delta = 0 #0.05 #0.1 #0.15 #0.2 #0.25
        n0 = 1000
        R0 = 2
        np.random.seed(2024)
        [B,C,theta_true] = generate_data2(delta,R=2)
        #print(norm(C))
        np.random.seed(seed)
        [X,c,AX,y] = generate_sample2(B,C,theta_true,n0)
        [time,mseB,mseC,mse1,mse2,H,CH,theta] = tensor_r_lasso(X,c,AX,y,B,C,theta_true,R0,R0,R0,lambda1,lambdaa,seed,tol=0.001,MAX_ITER1=50)
        results = []
        for i in range(b):
            np.random.seed(i)
            V = np.random.normal(0,1,size=n0)
            result = bootstrap_value(V,n0,y,X,AX,c,theta,H,CH,R0,lambda1,lambdaa,seed)
            results.append(result)
        l1 = np.array(results)[:,0] 
        l2 = np.array(results)[:,1] 
        p_val1 = (np.sum(l1 > np.dot(theta[-5:].ravel(),theta[-5:].ravel())))/b
        p_val2 = (np.sum(l2 > np.dot(np.ravel(unfold(CH,0)),np.ravel(unfold(CH,0)))))/b
        print("p_1 = {}".format(p_val1))
        print("p_2 = {}".format(p_val2))
        result_queue.put((p_val1,p_val2))
        #return p_val1,p_val2

num_pro = 2
repeat = 100
values = []
if __name__ == '__main__':
    np.random.seed(1995)
    seed = np.random.randint(1, 10000, size=repeat, dtype=int)
    a = time.time()
    stride = math.ceil(repeat/num_pro)
    pool = []
    result_queue = multiprocessing.Queue()
    for i in range(num_pro):
        p = multiprocessing.Process(target=replication,args=(seed[i*stride:(i+1)*stride],result_queue))
        pool.append(p)
        p.start()

    for p in pool:
        p.join()
    
    results = []
    while not result_queue.empty():
        result = result_queue.get()
        results.append(result)

    # pool = multiprocessing.Pool(num_pro)
    # for i in range(repeat):
    #     value = pool.apply_async(replication, (seed[i], ))
    #     values.append(value)
    # final_values = [value.get() for value in values]
    
    # pool.close()
    # pool.join()
    
    b = time.time()
    time = b - a
    print(time)
    
    filename = 'p_value.csv'
    np.savetxt(filename,np.array(results),fmt='%s',delimiter=',')
    
