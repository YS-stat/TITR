# Low-dimensional Image Learning.
# The estimation is obtained by an alternating update algorithm based on Tucker decomposition. 
# This code includes the following functions:
##   tensor_r_tucker(): tensor regression, return
#####          (mseB,mseC,mse1,mse2): estimated errors, (H,CH,theta): the estimated coefficients, (BIC): Value of BIC.
##   generate_data(): generate data, return
#####          (X,c,AX,y): tensor covariate, vector covariate, tensor covariate and scalar response,
#####          (B,C): true tensor coefficients, (theta_true): true vector coefficient.
##   pi_opt(),pi_hat(),pi_random(),pi_notensor(): 
#####           return average outcome value and corresponding MSEs under different treatment rules.
##   select_rank(): select rank for tensor regression by BIC
##   replication(): replication of the simulation:
#####           n0: sample size, R0: true rank, p: dimesion of tensor, sigma: noise level
#####           return (mseB,mseC,mse1,mse2): estimated errors, 
#####                  (np.mean(V_opt),V_hat,V_random,V_notensor): average outcome values under different treatment rules, 
#####                  (MSE_hat,MSE_random,MSE_notensor): MSEs under different treatment rules,
#####                  (R_true): indicator for correct rank estimation.

import pdb,time
import pandas as pd
import math
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm,cholesky
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

#used functions
#Tensor Regression 
def tensor_r_tucker(X,c,AX,y,B,C,theta_true,R1,R2,R3,tol,MAX_ITER):
    ###X,AX: tensor covariates, c: vector covariate
    ###y: scalar response
    ###B,C: true tensor coefficients, theta_true: true vector coefficient
    ###R1,R2,R3: predetermined rank
    ###tol: convergence threshold, MAX_ITER: maximum number of iterations
    time_start=time.time()
    #initialization
    n, p1, p2, p3 = X.shape  
    n,l = c.shape 
    B1 = np.random.normal(0, 0.01, p1*R1).reshape((p1,R1))
    B2 = np.random.normal(0, 0.01, p2*R2).reshape((p2,R2))
    B3 = np.random.normal(0, 0.01, p3*R3).reshape((p3,R3))
    C1 = np.random.normal(0, 0.01, p1*R1).reshape((p1,R1))
    C2 = np.random.normal(0, 0.01, p2*R2).reshape((p2,R2))
    C3 = np.random.normal(0, 0.01, p3*R3).reshape((p3,R3))
    G = np.random.normal(0,0.01,size=(R1,R2,R3))
    AG = np.random.normal(0,0.01,size=(R1,R2,R3))
    H = np.zeros((p1,p2,p3)) 
    CH = np.zeros((p1,p2,p3))
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
    mse = 0
    for t in range(0, MAX_ITER):
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
        b1=np.linalg.solve(X1.T.dot(X1), X1.T.dot(y)) 
        B1=b1.reshape((p1,R1))
        #B2
        B2_old=np.copy(B2)
        X2=np.zeros((n,R2*p2))
        G1 = unfold(G,1)
        for i in range(n):
            m2 = np.zeros((R2*p2,1))
            M1 = unfold(X[i,:,:,:],1) 
            m2 = np.ravel(M1.dot(np.kron(B1,B3)).dot(G1.T))
            X2[i,:] = m2.T       
        b2=np.linalg.solve(X2.T.dot(X2), X2.T.dot(y))
        B2=b2.reshape((p2,R2))
        #B3
        B3_old=np.copy(B3)   
        X3=np.zeros((n,R3*p3))
        G2 = unfold(G,2)
        for i in range(n):
            m3=np.zeros((R3*p3,1))
            M2=unfold(X[i,:,:,:],2)
            m3=np.ravel(M2.dot(np.kron(B1,B2)).dot(G2.T))
            X3[i,:]=m3.T
        b3=np.linalg.solve(X3.T.dot(X3), X3.T.dot(y))
        B3=b3.reshape((p3,R3))
        #G
        G_old = np.copy(G)
        XG = np.zeros((n,R1*R2*R3))
        for i in range(n):
            m = np.zeros((R1*R2*R3,1))
            Xvec = np.ravel(unfold(X[i,:,:,:],0))
            m = np.kron(np.kron(B1,B2),B3).T.dot(Xvec)
            XG[i,:]=m
        bG = np.linalg.solve(XG.T.dot(XG), XG.T.dot(y))
        G = bG.reshape((R1,R2,R3))
        H_old=np.copy(H)
        H=np.zeros((p1,p2,p3))
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
        c1=np.linalg.solve(X1.T.dot(X1), X1.T.dot(y))
        C1=c1.reshape((p1,R1))
        #C2
        C2_old=np.copy(C2)
        X2=np.zeros((n,R2*p2))
        AG1 = unfold(AG,1)
        for i in range(n):
            m2=np.zeros((R2*p2,1))
            M1=unfold(AX[i,:,:,:],1)
            m2 = np.ravel(M1.dot(np.kron(C1,C3)).dot(AG1.T))
            X2[i,:]=m2.T       
        c2=np.linalg.solve(X2.T.dot(X2), X2.T.dot(y))
        C2=c2.reshape((p2,R2))
        #C3
        C3_old=np.copy(C3)   
        X3=np.zeros((n,R3*p3))
        AG2 = unfold(AG,2)
        for i in range(n):
            m3=np.zeros((R3*p3,1))
            M2=unfold(AX[i,:,:,:],2)
            m3 = np.ravel(M2.dot(np.kron(C1,C2)).dot(AG2.T))
            X3[i,:]=m3.T
        c3=np.linalg.solve(X3.T.dot(X3), X3.T.dot(y))
        C3=c3.reshape((p3,R3))
        #AG
        AG_old = np.copy(AG)
        XAG = np.zeros((n,R1*R2*R3))
        for i in range(n):
            m = np.zeros((R1*R2*R3,1))
            AXvec = np.ravel(unfold(AX[i,:,:,:],0))
            m = np.kron(np.kron(C1,C2),C3).T.dot(AXvec)
            XAG[i,:] = m
        bAG = np.linalg.solve(XAG.T.dot(XAG),XAG.T.dot(y))
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
        #mse
        mseB = norm(H-B)
        mseC = norm(CH-C)
        mse1 = norm(theta[:(l//2)]-theta_true[:(l//2)])
        mse2 = norm(theta[(-l//2):]-theta_true[(-l//2):])
        #mseB = norm(H-H_old)/max(1,norm(H_old))
        #mseC = norm(CH-CH_old)/max(1,norm(CH_old))
        #mse1 = norm(theta[:(l//2)]-theta_old[:(l//2)])/max(1,norm(theta_old[:(l//2)]))
        #mse2 = norm(theta[(-l//2):]-theta_old[(-l//2):])/max(1,norm(theta_old[(-l//2):]))
        #print(t,mseB,mseC,mse1,mse2)
        mse_old = mse
        mse = mseC
        if t>5 and abs(mse-mse_old)<tol:
            break 
    print(t,mseB,mseC,mse1,mse2)
    #calculate BIC
    res = np.zeros((n,1))
    for i in range(n):
            res[i] = y_0[i,0] - np.sum(X[i,:,:,:]*H) - c.dot(theta)[i,0] - np.sum(AX[i,:,:,:]*CH)
    loss = norm(res)**2/n
    df = R1*R2*R3+R1*(p1-R1)+R2*(p2-R2)+R3*(p3-R3)
    #BIC = n*np.log(loss)+(df+1)*np.log(n)
    BIC = np.log(loss)+ df*(np.log(n))/(2*n)
    
    time_end=time.time()
    time=time_end-time_start
    return time,mseB,mseC,mse1,mse2,H,CH,theta,BIC

# simulate tensor
def generate_data(n,p,R,sigma):
    ###n: sample size, p: dimension of tensor
    ###R: rank of tensor, sigma: noise level
    random_matrix1 = np.random.standard_normal(size=(p,R))
    B1, R1 = np.linalg.qr(random_matrix1)
    random_matrix2 = np.random.standard_normal(size=(p,R))
    B2, R2 = np.linalg.qr(random_matrix2)
    random_matrix3 = np.random.standard_normal(size=(p,R))
    B3, R3 = np.linalg.qr(random_matrix3)
    G = np.random.normal(0,1,size=(R,R,R))
    random_matrix4 = np.random.standard_normal(size=(p,R))
    C1, R4 = np.linalg.qr(random_matrix4)
    random_matrix5 = np.random.standard_normal(size=(p,R))
    C2, R5 = np.linalg.qr(random_matrix5)
    random_matrix6 = np.random.standard_normal(size=(p,R))
    C3, R6 = np.linalg.qr(random_matrix6)
    AG = np.random.normal(0,1,size=(R,R,R))
    B = np.tensordot(np.tensordot(np.tensordot(G,B1,axes=(0,1)),B2,axes=(0,1)),B3,axes = (0,1))
    C = np.tensordot(np.tensordot(np.tensordot(AG,C1,axes=(0,1)),C2,axes=(0,1)),C3,axes = (0,1))
    theta1=np.ones((5,1))
    theta2=np.ones((5,1))
    theta_true=np.ones((10,1))
    X = np.random.normal(0,1,size=(n,p,p,p))
    c1 = np.random.normal(0,1,size=(n,5))
    AX = np.zeros((n,p,p,p))
    c = np.zeros((n,10))
    y = np.zeros((n,1))
    for i in range(n):
        a = np.random.binomial(1., 0.7)
        c[i,0:5] = c1[i,:]
        c[i,5:10] = a*c1[i,:]
        AX[i,:,:,:] = a*X[i,:,:,:]
        y[i,0] = np.sum(X[i,:,:,:]*B)+c.dot(theta_true)[i,0]+np.sum(AX[i,:,:,:]*C)+np.random.normal(0,sigma,1)
    tol = 0.001
    MAX_ITER = 100
    return X,c,AX,y,B,C,theta_true

# pi_opt
def pi_opt(n,X_test,c1_test,theta_true1,theta_true2,B,C): 
    pi_opt = np.zeros((n,1))
    V_opt = np.zeros((n,1))
    for i in range(n):
        if (np.sum(X_test[i,:,:,:]*C) + c1_test[i].dot(theta_true2)) > 0:
            pi_opt[i] = 1
            V_opt[i] = np.sum(X_test[i,:,:,:]*B) + c1_test[i].dot(theta_true1) + c1_test[i].dot(theta_true2) + np.sum(X_test[i,:,:,:]*C)
        else:
            V_opt[i] = np.sum(X_test[i,:,:,:]*B) + c1_test[i].dot(theta_true1)  
    return V_opt
# pi_hat (estimated ITR)
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
# pi_random (randomly assigns the treatment)
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
# pi_notensor (without using any image data but only with covariate information)
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
# select model by BIC
def select_rank(p,X,c,AX,y,B,C,theta_true):
    Rgrid = np.array(np.meshgrid([2,3], [2,3], [2,3])).T.reshape(-1, 3)
    BIC_value = np.full((Rgrid.shape[0],1),1000)
    time1 = np.zeros((Rgrid.shape[0],1))
    eeB = np.zeros((Rgrid.shape[0],1))
    eeC = np.zeros((Rgrid.shape[0],1))
    ee1 = np.zeros((Rgrid.shape[0],1))
    ee2 = np.zeros((Rgrid.shape[0],1))
    H = np.zeros((Rgrid.shape[0],p,p,p))
    CH = np.zeros((Rgrid.shape[0],p,p,p))
    theta = np.zeros((Rgrid.shape[0],10,1))
    for i in range(Rgrid.shape[0]):
        R = Rgrid[i,:]
        try:
            [time1[i],eeB[i],eeC[i],ee1[i],ee2[i],H[i,:,:,:],CH[i,:,:,:],theta[i,:,:],BIC] = tensor_r_tucker(X,c,AX,y,B,C,theta_true,R[0],R[1],R[2],tol=0.001,MAX_ITER=100)
        except Exception as e:
            print(f"Error when rank = {R}: {e}")
            continue
        print(i)
        BIC_value[i] = BIC
    inx = np.argmin(BIC_value) 
    R_hat = Rgrid[np.argmin(BIC_value),:]
    return R_hat,time1[inx][0],eeB[inx][0],eeC[inx][0],ee1[inx][0],ee2[inx][0],H[inx,:,:,:],CH[inx,:,:,:],theta[inx,:]

import time
def replication(seed):
    np.random.seed(seed)
    n0=1000  #2000
    R0=2  #3
    p=10
    sigma=0.5
    [X,c,AX,y,B,C,theta_true] = generate_data(n=n0,p=p,R=R0,sigma=sigma)
    #select Rank
    [R_hat,time0,mseB,mseC,mse1,mse2,H,CH,theta]=select_rank(p,X,c,AX,y,B,C,theta_true)
    R_true = 0
    if R_hat[0]==R0 & R_hat[1]==R0 & R_hat[2]==R0:
        R_true = 1  
    #testing sample
    n=int(0.4*n0)
    [X_test,c_test,AX_test,y_test,B_test,C_test,theta_true] = generate_data(n,p=p,R=R0,sigma=sigma)
    c1_test = c_test[:,0:5]
    theta_true1 = np.ones((5,1))
    theta_true2 = np.ones((5,1))
    V_opt = pi_opt(n,X_test,c1_test,theta_true1,theta_true2,B,C)
    [V_hat,MSE_hat] = pi_hat(n,X_test,c1_test,H,CH,theta,V_opt)
    [V_random,MSE_random] = pi_random(n,X_test,c1_test,H,CH,theta,V_opt)
    [V_notensor,MSE_notensor] = pi_notensor(y,c,n,c1_test,V_opt)
    return time0,mseB,mseC,mse1,mse2,np.mean(V_opt),V_hat,V_random,V_notensor,MSE_hat,MSE_random,MSE_notensor,R_true

# parallel computing
num_pro = 2
b = 100
results = []
if __name__ == '__main__':
    np.random.seed(2024)
    seed = np.random.randint(1, 10000, size=b, dtype=int)
    a = time.time()
    pool = multiprocessing.Pool(num_pro)
    for i in range(b):
        #print(i)
        result = pool.apply_async(replication, (seed[i], ))
        results.append(result)
    final_results = [result.get() for result in results]
    
    pool.close()
    pool.join()

    R_true_all = np.array([r[12] for r in final_results]) 
    #print(final_results)
    R_true_rate = np.mean(R_true_all)
    print(R_true_rate)
    b = time.time()
    time = b - a
    print(time)
    print("time = {}".format(b-a))
    
    filename = '1000_10_2_0.5.csv'
    stacked_results = np.vstack(final_results)
    np.savetxt(filename, stacked_results, fmt='%s', delimiter=',')

