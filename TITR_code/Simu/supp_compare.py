# Supplementary Comparison between Q-learning and A-learning.
# This code compares Q-learning and A-learning for estimating individualized treatment rules under two scenarios.
# This code includes the following functions:
##   tensor_r_tucker(): tensor regression, return
#####          (mseB,mseC,mse1,mse2): estimated errors,
#####          (H,CH,theta): the estimated coefficients.
##   generate_data(): generate data under the two treatment assignment regimes, return
#####          (X,c,AX,y): tensor covariate, vector covariate, treatment-weighted
#####          tensor covariate and scalar response,
#####          (B,C): true tensor coefficients, (theta_true): true vector coefficient,
#####          (A_list,c1): treatment indicators and baseline scalar covariates.
##   pi_opt(): calculate the oracle optimal treatment rule and corresponding value.
##   pi_hat(): calculate the estimated treatment rule, average value and decision accuracy.
##   replication_compare(): replication of the simulation comparing Q-learning and A-learning:
#####          n0: sample size, R0: true rank, p: dimension of tensor, sigma: noise level
#####          return (V_opt_mean,V_Q,V_A): oracle value, Q-learning value and A-learning value,
#####                 (acc_Q,acc_A): decision accuracies of Q-learning and A-learning.


import pandas as pd
import math
import numpy as np
import scipy.sparse as sparse
from numpy.linalg import norm
import tensorly as tl
from tensorly import unfold
import warnings
warnings.filterwarnings('ignore')
import time
from sklearn.linear_model import LogisticRegression
import multiprocessing 
from scipy.special import expit
import traceback
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import PolynomialFeatures


# ---------------------------------------------------------
def tensor_r_tucker(X,c,AX,y,B,C,theta_true,R1,R2,R3,tol,MAX_ITER):
    time_start=time.time()
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
        yt = y_0 - c.dot(theta)
        y = yt - w2
        X1=np.zeros((n,R1*p1))
        G0 = unfold(G,0)
        for i in range(n):
            M0 = unfold(X[i,:,:,:],0) 
            m1 = np.ravel(M0.dot(np.kron(B2,B3)).dot(G0.T))
            X1[i,:] = m1.T  
        b1=np.linalg.solve(X1.T.dot(X1), X1.T.dot(y)) 
        B1=b1.reshape((p1,R1))
        
        X2=np.zeros((n,R2*p2))
        G1 = unfold(G,1)
        for i in range(n):
            M1 = unfold(X[i,:,:,:],1) 
            m2 = np.ravel(M1.dot(np.kron(B1,B3)).dot(G1.T))
            X2[i,:] = m2.T       
        b2=np.linalg.solve(X2.T.dot(X2), X2.T.dot(y))
        B2=b2.reshape((p2,R2))
        
        X3=np.zeros((n,R3*p3))
        G2 = unfold(G,2)
        for i in range(n):
            M2=unfold(X[i,:,:,:],2)
            m3=np.ravel(M2.dot(np.kron(B1,B2)).dot(G2.T))
            X3[i,:]=m3.T
        b3=np.linalg.solve(X3.T.dot(X3), X3.T.dot(y))
        B3=b3.reshape((p3,R3))
        
        XG = np.zeros((n,R1*R2*R3))
        for i in range(n):
            Xvec = np.ravel(unfold(X[i,:,:,:],0))
            m = np.kron(np.kron(B1,B2),B3).T.dot(Xvec)
            XG[i,:]=m
        bG = np.linalg.solve(XG.T.dot(XG), XG.T.dot(y))
        G = bG.reshape((R1,R2,R3))
        
        H_old=np.copy(H)
        H = np.tensordot(np.tensordot(np.tensordot(G,B1,axes=(0,1)),B2,axes=(0,1)),B3,axes = (0,1))
        for i in range(n):
            w1[i,0]=np.sum(X[i,:,:,:]*H)
        y = yt - w1
        
        X1=np.zeros((n,R1*p1))
        AG0 = unfold(AG,0)
        for i in range(n):
            M0=unfold(AX[i,:,:,:],0)
            m1 = np.ravel(M0.dot(np.kron(C2,C3)).dot(AG0.T))
            X1[i,:]=m1.T        
        c1=np.linalg.solve(X1.T.dot(X1), X1.T.dot(y))
        C1=c1.reshape((p1,R1))
        
        X2=np.zeros((n,R2*p2))
        AG1 = unfold(AG,1)
        for i in range(n):
            M1=unfold(AX[i,:,:,:],1)
            m2 = np.ravel(M1.dot(np.kron(C1,C3)).dot(AG1.T))
            X2[i,:]=m2.T       
        c2=np.linalg.solve(X2.T.dot(X2), X2.T.dot(y))
        C2=c2.reshape((p2,R2))
        
        X3=np.zeros((n,R3*p3))
        AG2 = unfold(AG,2)
        for i in range(n):
            M2=unfold(AX[i,:,:,:],2)
            m3 = np.ravel(M2.dot(np.kron(C1,C2)).dot(AG2.T))
            X3[i,:]=m3.T
        c3=np.linalg.solve(X3.T.dot(X3), X3.T.dot(y))
        C3=c3.reshape((p3,R3))
        
        XAG = np.zeros((n,R1*R2*R3))
        for i in range(n):
            AXvec = np.ravel(unfold(AX[i,:,:,:],0))
            m = np.kron(np.kron(C1,C2),C3).T.dot(AXvec)
            XAG[i,:] = m
        bAG = np.linalg.solve(XAG.T.dot(XAG),XAG.T.dot(y))
        AG = bAG.reshape(R1,R2,R3)
        
        CH_old = np.copy(CH)
        CH = np.tensordot(np.tensordot(np.tensordot(AG,C1,axes=(0,1)),C2,axes=(0,1)),C3,axes = (0,1))
        for i in range(n):
            w2[i,0]=np.sum(AX[i,:,:,:]*CH)
        y = y_0 - w1 - w2
        theta_old = np.copy(theta)
        theta = np.linalg.solve(c.T.dot(c), c.T.dot(y))
        
        mseB = norm(H-B)
        mseC = norm(CH-C)
        mse1 = norm(theta[:(l//2)]-theta_true[:(l//2)])
        mse2 = norm(theta[(-l//2):]-theta_true[(-l//2):])
        
        mse_old = mse
        mse = mseC
        if t>5 and abs(mse-mse_old)<tol:
            break 
        # mseB = norm(H-H_old)/max(1,norm(H_old))
        # mseC = norm(CH-CH_old)/max(1,norm(CH_old))
        # mse1 = norm(theta[:(l//2)]-theta_old[:(l//2)])/max(1,norm(theta_old[:(l//2)]))
        # mse2 = norm(theta[(-l//2):]-theta_old[(-l//2):])/max(1,norm(theta_old[(-l//2):]))
        # if mseC<tol:
        #     break 
    time_taken=time.time()-time_start
    return time_taken,mseB,mseC,mse1,mse2,H,CH,theta,0


# ---------------------------------------------------------
def generate_data(n, p, R, sigma, scenario='standard'):
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
    
    S = np.random.normal(0, 0.5, size=(p,p,p)) / p 
    gamma = np.array([0.5, -0.5, 0.5, -0.5, 0.0])
    #gamma = np.array([1.0, -1.0, 1.0, -1.0, 0.5])
    theta_true = np.ones((10,1))
    
    X = np.random.normal(0,1,size=(n,p,p,p))
    c1 = np.random.normal(0,1,size=(n,5))
    AX = np.zeros((n,p,p,p))
    c = np.zeros((n,10))
    y = np.zeros((n,1))
    A_list = np.zeros(n)
    
    for i in range(n):
        if scenario == 'A':
            logit_e = c1[i,:].dot(gamma) + 1.5 * (c1[i,0]**2 - 1)
        else:
            logit_e = c1[i,:].dot(gamma) + 2.0 * np.sum(X[i,:,:,:] * S) 
            
        e_i = expit(logit_e)
        a = np.random.binomial(1., e_i)
        
        A_list[i] = a
        c[i,0:5] = c1[i,:]
        c[i,5:10] = a * c1[i,:]
        AX[i,:,:,:] = a * X[i,:,:,:]
        
        main_effect = np.sum(X[i,:,:,:] * B) + c1[i,:].dot(theta_true[:5]).item()
        if scenario == 'A':
            main_effect += 20.0 * (c1[i,0]**2)
            
        interaction_effect = c1[i,:].dot(theta_true[5:10]).item() + np.sum(X[i,:,:,:] * C)
        
        y[i,0] = main_effect + a * interaction_effect + np.random.normal(0,sigma,1)[0]
        
    return X, c, AX, y, B, C, theta_true, A_list, c1


# ---------------------------------------------------------
def pi_opt(n, X_test, c1_test, theta_true1, theta_true2, B, C, scenario='standard'): 
    pi_action_opt = np.zeros(n)
    V_opt_vals = np.zeros(n)
    
    for i in range(n):
        if (np.sum(X_test[i]*C) + c1_test[i].dot(theta_true2).item()) > 0:
            pi_action_opt[i] = 1
            
        main_eff = np.sum(X_test[i]*B) + c1_test[i].dot(theta_true1).item()
        if scenario == 'A':
            main_eff += 20.0 * (c1_test[i,0]**2)
            
        adv_eff = np.sum(X_test[i]*C) + c1_test[i].dot(theta_true2).item()
        V_opt_vals[i] = main_eff + adv_eff if pi_action_opt[i] == 1 else main_eff
        
    return V_opt_vals, pi_action_opt

def pi_hat(n, X_test, c1_test, H, CH, theta, B_true, C_true, theta_true1, theta_true2, pi_action_opt, scenario='standard'):
    pi_action_hat = np.zeros(n)
    V_hat_vals = np.zeros(n)
    
    for i in range(n):
        if (np.sum(X_test[i]*CH) + c1_test[i].dot(theta[-5:]).item()) > 0:
            pi_action_hat[i] = 1
            
        main_eff = np.sum(X_test[i]*B_true) + c1_test[i].dot(theta_true1).item()
        if scenario == 'A':
            main_eff += 20.0 * (c1_test[i,0]**2)
            
        adv_eff = np.sum(X_test[i]*C_true) + c1_test[i].dot(theta_true2).item()
        V_hat_vals[i] = main_eff + adv_eff if pi_action_hat[i] == 1 else main_eff
            
    # Accuracy
    accuracy = np.mean(pi_action_hat == pi_action_opt)
    return np.mean(V_hat_vals), accuracy


# ---------------------------------------------------------
def replication_compare(seed, scenario='A'):
    np.random.seed(seed)
    n0 = 1000  
    R0 = 2  
    p = 10
    sigma = 0.5
    
    try:
        [X, c, AX, y, B, C, theta_true, A_list, c1] = generate_data(n=n0, p=p, R=R0, sigma=sigma, scenario=scenario)
        
        # Q-learning
        [_, _, _, _, _, H_Q, CH_Q, theta_Q, _] = tensor_r_tucker(X, c, AX, y, B, C, theta_true, R0, R0, R0, tol=0.001, MAX_ITER=100)

        # A-learning 
        if scenario == 'A':
            poly = PolynomialFeatures(degree=2, include_bias=False)
            c1_poly = poly.fit_transform(c1)
            clf = LogisticRegression(max_iter=1000).fit(c1_poly, A_list)
            e_hat = clf.predict_proba(c1_poly)[:, 1]
        else:
            clf = LogisticRegression(max_iter=1000).fit(c1, A_list)
            e_hat = clf.predict_proba(c1)[:, 1]

        e_hat = np.clip(e_hat, 0.05, 0.95)
        
        c_Alearn = np.zeros((n0, 10))
        AX_Alearn = np.zeros((n0, p, p, p))
        for i in range(n0):
            c_Alearn[i, 0:5] = c1[i, :]
            c_Alearn[i, 5:10] = (A_list[i] - e_hat[i]) * c1[i, :]
            AX_Alearn[i, :, :, :] = (A_list[i] - e_hat[i]) * X[i, :, :, :]
            
        [_, _, _, _, _, H_A, CH_A, theta_A, _] = tensor_r_tucker(X, c_Alearn, AX_Alearn, y, B, C, theta_true, R0, R0, R0, tol=0.001, MAX_ITER=100)

        # Test Set
        n_test = int(0.4 * n0)
        [X_test, c_test, AX_test, y_test, B_test, C_test, theta_true_test, _, c1_test] = generate_data(n=n_test, p=p, R=R0, sigma=sigma, scenario=scenario)
        
        theta_true1 = np.ones((5,1))
        theta_true2 = np.ones((5,1))
        
        # Oracle
        V_opt_vals, pi_action_opt = pi_opt(n_test, X_test, c1_test, theta_true1, theta_true2, B, C, scenario)
        V_opt_mean = np.mean(V_opt_vals)
        
        # Q-learning
        V_Q, acc_Q = pi_hat(n_test, X_test, c1_test, H_Q, CH_Q, theta_Q, B, C, theta_true1, theta_true2, pi_action_opt, scenario)
        
        # A-learning
        V_A, acc_A = pi_hat(n_test, X_test, c1_test, H_A, CH_A, theta_A, B, C, theta_true1, theta_true2, pi_action_opt, scenario)
        
        return V_opt_mean, V_Q, V_A, acc_Q, acc_A
        
    except Exception as e:
        print(f"\n[Error in Seed {seed}]: {e}")
        traceback.print_exc()
        return None

if __name__ == '__main__':
    np.random.seed(2024)
    b = 100
    seeds = np.random.randint(1, 10000, size=b, dtype=int)
    
    for scen in ['A','B']:
        print(f"\n--- Running Scenario {scen} ---")
        results = []
        pool = multiprocessing.Pool(min(4, multiprocessing.cpu_count()))
        
        #pool = multiprocessing.Pool(multiprocessing.cpu_count())
        
        for i in range(b):
            res = pool.apply_async(replication_compare, (seeds[i], scen))
            results.append(res)
        
        final_results = [r.get() for r in results if r.get() is not None]
        pool.close()
        pool.join()
        
        if len(final_results) > 0:
            V_opt_mean = np.mean([r[0] for r in final_results])
            V_Q_mean = np.mean([r[1] for r in final_results])
            V_A_mean = np.mean([r[2] for r in final_results])
            acc_Q_mean = np.mean([r[3] for r in final_results])
            acc_A_mean = np.mean([r[4] for r in final_results])
            
            print(f"Scenario {scen} - Oracle V: {V_opt_mean:.3f}")
            print(f"              Q-learning -> V: {V_Q_mean:.3f}, Acc: {acc_Q_mean:.3f}")
            print(f"              A-learning -> V: {V_A_mean:.3f}, Acc: {acc_A_mean:.3f}")