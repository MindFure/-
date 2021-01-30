import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm.notebook import tqdm
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_log_error, mean_squared_error

def dS_dt(a_t,tau,alf_I,S,I,E,R,N,alf_E,y):
    return -(((5-(a_t-tau))/5)*((((alf_I*S*I)/N))+(((alf_E*S*E)/N)))+y*R)



def dE_dt(a_t,tau,alf_I,S,I,E,N,k,p,alf_E):
    return ((5-(a_t-tau))/5)*(((alf_I*S*I)/N)+((alf_E*S*I)/N))-(k+p)*E



def dI_dt(k,E,b,u,I):
    return (k*E-(b+u)*I)


def dR_dt(b,I,p,E,y_s,R,eps_h_r,H):
    return (b*I+p*E-y_s*R+eps_h_r*H)


def dH_dt(u,I,eps_c_h,C,eps_h_c,eps_h_r,H):
    return (u*I+eps_c_h*C-(eps_h_c+eps_h_r)*H)


def dC_dt(eps_h_c,H,eps_c_h,C,m):
    return (eps_h_c*H-(eps_c_h+m)*C)


def dD_dt(m,C):
    return (m*C)


def SEIR_HCD(t,y,a_t=2,tau=2,alf_I=0.5,alf_E=0.5,k=0.5,p=0,b=0.5,y_s=0,u=0.5,eps_h_r=0.225,eps_h_c=0.025,eps_c_h=0.5,m=0.0001):
      
    S, E, I, R, H, C, D = y
    
    S_0 = dS_dt(S,I,a_t,tau,alf_I,E,R,N,alf_E,y_s)
    E_0 = dE_dt(a_t,tau,alf_I,S,I,E,N,k,p,alf_E)
    I_0 = dI_dt(k,E,b,u,I)
    R_0 = dR_dt(b,I,p,E,y_s,R,eps_h_r,H)
    H_0 = dH_dt(u,I,eps_c_h,C,eps_h_c,eps_h_r,H)
    C_0 = dC_dt(eps_h_c,H,eps_c_h,C,m)
    D_0 = dD_dt(m,C)
    
    return (S_0, E_0, I_0, R_0, H_0, C_0, D_0)

N = 100000  #Количество человек
N_inf = 10000 # кол-во инфицированных
max_days = 365 #дней с момента заражения
init_stat = [(N - N_inf)/ N, 500, N_inf / N, 0, 0, 0, 0]
stat = solve_ivp(SEIR_HCD, [0, max_days], init_stat)

print(stat)



