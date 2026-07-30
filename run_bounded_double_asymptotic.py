#!/usr/bin/env python3
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import betaln
from scipy.optimize import minimize_scalar,brentq

OUT=Path(__file__).resolve().parent
RNG=np.random.default_rng(20260725)
a,b=.3,.5
I=b*math.log(b/a)+(1-b)*math.log((1-b)/(1-a))
LOGB0=betaln(.5,.5)

def kl(p,q):return p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q))

def log_m(n,S,theta):
    return betaln(S+.5,n-S+.5)-LOGB0-S*np.log(theta)-(n-S)*np.log1p(-theta)

def min_log_two(n0,S0,n1,S1):
    mean0=S0/n0;mean1=S1/n1
    lo=np.clip(np.minimum(mean0,mean1),1e-10,1-1e-10)
    hi=np.clip(np.maximum(mean0,mean1),1e-10,1-1e-10)
    for _ in range(30):
        theta=(lo+hi)/2
        z0=log_m(n0,S0,theta);z1=log_m(n1,S1,theta);mx=np.maximum(z0,z1)
        w0=np.exp(z0-mx);w1=np.exp(z1-mx)
        g0=n0*(theta-mean0)/(theta*(1-theta));g1=n1*(theta-mean1)/(theta*(1-theta))
        grad=(w0*g0+w1*g1)/(w0+w1)
        lo=np.where(grad<0,theta,lo);hi=np.where(grad>=0,theta,hi)
    theta=(lo+hi)/2
    return np.logaddexp(log_m(n0,S0,theta),log_m(n1,S1,theta))

def j_cross(c):
    r=(c*a+b)/(c+1.)
    ans=minimize_scalar(lambda th:max(kl(b,th),(c+1.)*kl(r,th)),bounds=(1e-10,1-1e-10),method='bounded',options={'xatol':1e-12})
    return ans.fun

def finite_prediction(T,B):
    f=lambda d:d*j_cross(T/d)-B
    lo=1e-8;hi=max(10.,4*B/I)
    while f(hi)<0:hi*=2
    return brentq(f,lo,hi)

def run(A,reps):
    B=math.log(A);d0=B/I
    T=math.ceil((B**2/5)*d0)
    horizon=math.ceil(4*d0+600)
    spre=RNG.binomial(T,a,size=reps)
    post=np.cumsum(RNG.binomial(1,b,size=(reps,horizon)),axis=1)
    stop=np.full(reps,horizon+1,int);active=np.ones(reps,bool)
    for d in range(1,horizon+1):
        s0=post[:,d-1]
        value=min_log_two(d,s0,T+d,spre+s0)
        hit=active&(value>=B);stop[hit]=d;active[hit]=False
        if not active.any():break
    detected=stop<=horizon;delay=stop[detected]
    dft=finite_prediction(T,B)
    return dict(A=A,logA=B,T=T,T_over_d0=T/d0,reps=reps,detection_fraction=detected.mean(),information=I,
                first_order=d0,finite_T_prediction=dft,median=float(np.median(delay)),mean=float(np.mean(delay)),
                q10=float(np.quantile(delay,.1)),q90=float(np.quantile(delay,.9)),ratio_first=float(np.median(delay)/d0),
                ratio_finite=float(np.median(delay)/dft))

def main():
    levels=[1e5,1e8,1e12,1e20];reps=[1200,1200,1000,600]
    rows=[]
    for A,r in zip(levels,reps):
        rows.append(run(A,r));pd.DataFrame(rows).to_csv(OUT/'bounded_double_asymptotic_new.csv',index=False);print(rows[-1],flush=True)
if __name__=='__main__':main()
