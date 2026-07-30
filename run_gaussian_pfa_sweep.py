#!/usr/bin/env python3
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar,brentq
from gaussian_simulation_utils import sample_gauss,gmax_stops

OUT=Path(__file__).resolve().parent
RNG=np.random.default_rng(20260801)
alpha=1e-8
L=math.log(1/alpha)
I=.5*math.log(2.0)

def logpi(s:int)->float:
    return math.log(.5)-math.log(s)-2*math.log(math.log(math.e*s))

def solve_T(c:float)->tuple[int,float]:
    T=1000.0
    for _ in range(200):
        B=L-logpi(int(round(T))+1)
        new=c*B/I
        if abs(new-T)<1e-8:break
        T=.5*T+.5*new
    Ti=int(round(T));d0=(L-logpi(Ti+1))/I
    return Ti,d0

def J(m,theta):return .5*math.log1p((m-theta)**2)

def finite_value(T,d):
    r=minimize_scalar(lambda th:max(logpi(1)+T*J(0.,th),logpi(T+1)+d*J(1.,th)),bounds=(-1.,2.),method='bounded',options={'xatol':1e-12})
    return r.fun

def finite_prediction(T,d0):
    f=lambda d:finite_value(T,d)-L
    lo=1e-8;hi=2*d0
    while f(hi)<0:hi*=2
    return brentq(f,lo,hi,xtol=1e-10)

def main():
    rows=[]
    for c,reps in [(15,100),(30,80),(60,60)]:
        T,d0=solve_T(c);dp=finite_prediction(T,d0)
        post=int(math.ceil(3.0*dp+120));H=T+post
        x=sample_gauss(RNG,reps,H,T,0.,1.,1.,1.)
        st=gmax_stops(x,1,L,1.,T);det=st<=H;d=st[det]-T
        rows.append(dict(alpha=alpha,L=L,T_over_d0=c,T=T,reps=reps,detection_fraction=det.mean(),information=I,
                         effective_boundary=L-logpi(T+1),first_order=d0,finite_T_prediction=dp,
                         median=float(np.median(d)),q10=float(np.quantile(d,.1)),q90=float(np.quantile(d,.9)),
                         ratio_first=float(np.median(d)/d0),ratio_finite=float(np.median(d)/dp)))
        pd.DataFrame(rows).to_csv(OUT/'gaussian_pfa_sweep_new.csv',index=False)
        print(rows[-1],flush=True)
if __name__=='__main__':main()
