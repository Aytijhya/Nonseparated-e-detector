#!/usr/bin/env python3
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar,brentq
from gaussian_simulation_utils import sample_gauss,gmax_stops

OUT=Path(__file__).resolve().parent
RNG=np.random.default_rng(20260731)
A=1e12
B=math.log(2*A)
I=.5*math.log(2.0)
d0=B/I

def J(m,theta): return .5*math.log1p((m-theta)**2)

def finite_value(T,d):
    r=minimize_scalar(lambda th:max(T*J(0.0,th),d*J(1.0,th)),bounds=(-1.,2.),method='bounded',options={'xatol':1e-12})
    return r.fun

def finite_prediction(T):
    f=lambda d:finite_value(T,d)-B
    lo=1e-8;hi=2*d0
    while f(hi)<0:hi*=2
    return brentq(f,lo,hi,xtol=1e-10)

def main():
    rows=[]
    for c,reps in [(15,100),(30,80),(60,60),(120,40)]:
        T=int(round(c*d0))
        dp=finite_prediction(T)
        post=int(math.ceil(3.0*dp+120))
        H=T+post
        x=sample_gauss(RNG,reps,H,T,0.,1.,1.,1.)
        st=gmax_stops(x,0,B,1.,T)
        det=st<=H;d=st[det]-T
        rows.append(dict(A=A,T_over_d0=c,T=T,reps=reps,detection_fraction=det.mean(),information=I,
                         boundary=B,first_order=d0,finite_T_prediction=dp,median=float(np.median(d)),
                         q10=float(np.quantile(d,.1)),q90=float(np.quantile(d,.9)),
                         ratio_first=float(np.median(d)/d0),ratio_finite=float(np.median(d)/dp)))
        pd.DataFrame(rows).to_csv(OUT/'gaussian_arl_sweep_new.csv',index=False)
        print(rows[-1],flush=True)

if __name__=='__main__':main()
