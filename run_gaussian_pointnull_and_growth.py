#!/usr/bin/env python3
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from gaussian_simulation_utils import sample_gauss,gaussian_oracle_paths,first_crossing

OUT=Path(__file__).resolve().parent
RNG=np.random.default_rng(20260730)
I=.5*math.log(2.0)

def main():
    # Point-null threshold convergence.
    levels=[2e4,1e8,1e20,1e40]
    Bmax=math.log(2*max(levels))
    H=int(math.ceil(1.6*(Bmax+12)/I))
    reps=2500
    x=sample_gauss(RNG,reps,H,0,0.0,1.0,1.0,1.0)
    ui,g,_=gaussian_oracle_paths(x,theta=0.0,c=1.0,nullvar=1.0)
    rows=[]
    for A in levels:
        for method,paths,B in [('UI-t',ui,math.log(A)),('reduced-G',g,math.log(2*A))]:
            st=first_crossing(paths,B)
            d=st[st<=H]
            pred=B/I
            rows.append(dict(A=A,method=method,reps=reps,information=I,boundary=B,prediction=pred,
                             detection_fraction=len(d)/reps,median=float(np.median(d)),
                             q10=float(np.quantile(d,.1)),q90=float(np.quantile(d,.9)),ratio=float(np.median(d)/pred)))
    pd.DataFrame(rows).to_csv(OUT/'gaussian_point_null_new.csv',index=False)

    # Fixed-block growth, including variance change.
    scenarios=[('N(1,1)',1.0,1.0),('N(1.5,1)',1.5,1.0),('N(0.75,2.25)',0.75,2.25)]
    n=700;reps=300
    gr=[]
    for label,b,v in scenarios:
        x=sample_gauss(RNG,reps,n,0,0.0,1.0,b,v)
        ui,g,full=gaussian_oracle_paths(x,theta=0.0,c=1.0,nullvar=1.0)
        It=.5*math.log1p(b*b/v)
        KL=.5*(math.log(1.0/v)+(v+b*b)-1.0)
        for method,arr,theory in [('UI-t',ui,It),('reduced-G',g,It),('full-Gaussian',full,KL)]:
            vals=arr[:,-1]/n
            gr.append(dict(scenario=label,method=method,n=n,reps=reps,empirical_rate=float(vals.mean()),
                           se=float(vals.std(ddof=1)/math.sqrt(reps)),theory_rate=theory,ratio=float(vals.mean()/theory)))
    pd.DataFrame(gr).to_csv(OUT/'gaussian_growth_new.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(gr).to_string(index=False))

if __name__=='__main__': main()
