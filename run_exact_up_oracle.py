#!/usr/bin/env python3
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import betaln, logsumexp
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from scipy.stats import beta as beta_dist

OUT=Path(__file__).resolve().parent
RNG=np.random.default_rng(20260728)
a=0.3; b=0.5
levels=np.array([1e5,1e8,1e12,1e20],float)
B=np.log(levels)


def binary_kl(p,q):
    return p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q))

def ibet_beta(concentration=20.0):
    aa,bb=concentration*b,concentration*(1-b)
    lo,hi=-1/(1-a),1/a
    def growth(lam):
        return quad(lambda x: beta_dist.pdf(x,aa,bb)*math.log1p(lam*(x-a)),0,1,limit=300,epsabs=1e-11,epsrel=1e-11)[0]
    r=minimize_scalar(lambda z:-growth(z),bounds=(lo+1e-11,hi-1e-11),method='bounded',options={'xatol':1e-12})
    return -r.fun

def ibet_twopoint(h=0.05):
    z=np.array([b-h,b+h])
    lo,hi=-1/(1-a),1/a
    def growth(lam): return float(np.mean(np.log1p(lam*(z-a))))
    r=minimize_scalar(lambda z:-growth(z),bounds=(lo+1e-11,hi-1e-11),method='bounded',options={'xatol':1e-12})
    return -r.fun

def first_crossing(logpaths,boundary):
    hit=logpaths>=boundary
    anyh=hit.any(axis=1)
    out=np.full(logpaths.shape[0],logpaths.shape[1]+1,dtype=int)
    out[anyh]=hit[anyh].argmax(axis=1)+1
    return out

def bernoulli_exact_paths(reps,H):
    x=RNG.binomial(1,b,size=(reps,H))
    S=np.cumsum(x,axis=1)
    n=np.arange(1,H+1)[None,:]
    return (betaln(S+0.5,n-S+0.5)-betaln(0.5,0.5)
            -S*math.log(a)-(n-S)*math.log1p(-a))

def exact_up_paths(x,H):
    # Under the Jeffreys prior, K-point Gauss--Chebyshev quadrature is exact
    # for polynomials through degree 2K-1.  The time-n UP integrand is a
    # degree-n polynomial in p, so K >= (H+1)/2 is exact for every n <= H.
    K=(H+2)//2
    j=np.arange(1,K+1,dtype=float)
    p=(1+np.cos((2*j-1)*np.pi/(2*K)))/2
    comp=np.zeros((x.shape[0],K),dtype=float)
    out=np.empty((x.shape[0],H),dtype=float)
    pp=p[None,:]
    for t in range(H):
        z=x[:,t,None]
        fac=z*pp/a+(1-z)*(1-pp)/(1-a)
        comp += np.log(fac)
        out[:,t]=logsumexp(comp,axis=1)-math.log(K)
    return out,K

def summarize(family,info,paths,K):
    rows=[]
    H=paths.shape[1]
    for A,boundary in zip(levels,B):
        st=first_crossing(paths,boundary)
        det=st<=H
        d=st[det]
        pred=boundary/info
        rows.append(dict(family=family,A=A,log_boundary=boundary,reps=len(st),quadrature_nodes=K,
                         detection_fraction=det.mean(),information=info,prediction=pred,
                         median=float(np.median(d)),q10=float(np.quantile(d,.1)),q90=float(np.quantile(d,.9)),
                         ratio=float(np.median(d)/pred)))
    return rows

def main():
    rows=[]
    # Bernoulli: UP and beta mixture coincide exactly.
    info_bern=binary_kl(b,a)
    H_bern=1400; reps_bern=4000
    rows += summarize('Bernoulli(0.5)',info_bern,bernoulli_exact_paths(reps_bern,H_bern),0)

    # Low-variance continuous and discrete alternatives: exact continuous UP.
    H=220; reps=4000
    x=RNG.beta(10,10,size=(reps,H))
    paths,K=exact_up_paths(x,H)
    rows += summarize('Beta(10,10)',ibet_beta(),paths,K)

    x=b+0.05*(2*RNG.integers(0,2,size=(reps,H))-1)
    paths,K=exact_up_paths(x,H)
    rows += summarize('two-point {0.45,0.55}',ibet_twopoint(),paths,K)

    df=pd.DataFrame(rows)
    df.to_csv(OUT/'exact_up_oracle.csv',index=False)
    print(df.to_string(index=False))

if __name__=='__main__': main()
